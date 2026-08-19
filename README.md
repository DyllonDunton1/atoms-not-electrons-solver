# Atoms Not Electrons Solver

A Python solver for Tutor Intelligence's **Atoms Not Electrons** warehouse optimization challenge.

Challenge site: https://www.atomsnotelectrons.com

## The challenge

The warehouse is a **60 x 40 grid** containing 5 robots, 100 SKU types, 240 pallets, and 1,000 orders. Fulfillment happens on `y = 0`; replenishment happens on `y = 39`.

Each robot can perform at most one action per timestep. Robots move one orthogonal grid cell at a time, pick items from adjacent pallets, dock to pallets, move docked pallets as a rigid footprint, replenish them at the bottom row, and fulfill completed orders at the top row.

The score is the total number of timesteps required to fulfill all 1,000 orders. **Lower is better.**

## Submission format

The solver writes actions as:

```text
<timestep> <robot_id> <action> <x> <y>
```

Missing robot actions are waits. Generated schedules are replayed through the local simulator before being treated as valid outputs.

## Current architecture

The project keeps warehouse rules, spatial pathfinding, fleet traffic, task assignment, and collection strategy separate.

1. `parser.py` reads `BIG_ORDER.txt` into the mutable `WorldState` model.
2. `Simulator` is the final authority on action legality and state mutation.
3. `PathPlanner` performs static four-connected A* with complete robot + docked-pallet footprints.
4. `MultiRobotSolver` assigns FIFO orders and resolves only the first fleet move each timestep.
5. `AisleAwareSolver` adds batched collection, claims, refill/resume behavior, and persistent neighboring-pallet coordination.
6. Collection planners decide **what** to service; the existing pathfinder still decides **how to move** through the warehouse.

There is deliberately no multi-timestep prediction of future robot positions in the traffic layer. Every timestep starts again from the new real world state.

## Fleet traffic priority

Traffic priority is dynamic:

```python
(-num_docked_pallets, robot_id)
```

Robots carrying more docked pallets plan first. If pallet counts are equal, lower robot ID keeps the original priority.

This means a refill robot carrying a rigid pallet assembly is not forced to yield to an empty robot simply because its numeric ID is larger. The complete current and committed footprint of every higher-priority robot shapes lower-priority routes. Real current occupancy is still checked before a first move executes.

If two robots carry the same number of pallets, the current implementation falls back to robot ID. More sophisticated pallet-versus-pallet conflict handling is intentionally deferred until a reproducible case requires it.

## Existing 12-island aisle strategy

`src/aisles.py` and `src/aisle_solver.py` implement the original aisle-aware collection strategy.

- The 240 pallet homes are grouped into 12 connected 2 x 10 physical pallet islands.
- Aisles are scored using useful work, route distance, and congestion.
- Detailed service routes use static A* plus deterministic stop ordering.
- Pallets are claimed only for the active stop.
- Refill is a subroutine of the current collection visit.
- The previously visited aisle is omitted from the next normal choice, with a fallback that allows it again if it is the only useful option.

A distinct-SKU scoring experiment improved the full 5-robot / 1,000-order result to **64,090 timesteps** with **219,088 collection moves**. That result is the current measured reference before the directed-column experiment.

## Experimental 24-column strategy

`src/column_solver.py` leaves the existing pathfinding and traffic machinery untouched and changes only the collection abstraction.

Each 2 x 10 physical island is split into two exposed 10-pallet service columns, producing **24 service columns** total. Each column owns one dedicated service lane on its exposed side.

At every new collection decision, the planner evaluates up to:

```text
24 columns x 2 directions = 48 directed routes
```

There is no cheap shortlist. Every useful directed candidate is evaluated directly.

For one selected column:

- one useful SKU contributes one unit of utility regardless of required item quantity;
- duplicated SKUs prefer a stock-sufficient pallet when available;
- all pickups lie on the dedicated exposed-side lane;
- `up` plans order stops by decreasing `y`;
- `down` plans order stops by increasing `y`;
- A* computes the approach from the robot's current position to the first useful stop;
- the in-column traversal is monotonic and therefore has exact distance `abs(last_y - first_y)`;
- a candidate is rejected if the straight service-lane segment is not actually clear in the current static geometry;
- congestion is tracked per service column.

The directed route score is currently:

```text
                 distinct useful SKUs
score = ------------------------------------------
        approach + column span + 1 + congestion
```

where congestion is converted to distance using the same configured penalty used by the aisle planner.

### No final same-column backtracking

The directed-column solver intentionally removes the old final same-aisle rescan.

Once a monotonic column pass reaches its last stored stop, the robot leaves that column immediately and performs a new global 48-route decision. A hard-unavailable stop is deferred rather than causing the robot to reverse direction.

The just-finished column is excluded from the next normal choice. The fallback may choose it again only when no other useful column exists. That is the deliberate **must backtrack** case.

### Previous-column adjacency policy

Persistent robot-pallet adjacency is deliberately **not** used to reject normal 48-route candidates. A robot may be many timesteps away from a promising column, so another robot merely standing beside one of its pallets now should not make that route look bad. Claims, docked pallets, and moved pallets remain hard exclusions everywhere.

Persistent adjacency is consulted only when the solver is about to fall back into the **previous column** because no other useful column exists. If the useful previous-column pallet is still persistently occupied by a higher-priority robot, the solver waits for a new world timestep instead of immediately re-entering the same column. Once the blocker clears, the previous column can be chosen as the required backtrack.

## Why the column experiment exists

The physical warehouse geometry makes a single exposed pallet column almost one-dimensional. Once the entry side and direction are selected, useful stops have a deterministic monotonic order. This removes the nearest-neighbor/TSP-like part of the old in-aisle planner while preserving the existing A* traffic and rigid-footprint logic.

The experiment is intended to answer whether a simpler collection representation reduces travel and head-on interactions enough to beat the 64,090-timestep distinct-SKU reference.

## Running experiments

Known-good distinct-SKU experiment:

```bash
python3 manual_tests/distinct_sku_experiment.py
```

Directed-column experiment:

```bash
python3 manual_tests/column_solver_experiment.py
```

A partial directed-column run through timestep 8,000:

```bash
python3 manual_tests/column_solver_experiment.py --stop-timestep 8000
```

The column runner uses distinct output names beginning with:

```text
column_v1_distinct_skus_...
```

It also writes a full JSONL diagnostic trace and automatically runs the streaming traffic-trace analyzer.

## Architecture

```text
atoms-not-electrons-solver/
├── README.md
├── DEVELOPMENT_MANIFEST.md
├── source_material/
│   ├── BIG_ORDER.txt
│   └── CHALLENGE_README.md
├── src/
│   ├── models.py
│   ├── parser.py
│   ├── world.py
│   ├── pathfinding.py
│   ├── scheduler.py
│   ├── simulator.py
│   ├── tasks.py
│   ├── allocator.py
│   ├── solver.py
│   ├── multi_robot_solver.py
│   ├── aisles.py
│   ├── aisle_solver.py
│   ├── column_solver.py
│   ├── metrics.py
│   └── writer.py
├── tests/
│   ├── README.md
│   ├── test_column_solver.py
│   └── test_*.py
├── manual_tests/
│   ├── aisle_solver_smoke_test.py
│   ├── distinct_sku_experiment.py
│   ├── column_solver_experiment.py
│   └── analyze_traffic_trace.py
└── outputs/
```

## Module responsibilities

- **`models.py`** — Core robot, pallet, order, and action data classes.
- **`parser.py`** — Challenge-input parser.
- **`world.py`** — Warehouse state and invariants.
- **`pathfinding.py`** — Static A* for arbitrary rigid robot footprints.
- **`scheduler.py`** — One-timestep reservation primitives.
- **`simulator.py`** — Final action-legality authority.
- **`tasks.py` / `allocator.py`** — FIFO task representation and assignment.
- **`solver.py`** — Original single-robot baseline.
- **`multi_robot_solver.py`** — Five-robot FIFO execution and pallet-aware one-step traffic priority.
- **`aisles.py`** — Original 12-island geometry/scoring/service-route planner.
- **`aisle_solver.py`** — Shared aisle-aware collection state machine and refill behavior.
- **`column_solver.py`** — Experimental 24-column, 48-directed-route collection strategy.
- **`metrics.py`** — Schedule, movement, wait, and aisle-performance reporting.
- **`writer.py`** — Challenge submission serialization.

## Testing

Run the full automated suite from the repository root:

```bash
python3 -m unittest discover -s tests
```

Run the new strategy tests directly:

```bash
python3 -m unittest tests.test_column_solver -v
```

See [`tests/README.md`](tests/README.md) for the file-by-file guide.

## Performance history

| Strategy | Makespan | Total moves | Collection moves |
| --- | ---: | ---: | ---: |
| Original full FIFO baseline | 161,470 | 734,235 | 703,548 |
| Earlier aisle-aware reference | 67,840 | 263,139 | 229,603 |
| Previous-aisle exclusion baseline | 66,598 | 263,750 | 229,903 |
| Distinct-SKU + pallet-aware priority | **64,090** | **251,462** | **219,088** |

The 24-column strategy is a new experiment and should not replace the 64,090 reference until it passes the regression suite, fresh replay validation, and a complete 1,000-order benchmark.