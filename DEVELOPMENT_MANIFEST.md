# Development Manifest

Concise design notes, invariants, benchmark history, and current optimization boundaries for the solver.

## Current architecture

- `WorldState` owns mutable warehouse state and invariants.
- `Simulator` is the authority on action legality and state mutation.
- `PathPlanner` performs static Manhattan A* with arbitrary rigid robot footprints.
- `ReservationTable` protects only first moves already committed during the current timestep.
- `MultiRobotSolver` runs FIFO orders and pallet-aware one-step fleet traffic.
- `AislePlanner` / `AisleAwareSolver` implement the original 12-island collection strategy.
- `DirectedColumnPlanner` / `ColumnAwareSolver` implement the experimental 24-column collection strategy while reusing the same spatial pathfinding, traffic, refill, and simulator layers.
- `metrics.py` measures generated schedules independently of collection strategy.

## Non-negotiable correctness rules

- Warehouse size is 60 x 40; fulfillment is `y=0`; replenishment is `y=39`.
- A robot may move only one orthogonal cell per timestep.
- Missing robot action means wait.
- The simulator validates the whole timestep before mutating state.
- A robot may not enter a cell occupied at the start of the timestep even when that entity intends to leave during the same timestep.
- Docked pallets remain rigidly attached at cardinal offsets and are part of the robot footprint.
- Every movement/path legality check uses the complete rigid footprint, not only the robot center.
- Head-on use of the same movement edge is rejected.
- Pallet home cells remain warehouse structure even while a pallet is carried away.
- Robot centers never route onto pallet home cells.
- Exact order multisets are required for fulfillment.
- Replenishment happens automatically at the end of a timestep when a robot carrying docked pallets is on `y=39`.

## Spatial path planning

`PathPlanner` remains deliberately spatial rather than predictive.

- Four-connected Manhattan A*.
- Complete supplied footprint checked at every candidate robot-center location.
- Static pallet homes always remain blocked for robot centers.
- Moved undocked pallets block both home and current positions.
- Pallets represented by a moving rigid footprint may be omitted as independent obstacles through `ignored_pallet_ids`.
- Temporary robot blockers are supplied by the fleet layer.
- No future-time robot states are embedded in spatial A*.

The directed-column experiment does **not** change `src/pathfinding.py`.

## One-step fleet traffic model

Traffic is rebuilt from the real world state every timestep.

### Priority key

Current priority is:

```python
(-num_docked_pallets, robot_id)
```

Lower tuple value plans first.

Consequences:

- a robot carrying one pallet plans before every robot carrying none;
- equal pallet counts fall back to lower robot ID;
- the current both-robots-carrying-pallets behavior is intentionally only the tuple fallback until a concrete failure justifies a richer rule.

### Per-timestep sequence

1. Sort active robots by the dynamic priority key.
2. Each moving robot computes a complete spatial path to its current goal.
3. Only the first step may be committed.
4. Finished/inactive robots are permanent obstacles for everyone.
5. Higher-priority active robots shape a lower-priority robot's route using both current rigid footprints and committed destination footprints.
6. Lower-priority active robots are omitted from a higher-priority robot's preferred-route shaping.
7. Omitting lower-priority traffic does not remove physical occupancy: the first-step destination is still checked against every real current rigid footprint.
8. If the preferred first step is still physically occupied, the robot waits and replans next timestep instead of speculatively predicting future clearance.
9. `ReservationTable` protects only current-timestep cell/edge/footprint commitments.
10. After the simulator advances, all traffic route decisions are discarded.

This model intentionally has no generic forced-yield maneuver and no multi-step trajectory reservation.

## Waiting semantics

A wait means: **there is no safe first move worth committing from the current snapshot; remain in place and recompute next timestep.**

Typical causes:

- preferred first step physically occupied by lower-priority traffic that has not cleared yet;
- no complete current spatial route around higher-priority rigid blockers;
- an active picker temporarily occupies a cell another robot would prefer to use.

## FIFO task execution

- Unfulfilled orders live in a global FIFO queue.
- Free robots receive work deterministically.
- Each order is assigned exactly once.
- Replenishment is a private subroutine of the robot's active order.
- Finished robots become permanent obstacles only when no future work remains for them.

## Picking, docking, and replenishment

- Pick requires orthogonal adjacency.
- Docked pallets store owner and cardinal relative offset and move rigidly with the robot.
- Refill is dock -> travel to `y=39` -> automatic refill -> return pallet home -> undock -> resume collection.
- Refill planning rejects geometry that would place a carried pallet outside the grid at replenishment.

## Original 12-island collection strategy

`src/aisles.py` + `src/aisle_solver.py` remain intact as the existing strategy.

- 240 pallet homes grouped into 12 connected 2 x 10 islands.
- Cheap aisle scoring followed by detailed route evaluation.
- Deterministic nearest-useful-stop service ordering.
- Pallets claimed only for the active stop.
- Active aisle may be replanned from the robot's current position when future stops become unavailable.
- A final same-aisle rescan can recover deferred useful pallets before leaving.
- Previous aisle is excluded from the next normal choice, with a fallback if it is the only useful aisle.
- Refill preserves the active aisle commitment.

### Distinct-SKU scoring result

The distinct-SKU experiment changes utility from total required quantity to number of distinct useful SKUs. The latest completed 5-robot / 1,000-order reference is:

- makespan: **64,090**
- total moves: **251,462**
- collection moves: **219,088**
- waits: **2,976**
- refill trips: **253**
- physical aisle visits: **8,444**
- physical aisle re-entries: **34**

This is the benchmark the directed-column strategy must beat.

## Experimental 24-column collection strategy

`src/column_solver.py` is deliberately isolated from the existing 12-island implementation.

### Geometry

Each physical 2 x 10 pallet island is split into two exposed 10-pallet service columns:

```text
12 physical islands x 2 exposed sides = 24 service columns
```

Each service column has:

- 10 pallet homes at one fixed pallet `x`;
- one dedicated exposed-side service lane at adjacent fixed `x`;
- deterministic mapping from pallet ID to column ID.

### Global planning

At each new collection decision, evaluate up to:

```text
24 columns x 2 directions = 48 directed candidates
```

There is no cheap shortlist in this strategy.

For every useful column, evaluate both:

- `up`: useful stops ordered by decreasing `y`;
- `down`: useful stops ordered by increasing `y`.

Distinct useful SKU count is the utility numerator. If one SKU appears more than once in a column, stock-sufficient options are preferred; within the selected direction, the first preferred occurrence is used.

### Route cost

The existing static A* computes the approach from the current robot location to the first stop of the candidate directed pass.

The approach is prevented from entering the future service segment from the wrong end. After reaching the first stop, service is monotonic along one fixed `x`, so the exact in-column traversal distance is:

```text
abs(last_useful_y - first_useful_y)
```

The candidate is rejected if the straight service-lane segment is not truly clear in the current static geometry.

Current score:

```text
                 distinct useful SKUs
score = ------------------------------------------
        approach + column span + 1 + congestion
```

with congestion converted to distance using the configured congestion penalty.

### Monotonic execution / no final backtrack

The directed-column strategy intentionally removes the old final same-unit rescan.

- Once a direction is chosen, future stops keep that monotonic order.
- The strategy does not rerun greedy nearest-neighbor ordering every timestep.
- If a future stop becomes unavailable, it is deferred/skipped instead of reversing the pass.
- Refill of an already-active stop remains allowed and preserves the current pass.
- Reaching the final stored stop ends the pass immediately.
- Remaining work is sent back to the global 48-route decision.
- The just-finished column is excluded from the next normal decision.
- The inherited fallback may select the same previous column again only if no other useful column exists. This is the intentional **must backtrack** case.

### What remains unchanged

The column experiment does not alter:

- static A* implementation;
- rigid-footprint representation;
- one-step traffic scheduler;
- pallet-aware fleet priority;
- simulator rules;
- FIFO task assignment;
- refill mechanics;
- fulfillment logic.

Only collection grouping, route candidate generation, scoring, and stop order are changed.

## Diagnostic runners

Known-good distinct-SKU runner:

```bash
python3 manual_tests/distinct_sku_experiment.py
```

Directed-column runner:

```bash
python3 manual_tests/column_solver_experiment.py
```

Partial column run:

```bash
python3 manual_tests/column_solver_experiment.py --stop-timestep 8000
```

Column outputs use the unique stem:

```text
column_v1_distinct_skus_<robots>r_<orders>o[_<stop>t]
```

The runner writes a full trace and automatically invokes `manual_tests/analyze_traffic_trace.py`.

## Regression coverage

Important fleet tests:

- `tests/test_multi_robot_solver.py` — one-step traffic and rigid-footprint behavior.
- `tests/test_pallet_priority.py` — pallet-carrying robot priority override and equal-pallet-count fallback.
- `tests/test_scheduler.py` — current-timestep reservation primitives.

Important collection tests:

- `tests/test_aisles.py` — original 12-island layout/scoring/service planner.
- `tests/test_aisle_solver.py` — original aisle-aware coordination/refill behavior.
- `tests/test_previous_aisle.py` — previous-unit exclusion and fallback.
- `tests/test_column_solver.py` — 24-column geometry, directed monotonic plans, distinct-SKU utility, and removal of the final same-column rescan.

## Benchmark history

| Strategy | Makespan | Total moves | Collection moves |
| --- | ---: | ---: | ---: |
| Original full FIFO baseline | 161,470 | 734,235 | 703,548 |
| Earlier aisle-aware reference | 67,840 | 263,139 | 229,603 |
| Previous-aisle exclusion baseline | 66,598 | 263,750 | 229,903 |
| Distinct-SKU + pallet-aware priority | **64,090** | **251,462** | **219,088** |

The 24-column result is intentionally blank until the new strategy passes regression testing and a fresh complete replay.

## Current optimization boundary

Correctness first:

- preserve simulator legality;
- preserve complete rigid footprints;
- preserve pallet-home invariants;
- preserve dynamic pallet-aware traffic priority;
- preserve active pallet claims;
- preserve FIFO assignment and exact fulfillment;
- keep experiments isolated enough to compare them against the 64,090 reference;
- do not add movement exceptions unless a reproducible trace demonstrates the need.

## Possible later traffic experiment

Soft warehouse-direction costs remain optional and are not part of the first directed-column implementation. If trace data shows repeated opposite-direction conflicts in the same service lanes, possible later experiments include:

- soft penalties for opposing an already-active lane direction;
- temporary same-direction admission while a column is occupied;
- broader preferred vertical travel corridors.

These should remain soft optimization experiments unless correctness requires otherwise.

## Architecture summary

> Collection strategy decides what pallet column and direction to service. Static A* decides the preferred route through the current geometry. The pallet-aware fleet scheduler commits one footprint-safe step at a time. The simulator decides whether the complete timestep is legal. Then traffic planning restarts from the new real world state.
