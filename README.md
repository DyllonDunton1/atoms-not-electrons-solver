# Atoms Not Electrons Solver

A Python solver for Tutor Intelligence's **Atoms Not Electrons** warehouse optimization challenge.

Challenge site: https://www.atomsnotelectrons.com

## The challenge

The warehouse is a **60 x 40 grid** containing 5 robots, 100 SKU types, 240 pallets, and 1,000 orders. Fulfillment happens on `y = 0`; replenishment happens on `y = 39`.

Each robot can perform at most one action per timestep. Robots move one orthogonal grid cell at a time, pick items from adjacent pallets, dock to pallets, move docked pallets as a rigid footprint, replenish them at the bottom row, and fulfill completed orders at the top row.

A robot can fulfill an order only when its internal storage exactly matches an unfulfilled order. Pallets have finite stock, so depleted pallets eventually need a refill trip.

The score is the total number of timesteps required to fulfill all 1,000 orders. **Lower is better.**

## Submission format

The solver writes actions as:

```text
<timestep> <robot_id> <action> <x> <y>
```

For example:

```text
0 0 move 25 21
0 1 move 34 14
1 0 pick 24 21
2 0 move 25 21
```

Missing robot actions are waits. Generated schedules are replayed locally through the simulator before being uploaded to Tutor's Testbench for final visualization/validation.

## Current solver strategy

The project is built correctness-first and keeps navigation, traffic, task execution, and optimization separate.

1. Parse `BIG_ORDER.txt` into a mutable world model.
2. Assign unfulfilled orders through a deterministic FIFO queue.
3. Use aisle-aware collection planning to group useful pallet stops and reduce repeated travel.
4. Use static footprint-aware A* to calculate each robot's preferred route.
5. Recompute fleet traffic every timestep in robot-ID order and commit only the first move of each route.
6. Lower numeric robot IDs have priority. Higher-ID robots spatially route around already-committed lower-ID footprints.
7. Lower-ID robots do not take large speculative detours around active higher-ID robots. If the preferred first step is still physically occupied, they wait and recompute next timestep.
8. Every movement check uses the complete robot + docked-pallet rigid footprint.
9. If stock runs out, the current robot docks the pallet, carries it to `y = 39`, returns it home after replenishment, undocks, and resumes collection.
10. Once storage exactly matches the assigned order, the robot travels to `y = 0` and fulfills it.

There is deliberately **no multi-timestep prediction of other robots' future positions** in the current traffic layer. The solver plans a complete spatial route for guidance, commits one safe first step, advances the simulator, then starts traffic planning again from the new real world state.

## Traffic priority in one sentence

> Lower-ID robots own the preferred route; higher-ID robots adapt around lower-ID current/committed rigid footprints, while real current occupancy can still force a one-timestep wait.

This keeps priority separate from physical reality. A lower-ID robot may plan straight through where a higher-ID robot is standing because that traffic should eventually clear, but it still cannot execute an overlapping first move while the higher-ID footprint is actually there.

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
│   ├── metrics.py
│   └── writer.py
├── tests/
│   ├── README.md
│   └── test_*.py
├── manual_tests/
└── outputs/
```

### Module responsibilities

- **`models.py`** — Core data classes for robots, pallets, orders, and actions.
- **`parser.py`** — Parse the challenge input.
- **`world.py`** — Warehouse state, occupancy, and invariants.
- **`pathfinding.py`** — Static A* for arbitrary rigid robot footprints.
- **`scheduler.py`** — One-timestep reservation primitives for already-committed lower-ID moves.
- **`simulator.py`** — Final authority on action legality and state mutation.
- **`tasks.py` / `allocator.py`** — FIFO task representation and assignment.
- **`solver.py`** — Original single-robot baseline.
- **`multi_robot_solver.py`** — Five-robot FIFO execution plus the simple one-step priority traffic layer.
- **`aisles.py`** — Aisle geometry, scoring, and service-route planning.
- **`aisle_solver.py`** — Aisle-aware five-robot strategy using the same traffic layer.
- **`metrics.py`** — Schedule/robot/aisle performance reporting.
- **`writer.py`** — Challenge submission-file output.

## Testing

Run the whole automated suite from the repository root:

```bash
python3 -m unittest discover -s tests
```

Run a single module while iterating:

```bash
python3 -m unittest tests.test_multi_robot_solver -v
```

See [`tests/README.md`](tests/README.md) for the file-by-file guide. `test_multi_robot_solver.py` contains the main fleet-traffic regressions, including rigid docked footprints, the replenishment-row clearance case, the south-docked-pallet wall case, and waiting while a higher-ID robot is picking.

## Performance history

The original full FIFO baseline completed in **161,470 timesteps**. A later aisle-aware version completed all 1,000 orders in **67,840 timesteps**, with collection movement reduced from 703,548 to 229,603 moves.

Those runs used older traffic implementations. The simplified one-step priority scheduler needs a fresh full-run benchmark before its performance is treated as the current baseline.

## Planned traffic optimization

Soft directional warehouse traffic remains a later experiment rather than a correctness dependency. The idea is to give small cost preferences to opposing north/south lanes and center east/west lanes so head-on encounters become rarer without making any direction illegal.

The simple scheduler should be measured first; then soft traffic can be compared as an isolated optimization.
