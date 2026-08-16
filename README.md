# Atoms Not Electrons Solver

A Python solver for Tutor Intelligence's **Atoms Not Electrons** warehouse optimization challenge.

Challenge site: https://www.atomsnotelectrons.com

## The Challenge

The warehouse is a **60 x 40 grid** containing:

- 5 robots
- 100 SKU types
- 240 pallets
- 1,000 orders
- A fulfillment row at `y = 0`
- A replenishment row at `y = 39`

Each robot can perform at most one action per timestep. Robots move one grid cell at a time, pick items from adjacent pallets, dock to pallets, move docked pallets, replenish them at the bottom row, and fulfill completed orders at the top row.

A robot can fulfill an order only when its internal storage **exactly matches** an unfulfilled order. Pallets have finite stock, so robots must eventually dock to depleted pallets and bring them to the replenishment row.

The score is the total number of timesteps required to fulfill all 1,000 orders. **Lower is better.**

## Submission Format

The solver produces a text file containing robot commands in the form:

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

The generated file can be uploaded to the online Testbench for validation and visualization.

## Baseline Solver

The first goal of this project is a simple, correct baseline rather than an immediately optimal solution.

The baseline design is:

1. Parse `BIG_ORDER.txt` into a world model.
2. Maintain a queue of unfulfilled orders.
3. Assign the next available order to the next available robot.
4. Select pallets containing the required SKUs.
5. Use shortest-path planning to move robots through the warehouse.
6. Reserve space over time so planned robot paths do not collide.
7. If a required pallet runs out of stock, the robot using it docks to it, carries it to `y = 39`, replenishes it, returns it to its original location, and resumes the order.
8. Once a robot's storage exactly matches its assigned order, route it to `y = 0` and fulfill the order.
9. Repeat until all 1,000 orders are complete.

This intentionally leaves more advanced optimization for later iterations.

## Architecture

```text
atoms-not-electrons-solver/
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── parser.py
│   ├── models.py
│   ├── world.py
│   ├── pathfinding.py
│   ├── scheduler.py
│   ├── tasks.py
│   ├── allocator.py
│   ├── simulator.py
│   ├── solver.py
│   └── writer.py
├── tests/
│   ├── test_parser.py
│   ├── test_pathfinding.py
│   ├── test_simulator.py
│   └── test_docking.py
└── outputs/
    └── .gitkeep
```

### Module Responsibilities

- **`parser.py`** — Read `BIG_ORDER.txt` and construct the initial problem state.
- **`models.py`** — Core data classes such as robots, pallets, orders, actions, and tasks.
- **`world.py`** — Warehouse state, occupancy, inventory, and rule checks.
- **`pathfinding.py`** — Shortest-path planning for robots and docked robot footprints.
- **`scheduler.py`** — Time-based reservations and multi-robot collision avoidance.
- **`tasks.py`** — Order fulfillment and replenishment task definitions.
- **`allocator.py`** — Assign queued work to available robots.
- **`simulator.py`** — Execute and validate generated actions locally.
- **`solver.py`** — Main coordination loop.
- **`writer.py`** — Export the final action schedule in challenge submission format.

## Planned Development Order

The initial implementation will be built incrementally:

1. Parse the challenge input.
2. Build the world and data models.
3. Implement static shortest-path planning.
4. Implement the local simulator and rule validation.
5. Complete orders with one robot.
6. Add all five robots.
7. Add time-based collision reservations.
8. Add pallet docking and replenishment.
9. Produce a complete 1,000-order baseline submission.
10. Measure bottlenecks and begin optimization.

Later strategies may include improved task allocation, order reordering, pallet relocation, high-runner SKU staging, predictive replenishment, and search-based optimization.

## Running

The exact command-line interface will be added as the implementation develops. The intended workflow is:

```text
BIG_ORDER.txt -> solver -> local validation -> submission.txt -> online Testbench
```

The online Testbench remains the final source of truth for challenge validation and visualization.
