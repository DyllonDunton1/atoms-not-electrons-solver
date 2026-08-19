# Test Suite Guide

Run the full automated suite from the repository root with:

```bash
python3 -m unittest discover -s tests
```

Run one module while iterating, for example:

```bash
python3 -m unittest tests.test_column_solver -v
```

| File | What it covers |
| --- | --- |
| `test_aisle_solver.py` | Existing 12-island aisle-aware fleet execution, neighboring-pallet coordination, active pallet claims, final same-aisle rescanning, and refill/resume behavior. |
| `test_aisles.py` | Existing 12-island geometry, scoring, deterministic service plans, pallet availability, and refill-safe pickup choices. |
| `test_column_solver.py` | Experimental 24-column layout, exposed-side pickup lanes, both directed monotonic routes, distinct-SKU utility, no-final-rescan behavior, and previous-column-only persistent-adjacency fallback. |
| `test_previous_aisle.py` | Previous-unit exclusion and fallback when the previous unit is the only remaining useful choice. |
| `test_pallet_priority.py` | Dynamic traffic priority where docked-pallet count precedes robot ID. |
| `test_multi_robot_solver.py` | Multi-robot FIFO integration plus one-step traffic, rigid-footprint, and picking-blocker regressions. |
| `test_allocator.py` | FIFO task queue behavior and assignment. |
| `test_docking.py` | Dock/undock rules, rigid docked-pallet footprints, and docking legality. |
| `test_metrics.py` | Schedule metrics, movement categories, waits, aisle visits, and related reporting. |
| `test_order_simulation.py` | Order picking and fulfillment behavior through the simulator. |
| `test_parser.py` | Parsing challenge input into robots, pallets, SKU capacities, and orders. |
| `test_pathfinding.py` | Static A* routing, pallet-home blocking, and larger robot/docked-pallet footprints. |
| `test_replenishment.py` | Replenishment-row behavior and refill-related movement mechanics. |
| `test_scheduler.py` | One-timestep reservation primitives used to protect committed moves and reject footprint/edge conflicts. |
| `test_simulator.py` | Individual simulator legality rules and state mutation. |
| `test_simulator_integration.py` | Multi-action and multi-timestep simulator integration/replay behavior. |
| `test_solver.py` | Original single-robot baseline solver. |
| `test_world.py` | World-state invariants, occupancy, docked-pallet consistency, and validation failures. |
| `test_writer.py` | Submission-file formatting and action serialization. |

## Traffic regressions to keep especially visible

`test_multi_robot_solver.py` separates preferred route choice from physical first-step legality. A higher-priority robot may keep a natural route through lower-priority traffic, but it still cannot execute an overlapping move while that lower-priority rigid footprint physically occupies the requested cells.

`test_pallet_priority.py` covers the current dynamic priority key:

```python
(-num_docked_pallets, robot_id)
```

A robot carrying a pallet therefore plans ahead of one carrying none. Equal pallet counts retain the robot-ID fallback.

## Original aisle coordination

`test_aisle_solver.py` covers the warehouse-specific coordination used by the existing 12-island strategy, including deferred future pallet selection and preservation of an already-active picker.

The old strategy still performs a final same-aisle rescan before leaving. That behavior is intentionally retained for the old solver and intentionally removed only in the new column strategy.

## Directed-column regressions

`test_column_solver.py` checks the new collection abstraction without changing spatial pathfinding or fleet traffic:

- 12 physical 2 x 10 islands become exactly 24 exposed service columns;
- every column has 10 pallets and one fixed exposed-side pickup lane;
- both `up` and `down` plans keep stop `y` values monotonic;
- useful quantity in the column experiment means **distinct useful SKU count**;
- completing the final stored stop ends the column pass immediately instead of performing the old final same-unit rescan;
- persistent adjacency does not reject a normal selected column or active stop;
- persistent adjacency is added only when the solver must fall back into the previous column, and only previous-column pallets are affected;
- remaining work is left for a fresh global route choice, with the previous column re-entered only when it is both necessary and currently usable.

The directed-column benchmark runner is:

```bash
python3 manual_tests/column_solver_experiment.py
```

and a short diagnostic run can be generated with:

```bash
python3 manual_tests/column_solver_experiment.py --stop-timestep 8000
```
