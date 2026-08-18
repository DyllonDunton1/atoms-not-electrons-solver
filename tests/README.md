# Test Suite Guide

This directory contains the automated `unittest` coverage for the solver. Run the full suite from the repository root with:

```bash
python3 -m unittest discover -s tests
```

Run one module while iterating with, for example:

```bash
python3 -m unittest tests.test_multi_robot_solver -v
```

| File | What it covers |
| --- | --- |
| `test_aisle_solver.py` | End-to-end aisle-aware fleet execution, live final-aisle rescanning, and refill/resume behavior. |
| `test_aisles.py` | Aisle geometry, aisle scoring, deterministic service plans, pallet availability, and refill-safe pickup choices. |
| `test_allocator.py` | FIFO task queue behavior and assignment of work to robots. |
| `test_docking.py` | Dock/undock rules, rigid docked-pallet footprints, and docking legality. |
| `test_metrics.py` | Schedule metrics, movement categories, waits, aisle visits, and related reporting. |
| `test_multi_robot_solver.py` | Multi-robot FIFO integration plus the one-step robot-ID priority traffic model, including rigid-footprint and picking-blocker regressions. |
| `test_order_simulation.py` | Order picking and fulfillment behavior through the simulator. |
| `test_parser.py` | Parsing challenge input into robots, pallets, SKU capacities, and orders. |
| `test_pathfinding.py` | Static A* routing, pallet-home blocking, and larger robot/docked-pallet footprints. |
| `test_replenishment.py` | Replenishment-row behavior and refill-related movement mechanics. |
| `test_scheduler.py` | One-timestep reservation primitives used to protect already-committed lower-ID moves and reject footprint/edge conflicts. |
| `test_simulator.py` | Individual simulator legality rules and state mutation for all supported actions. |
| `test_simulator_integration.py` | Multi-action and multi-timestep simulator integration/replay behavior. |
| `test_solver.py` | The original single-robot baseline solver. |
| `test_world.py` | World-state invariants, occupancy, docked-pallet consistency, and validation failures. |
| `test_writer.py` | Submission-file formatting and action serialization. |

## Traffic tests to keep especially visible

`test_multi_robot_solver.py` is the main regression file for fleet traffic. Its synthetic cases intentionally separate **preferred route choice** from **physical first-step legality**: lower-ID robots do not reroute around active higher-ID robots, but they still wait if a higher-ID rigid footprint physically occupies the requested first step. Higher-ID robots then plan around lower-ID current/committed footprints.

The same file also covers the replenishment-row rigid-footprint case, the south-docked-pallet wall case, and the accepted behavior where a lower-ID robot waits while a higher-ID robot finishes picking before traffic can clear.
