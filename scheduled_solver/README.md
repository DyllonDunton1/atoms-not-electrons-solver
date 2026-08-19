# Scheduled Solver (V2)

This package is intentionally independent from the legacy `src/` solver.
Algorithmic code here does **not** import `src.column_solver`, `src.aisle_solver`,
`src.multi_robot_solver`, or `src.pathfinding`.

The architecture is prioritized full-horizon scheduling:

1. FIFO assigns the next queued order to the next free robot.
2. That robot beam-searches a complete sequence of directed service columns.
3. Every beam transition uses space-time A* against already committed cell and
   edge reservations.
4. Pallet service and inventory are time-aware. Refill trips are explicitly
   planned with the enlarged robot+pallet footprint.
5. Pallet home cells remain permanent static obstacles even while a pallet is
   docked and physically moved away.
6. The winning complete order schedule is atomically committed before the next
   robot/order is planned.

The default reservation padding is one timestep on each side of every occupied
pose and movement edge. This is configurable through `SchedulerConfig`.

The package has its own parser, models, writer, geometry, reservation table,
space-time A*, inventory timeline, beam planner, scheduler, validation, and
planning metrics. The experiment runner is the only bridge to `src/`: it
converts the generated action list solely so the existing simulator can act as
an independent replay validator and legacy-metrics referee.

Run unit tests with:

```bash
python3 -m unittest discover -s tests/scheduled -v
```

Try a small experiment first:

```bash
python3 manual_tests/scheduled_solver_experiment.py --orders 10 --beam-width 8
```

Then scale upward after replay validation succeeds.
