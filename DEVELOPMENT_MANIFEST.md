# Development Manifest

Concise design notes and failure history for the solver.

- **Correctness-first, layered build**
  - Parser -> world -> writer -> spatial A* -> simulator -> order mechanics -> docking/replenishment -> one-robot solver -> FIFO allocation -> time-aware reservations -> five-robot solver.
  - Each mechanic is tested independently; major milestones are also checked in Tutor's Testbench.

- **Clear ownership**
  - `WorldState`: mutable state + invariants.
  - `Simulator`: authority on action legality and state mutation.
  - `PathPlanner`: shortest static spatial paths.
  - `Scheduler`: time-aware multi-robot planning and reservations.
  - `Solver` / `MultiRobotSolver`: task/order strategy and fleet execution.

- **Atomic timestep simulation**
  - Validate all actions before applying any.
  - Missing robot action = wait.
  - Conservative movement: a robot cannot enter a cell occupied at timestep start, even if it is being vacated.

- **Spatial A***
  - 4-connected Manhattan A* with arbitrary robot-relative footprints.
  - Every pallet `original_position` is permanently treated as reserved warehouse structure.
  - A pallet away from home also blocks its current position.
  - An ignored/docked pallet may occupy its own home cell as part of the moving footprint, but the robot center may never route onto any pallet home cell.

- **Picking / fulfillment**
  - Pick requires orthogonal adjacency; simultaneous picks share starting stock.
  - Fulfill requires `y=0` and an exact order multiset match (`Counter`).
  - Successful fulfillment clears storage.

- **Docking / replenishment**
  - Docked pallets store owner + cardinal relative offset and move rigidly with the robot.
  - Replenishment is automatic at end of timestep when the robot is on `y=39`.
  - Baseline replenish routine: dock -> carry to `y=39` -> refill -> return pallet to its original position -> undock.

- **FIFO task allocation**
  - Global order tasks use a FIFO queue; free robots receive work in deterministic robot-ID order.
  - Assignment removes each task exactly once.
  - Replenishment remains a private subroutine of an order task, not a global queued task.

- **Time-aware reservations**
  - Reservation table tracks occupied cells by state timestep and movement edges by action timestep.
  - Scheduler searches `(position, timestep)` with move + wait successors.
  - Prevents same-cell conflicts, head-on edge swaps, and entering cells another robot occupies at the action-start state.
  - A transition `A@t -> B@(t+1)` reserves `A@t`, `B@t`, and `B@(t+1)` so reservation semantics exactly match the simulator's conservative movement rule.
  - Reservations cover the full docked footprint, including every moving footprint edge.

- **Five-robot baseline**
  - One global timestep loop drives five per-robot order state machines through collect, replenish, return, and fulfill phases.
  - Robots use FIFO orders, pallet claims, deterministic priority, rolling space-time planning, and cached movement plans.
  - Finished robots are permanent obstacles rather than temporary traffic.
  - Generated schedules are independently replayed from fresh input through the simulator.
  - A complete valid five-robot solution is now established; optimization is the next phase.

## Failure history and fixes

- **Completed robots caused a small-run livelock**
  - Symptom: the five-robot / ten-order run could stop at 9/10 because a finished robot parked on the fulfillment row was treated as temporary traffic.
  - Cause: route-goal selection could choose cells that a permanently idle robot would never vacate.
  - Fix: finished/idle robots are classified as permanent obstacles for spatial goal selection and timed planning.

- **Two-robot oscillation caused the 100-order run to stall at 55/100**
  - Symptom: robots 0 and 1 entered a perfect two-cell oscillation; other robots later accumulated around them.
  - Cause: reservation checking required a destination to be free at both `t` and `t+1`, but reservation recording only reserved the destination at `t+1`. A lower-priority robot could therefore legally plan to occupy a cell at `t` that a higher-priority robot intended to enter during that timestep.
  - Fix: every reserved transition now reserves its destination footprint at both the action-start and resulting timesteps. Regression tests cover the pre-arrival destination reservation.

- **Vacated pallet homes caused the 1000-order run to deadlock at 864/1000**
  - Symptom: while pallet 82 was temporarily carried away from `(38, 8)`, another robot selected `(38, 8)` as a pickup position for neighboring pallet 84. Pallet 82 later returned home, making the stored pickup goal permanently impossible and creating a resource/traffic deadlock.
  - Cause: pathfinding treated a pallet's current position as blocked but treated its temporarily empty original position as normal floor.
  - Fix: all pallet original positions are permanently blocked in both spatial and time-aware planning, while moved pallets additionally block their current positions. Docked-footprint exceptions allow the carried pallet itself to leave and return without allowing robot centers to use pallet slots.

- **Regression coverage updated with each fix**
  - Added tests for conservative destination reservations, vacated pallet-home blocking, timed pallet-home blocking, and docked-pallet home-cell behavior.
  - Updated an older docking test whose expectation intentionally conflicted with the new permanent-home-cell invariant.

## Current optimization boundary

- Preserve simulator legality, exact fulfillment, pallet-home invariants, collision reservations, deterministic replay, and the known-valid five-robot baseline.
- Improve timestep count iteratively from this baseline; measure each optimization against full validation rather than weakening correctness rules.
- Candidate optimization areas include pallet/order selection, fleet traffic efficiency, replenishment strategy, routing reuse, and broader cross-order coordination.

## Architecture summary

> The simulator owns legality, `PathPlanner` handles static geometry, `Scheduler` handles space-time conflicts, and the solver decides what work to do.

## Development summary

> Prove each rule independently, compose a deterministic valid baseline, preserve every discovered invariant, then optimize one measurable source of wasted timesteps at a time.
