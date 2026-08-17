# Development Manifest

Concise design notes, failure history, and optimization experiments for the solver.

- **Correctness-first, layered build**
  - Parser -> world -> writer -> spatial A* -> simulator -> order mechanics -> docking/replenishment -> one-robot solver -> FIFO allocation -> time-aware reservations -> five-robot baseline -> measured optimization experiments.
  - Each mechanic is tested independently; major milestones are also checked in Tutor's Testbench.

- **Clear ownership**
  - `WorldState`: mutable state + invariants.
  - `Simulator`: authority on action legality and state mutation.
  - `PathPlanner`: shortest static spatial paths.
  - `Scheduler`: time-aware multi-robot planning and reservations.
  - `MultiRobotSolver`: preserved known-valid FIFO five-robot baseline.
  - `AislePlanner`: aisle geometry, aisle scoring, and partial aisle service routes.
  - `AisleAwareSolver`: experimental fleet strategy built on the baseline movement/collision machinery.
  - `metrics.py`: solver-independent schedule measurement and benchmark reporting.

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
  - A pallet that will require replenishment is never planned with the pallet docked below the robot, because that footprint would extend to `y=40` on the replenishment row.

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
  - `src/multi_robot_solver.py` remains the preserved correctness baseline.
  - One global timestep loop drives five per-robot order state machines through collect, replenish, return, and fulfill phases.
  - Robots use FIFO orders, pallet claims, deterministic priority, rolling space-time planning, and cached movement plans.
  - Finished robots are permanent obstacles rather than temporary traffic.
  - Generated schedules are independently replayed from fresh input through the simulator.
  - Full 1000-order baseline: **161,470 timesteps**, **734,235 moves**, **703,548 collection moves**, **7,143 waits**, **31,121 aisle service visits**, **19,903 aisle re-entries**.
  - Baseline diagnosis: collection movement is the dominant cost; waits are under 1% of fleet robot-time.

- **Metrics**
  - Manual benchmark runs automatically print a human-readable report and write a matching `_metrics.json`.
  - Movement is decomposed into collection, refill, and fulfillment.
  - Reports include fleet robot-time, waits, action ratios, per-robot workload, refill trips, aisle visits, and aisle re-entries.
  - Aisle IDs now come from the same shared `build_aisle_layout()` definition used by the aisle solver.

## Optimization experiment 1: aisle-aware collection

- **Separate implementation**
  - `src/aisle_solver.py` defines `AisleAwareSolver`; the known-valid `MultiRobotSolver` is not replaced.
  - `manual_tests/aisle_solver_smoke_test.py` writes `aisle_v1_*` schedules and metrics so results remain directly comparable with the baseline.

- **Aisle model**
  - `src/aisles.py` groups the 240 pallet homes into the 12 connected 2x10 pallet islands.
  - An aisle plan is a sequence of only the pickup stops needed by the current order; no full loop is forced.
  - Once an aisle is selected, every remaining SKU requirement that can be serviced from that aisle is included in that aisle plan.

- **Two-stage aisle selection**
  - Cheap-score every useful aisle using required quantity, estimated entry distance, and the number of other robots currently committed to that aisle.
  - Current score is `useful_quantity / (distance + 1 + 8 * congestion)`; the congestion weight is intentionally modest and tunable.
  - Build detailed plans for the top three viable cheap candidates.
  - Detailed planning uses actual static A* distances and a multi-start nearest-neighbor route through the required pickup stops.
  - Re-score those candidates with the detailed planned distance and commit to the best result.

- **Aisle execution**
  - A robot keeps one `active_aisle_id` and executes the planned pickup stops in order.
  - Pallets are claimed only for the current stop, not for the entire future aisle route.
  - If a future stop becomes stale because another robot temporarily uses that pallet, replan the remaining work from the robot's current position.
  - After the aisle is complete, release the aisle commitment and score the remaining aisles again.

- **Replenishment separation**
  - Refill distance is deliberately excluded from aisle scoring so this experiment isolates collection-ordering improvements.
  - Refill is still performed whenever required while servicing an aisle.
  - Refill does not release the aisle commitment: the robot returns the pallet home, undocks, and resumes the same aisle stop.
  - Existing stock is consumed before a refill is triggered.

- **Experiment 1 success metrics**
  - Primary: reduce **703,548 baseline collection moves**.
  - Structural indicator: reduce **19,903 baseline aisle re-entries**.
  - Final objective: reduce the **161,470-timestep** makespan without weakening any correctness invariant.
  - Compare 5r/10, then 5r/100, then the full 5r/1000 benchmark before adding another optimization.

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

- **Bottom-docked refill footprint can leave the warehouse**
  - Risk identified during aisle-planner design: if the robot docks a pallet below itself, robot `y=39` implies pallet `y=40`.
  - Fix: when an aisle stop will require replenishment, pickup planning excludes the above-pallet robot position that would create a south/bottom docked pallet.

- **Regression coverage updated with each fix**
  - Added tests for conservative destination reservations, vacated pallet-home blocking, timed pallet-home blocking, docked-pallet home-cell behavior, deterministic aisle grouping, aisle scoring, partial aisle plans, refill-safe pickup selection, aisle-aware fleet replay, and refill-resume behavior.
  - Updated an older docking test whose expectation intentionally conflicted with the permanent-home-cell invariant.

## Current optimization boundary

- Preserve simulator legality, exact fulfillment, pallet-home invariants, collision reservations, deterministic replay, and the known-valid five-robot baseline.
- Do not relocate pallet home positions as an optimization; pallet homes are treated as fixed warehouse infrastructure.
- Change one measurable source of wasted timesteps at a time and compare against the saved baseline metrics.
- Directional traffic lanes and refill batching are intentionally deferred until aisle-aware collection is measured.

## Planned later experiments

- **Soft warehouse traffic**
  - Block ordinary through-travel on cells adjacent to stationary pallets unless that cell is the current service goal.
  - Preferred edges cost 1; normal edges cost 2.
  - Preferred vertical lanes: up at `x=(8,15,22,29,36,43)`, down at `x=(13,20,27,34,41,48)`.
  - Preferred center lanes: right at `y=21`, left at `y=18`.

- **Refill batching**
  - Opportunistically carry additional nearby pallets that already need replenishment when doing so adds little route cost and keeps the docked footprint legal.
  - Never use the bottom docking side for a trip to replenishment row `y=39`.

## Architecture summary

> The simulator owns legality, `PathPlanner` handles static geometry, `Scheduler` handles space-time conflicts, `MultiRobotSolver` preserves the known-valid baseline, and experimental solvers change only strategy above those layers.

## Development summary

> Prove each rule independently, preserve every discovered invariant, measure the baseline, then optimize one dominant source of wasted timesteps at a time.
