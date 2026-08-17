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
  - Deterministic priority is robot-ID order. Before a moving robot has been planned for the current timestep, its current footprint is treated as known occupancy only at `t`, not speculative occupancy at `t+1`.
  - Once a higher-priority robot's trajectory is chosen, its normal full trajectory reservations are authoritative; lower-priority robots planned afterward must yield or spatially detour immediately around those committed reservations.
  - Before committing a priority trajectory, the solver checks every still-unplanned lower-priority moving robot for at least one legal immediate transition. If the proposed trajectory would remove both waiting and every legal first move, the solver preserves that robot's best currently legal escape transition, replans the higher-priority trajectory around it, and then forces the yielding robot to execute that first move. This prevents a rigid docked footprint from being trapped by an otherwise valid higher-priority claim without adding a stall timer or deliberate recovery delay.

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
  - Before releasing an aisle after its final planned stop, rescan that same aisle against the robot's still-unfulfilled requirements and the current live pallet availability.
  - If a pallet that was previously busy has become available during the visit, extend the current aisle plan in place and service it before leaving.
  - The rescan does not predict release times, create future pallet reservations, or establish a dibs queue. If the pallet is still unavailable at the final rescan, release the aisle normally and let later warehouse-level scoring rediscover it if needed.
  - After the aisle is actually complete, release the aisle commitment and score the remaining aisles again.

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

- **Temporary pallet contention caused an avoidable aisle re-entry**
  - Observed in the first 5r/10 aisle-aware benchmark: robot 1 initially serviced aisle 5 while robot 3 was using the remaining needed SKU-1 pallet at `(45, 16)`. Robot 3 released it at `t=56`, but robot 1's original aisle plan did not include that pallet; robot 1 finished the rest of aisle 5 at `t=64` and did not rediscover the now-free pallet until returning at `t=135`.
  - Cause: the aisle plan correctly excluded unavailable pallets when it was built, but finishing the stored plan was treated as finishing the aisle even though live pallet availability could have changed during the visit.
  - Fix: after the last stored stop, rescan the current aisle using current claims and remaining requirements. If useful work is now available, extend the aisle plan without dropping the aisle commitment. No future release prediction or pallet queue is introduced.

- **Adjacent active robots could deadlock instead of passing**
  - Observed in the 5r/1000 run capped at 45,000 timesteps: robot 1 stopped at `(16,26)` after `t=40830` while robot 3 later stopped immediately ahead at `(16,25)`; independently, robot 2 stopped at `(33,28)` and robot 0 stopped behind it at `(33,29)`. Open side space existed for immediate passing.
  - Cause: while planning a robot, every other active robot's current cell was provisionally reserved at both `t` and `t+1`. That speculative `t+1` occupancy let timed A* repeatedly prefer waiting for the blocking robot to disappear rather than committing the higher-priority trajectory and forcing the lower-priority robot to move aside.
  - Fix: provisional occupancy for unplanned active robots is now reserved only at the known current state `t`. Actual chosen trajectories keep the existing conservative full reservations. Because robots are planned in ID order, the lower-ID trajectory can claim future space and the higher-ID robot planned afterward must yield or detour on the first available timestep.

- **Docked rigid footprint could lose its only yield move**
  - Observed in the next 5r/1000 run capped at 20,000 timesteps: at `t=7461`, robot 0 was at `(16,11)` while robot 1 docked the east-side pallet at `(17,10)` from center `(16,10)`. Robot 1's only legal immediate escape was left to `(15,10)`, which sweeps the docked pallet into robot 1's old center `(16,10)`. The higher-priority robot 0 trajectory could reserve that cell at the same future state, eliminating robot 1's only legal transition; both stopped and the remaining fleet eventually queued behind them.
  - Cause: treating an unplanned robot's future center as free was correct for ordinary single-cell yielding, but a rigid docked footprint may need one of its current cells as a swept footprint cell during the yield transition.
  - Fix: before a higher-priority trajectory is committed, test whether every lower-priority moving robot can still either wait or make at least one legal one-step move with its full current footprint. If not, select its best legal immediate escape from the pre-commit state, reserve that transition while replanning the higher-priority trajectory, and force the lower-priority robot to execute the escape when its turn is planned. The response is immediate; there is no wait threshold or deadlock-recovery timer.

- **Regression coverage updated with each fix**
  - Added tests for conservative destination reservations, vacated pallet-home blocking, timed pallet-home blocking, docked-pallet home-cell behavior, deterministic aisle grouping, aisle scoring, partial aisle plans, refill-safe pickup selection, aisle-aware fleet replay, refill-resume behavior, final-aisle live-availability rescanning, immediate priority-based yielding between adjacent active robots, and rigid docked-footprint escape preservation.
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
