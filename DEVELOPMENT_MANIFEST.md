# Development Manifest

Concise design notes for the solver.

- **Correctness-first, layered build**
  - Parser -> world -> writer -> spatial A* -> simulator -> order mechanics -> docking/replenishment -> one-robot solver -> FIFO allocation -> time-aware reservations.
  - Each mechanic is tested independently; major milestones are also checked in Tutor's Testbench.

- **Clear ownership**
  - `WorldState`: mutable state + invariants.
  - `Simulator`: authority on action legality and state mutation.
  - `PathPlanner`: shortest static spatial paths.
  - `Scheduler`: time-aware multi-robot planning and reservations.
  - `Solver`: task/order strategy.

- **Atomic timestep simulation**
  - Validate all actions before applying any.
  - Missing robot action = wait.
  - Conservative movement: a robot cannot enter a cell occupied at timestep start, even if it is being vacated.

- **Spatial A***
  - 4-connected Manhattan A*.
  - Supports arbitrary robot-relative footprints.
  - Docked pallets become part of the moving footprint rather than static obstacles.

- **Picking / fulfillment**
  - Pick requires orthogonal adjacency; simultaneous picks share starting stock.
  - Fulfill requires `y=0` and an exact order multiset match (`Counter`).
  - Successful fulfillment clears storage.

- **Docking / replenishment**
  - Docked pallets store owner + cardinal relative offset and move rigidly with the robot.
  - Replenishment is automatic at end of timestep when the robot is on `y=39`.
  - Baseline replenish routine: dock -> carry to `y=39` -> refill -> return pallet -> undock.

- **One-robot baseline**
  - Robot 0 solves orders sequentially; other robots are stationary obstacles.
  - Groups repeated SKUs, chooses nearest reachable pallets by actual A* path length, and replenishes when needed.
  - Every generated action is immediately replayed through the simulator.

- **FIFO task allocation**
  - Global order tasks use a simple FIFO queue.
  - Assignment removes the task exactly once; robot availability determines who receives the next task.
  - Replenishment remains a private subroutine of an order task, not a global queued task.

- **Time-aware reservations**
  - Reservation table tracks occupied cells by state timestep and movement edges by action timestep.
  - Scheduler searches `(position, timestep)` with move + wait successors.
  - Prevents same-cell conflicts, head-on edge swaps, and entering cells another robot is vacating that timestep.
  - Reservations cover the full docked footprint, including every moving footprint edge.

- **Prioritized multi-robot planning**
  - Robots plan deterministically in priority/call order.
  - Each completed trajectory is reserved before the next robot plans.
  - Lower-priority robots may wait or spatially detour around earlier trajectories.
  - Browser smoke tests verified crossings, forced waits, two-robot swaps, and four-robot head-on traffic.

- **Deterministic baseline**
  - Stable tie-breaking and FIFO order handling make behavior repeatable and debuggable.
  - Optimization comes after a fully valid five-robot solution.

- **Still to integrate / optimize**
  - Five-robot autonomous order execution using allocator + scheduler.
  - Dynamic replanning if prioritized planning cannot find a route.
  - Strategic pallet relocation and cross-order/fleet optimization.

## Architecture summary

> The simulator owns legality, `PathPlanner` handles static geometry, `Scheduler` handles space-time conflicts, and the solver decides what work to do.

## Development summary

> Prove each rule independently, compose a deterministic valid baseline, then add concurrency and optimize only after correctness is locked down.
