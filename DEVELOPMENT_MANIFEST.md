# Development Manifest

Concise notes on the main architectural and implementation decisions made while building the solver.

- **Built correctness-first, in layers**
  - Parser -> world state -> writer -> A* -> simulator -> picking/fulfillment -> docking -> replenishment -> one-robot solver.
  - Each mechanic was unit-tested before higher-level logic depended on it.
  - Major milestones were also checked against Tutor's official Testbench.

- **Separated responsibilities**
  - `WorldState`: owns mutable warehouse state and validates invariants.
  - `Simulator`: enforces challenge rules and applies actions.
  - `PathPlanner`: answers spatial shortest-path questions.
  - `Solver`: makes strategy decisions.
  - Solver strategy is intentionally separate from game-rule correctness.

- **Atomic timestep simulation**
  - Validate the complete timestep before mutating state.
  - Prevents partial execution if one action in a timestep is illegal.
  - Important for simultaneous movement, picks, fulfillment, and docking.

- **A* pathfinding**
  - Uses A* with Manhattan distance on the 4-connected grid.
  - Produces shortest spatial paths.
  - Static pallets are obstacles.
  - In the one-robot baseline, the other four robots are fixed obstacles.

- **Conservative movement semantics**
  - A move destination must be empty at the start of the timestep.
  - Robot swaps and entering a cell another robot is leaving are rejected for now.
  - Chosen as a safe baseline before concurrency optimization.

- **Picking**
  - Requires orthogonal adjacency.
  - One pick removes one item and adds that SKU to robot storage.
  - Simultaneous picks from one pallet are validated in aggregate against starting stock.

- **Fulfillment**
  - Robot must be on `y=0`.
  - Storage must exactly match an unfulfilled order.
  - Uses `Counter`, so SKU ordering does not matter but quantities must match exactly.
  - Successful fulfillment clears robot storage.

- **Docking representation**
  - Pallets store both owning robot and relative docking offset.
  - Legal offsets are the four cardinal sides.
  - Robot plus docked pallets form one rigid footprint.

- **Footprint-aware movement**
  - Every docked pallet moves by the same delta as the robot.
  - Every footprint cell must remain in bounds and collision-free.
  - A robot may move into the current cell of its own docked pallet because that pallet moves simultaneously.

- **Footprint-aware A***
  - Pathfinding accepts a robot-relative footprint.
  - Docked pallets are removed from the static obstacle set and represented as part of the moving footprint.

- **Replenishment**
  - Modeled as an automatic end-of-timestep rule, not an action.
  - If a robot ends a timestep on `y=39`, all pallets still docked to it refill to `max_count`.
  - Therefore:
    - move onto `y=39` -> refill immediately
    - pick at `y=39` -> pick first, refill second
    - dock at `y=39` -> can refill immediately
    - undock at `y=39` -> pallet does not refill

- **Baseline replenishment routine**
  - Dock depleted pallet.
  - Carry it to `y=39` with footprint-aware A*.
  - Refill automatically.
  - Return it to its original position.
  - Undock.

- **One-robot autonomous solver**
  - Robot 0 solves orders sequentially; other robots remain stationary obstacles.
  - Repeated SKUs are grouped so the required quantity is collected in one visit.
  - Chooses the nearest reachable pallet using actual A* path length.
  - Prefers a pallet with enough current stock; otherwise runs replenish-and-return.

- **Solver validates itself continuously**
  - Every generated action is immediately executed through the local simulator.
  - Planning always uses the updated authoritative state.
  - No duplicate tracking of robot position, pallet stock, or storage.

- **Deterministic baseline**
  - Tie-breaking is deterministic.
  - Orders are processed in increasing/FIFO order.
  - Goal is a predictable correctness baseline before optimization.

- **Deliberately postponed**
  - Multi-robot task allocation.
  - Reservation table / time-aware collision avoidance.
  - Concurrent autonomous planning.
  - Strategic pallet relocation.
  - Cross-order optimization.

## Architecture summary

> The simulator is the authority on legality, A* handles spatial planning, and the solver only decides what action should happen next.

## Development summary

> Build and validate every mechanic independently, then compose them into a one-robot solver that continuously simulates its own output before adding concurrency.
