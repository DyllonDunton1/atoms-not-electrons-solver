# Development Manifest

Concise design notes, invariants, benchmark history, and the current optimization boundary for the solver.

## Current architecture

- `WorldState` owns mutable warehouse state and invariants.
- `Simulator` is the authority on action legality and state mutation.
- `PathPlanner` performs static spatial A* with arbitrary rigid robot footprints.
- `ReservationTable` protects only the first moves already committed during the current timestep.
- `MultiRobotSolver` runs FIFO orders and deterministic robot-ID priority traffic.
- `AislePlanner` chooses useful aisles and service stops for an order.
- `AisleAwareSolver` layers aisle batching and live pallet-availability rules on top of the same fleet traffic machinery.
- `metrics.py` measures generated schedules independently of solver strategy.

## Non-negotiable correctness rules

- The warehouse is a 60 x 40 grid; fulfillment is at `y=0` and replenishment at `y=39`.
- A robot may move only one orthogonal cell per timestep.
- Missing robot action means wait.
- The simulator validates the whole timestep before mutating any state.
- A robot may not enter a cell occupied at the start of the timestep even if that entity intends to leave during the same timestep.
- Docked pallets remain rigidly attached at their cardinal offsets and are part of the moving robot footprint.
- Every movement/path legality check uses the complete rigid footprint, not only the robot center.
- Head-on use of the same movement edge is rejected.
- Pallet home cells remain warehouse structure even while the pallet is temporarily carried away.
- Robot centers never route onto pallet home cells.
- Exact order multisets are required for fulfillment.
- Replenishment happens automatically at the end of a timestep when a robot carrying docked pallets is on `y=39`.

## Spatial path planning

`PathPlanner` is deliberately spatial rather than predictive.

- It uses four-connected Manhattan A*.
- It checks the entire supplied footprint at every candidate robot-center position.
- Static pallet homes remain blocked.
- A moved undocked pallet blocks both its home and current location.
- Pallets represented by a moving robot footprint can be ignored as independent obstacles through `ignored_pallet_ids`; the robot center still cannot enter any pallet home cell.
- Temporary robot blockers are supplied by the fleet layer for the current path calculation.
- There are no future-time robot states inside spatial A*.

## One-step fleet traffic model

The fleet traffic model is intentionally simple and is rebuilt from the real world state every timestep.

1. Robots are processed in ascending robot ID. Lower numeric ID always has priority.
2. Each moving robot computes a complete spatial path to its current goal, but only the first step can be committed.
3. Finished/inactive robots are permanent obstacles for every robot.
4. When robot `R` chooses its preferred spatial path, active lower-ID robots are treated as static obstacles. Their complete current rigid footprints and any first-step destination footprints already committed this timestep are blocked.
5. Active higher-ID robots are intentionally omitted from `R`'s preferred route calculation. Their docked pallets are omitted too because they belong to the same lower-priority moving assembly.
6. Omitting higher-ID traffic from route choice does **not** remove physical occupancy. Before the first step executes, the candidate destination footprint is checked against every other robot's real current rigid footprint.
7. If a lower-ID robot's preferred first step is still occupied by a higher-ID robot, the lower-ID robot waits. It does not take a large speculative detour around traffic that is responsible for clearing the way.
8. Higher-ID robots then plan around lower-ID current/committed footprints and naturally take a spatial detour when one exists.
9. `ReservationTable` covers only these current-timestep commitments and full-footprint edge/cell conflicts. It does not reserve complete future trajectories.
10. After the simulator advances one timestep, all traffic path decisions are discarded and the process starts again from the new real positions.

This replaces the earlier space-time trajectory/caching/forced-yield machinery. The old implementation remains available in git history rather than being kept as dormant special-case code.

## Waiting semantics

Waiting remains a normal and important outcome, but it is no longer a prediction about future reservations.

A wait means: **there is no safe first move worth committing from the current snapshot; stay in place and recompute next timestep.**

Typical examples:

- A lower-ID robot wants a cell currently occupied by higher-ID traffic. It waits while that traffic clears through its own normal task/path decisions.
- A higher-ID robot has no complete spatial route around the current lower-ID blockers. It waits and tries again after the world changes.
- A higher-ID robot is currently performing repeated `PICK` actions in a lower-ID robot's preferred route. The lower-ID robot may wait for those picks to finish; task preemption is intentionally not part of the traffic model.

The fleet layer intentionally does **not** contain a generic forced-side-step rule for a synthetic exact-goal swap where two robots' final movement goals are literally each other's current cells. The warehouse strategy avoids the realistic source of that pattern at the aisle-selection layer instead of complicating traffic with another special case.

## Rigid-footprint priority behavior

Priority applies to the complete robot assembly.

- If R2 carries a south/east/west/north docked pallet, all cells occupied by that assembly are considered when R4 plans around R2.
- When R2 plans its own preferred path, an active higher-ID R4 and all pallets docked to R4 do not distort the route.
- R2 still cannot physically move its own rigid footprint into R4/R4's docked pallet during the current timestep; it waits until the space is actually clear.
- The replenishment-row regression specifically covers a two-step lateral clearance where the higher-ID rigid robot must move sideways twice before the lower-ID robot can continue.

## FIFO task execution

- Unfulfilled orders are kept in a global FIFO queue.
- Free robots receive work in deterministic robot-ID order.
- Each order is assigned exactly once.
- Replenishment remains a private subroutine of the robot's current order rather than a global queued task.
- Finished robots become permanent obstacles when there is no remaining work they could receive.

## Picking, docking, and replenishment

- Pick requires orthogonal adjacency.
- Multiple robots may pick from the same pallet in one timestep as long as combined requests do not exceed starting stock.
- Docked pallets store owner and cardinal relative offset and move rigidly with their robot.
- A refill trip is dock -> travel to `y=39` -> automatic refill -> return the pallet home -> undock -> resume collection.
- Refill planning avoids a bottom/south-docked geometry that would force a pallet outside the grid when the robot reaches `y=39`.

## Aisle-aware collection experiment

`AisleAwareSolver` remains the current collection optimization layer.

- The 240 pallet homes are grouped into 12 connected 2 x 10 pallet islands.
- An aisle visit includes only the stops needed by the current order; no full loop is forced.
- A cheap score ranks useful aisles using required quantity, entry distance, and current aisle congestion.
- The best few candidates receive detailed static A* route evaluation.
- Within an aisle, service order is deterministic and uses nearest useful stops.
- Pallets are claimed only for the current stop rather than the whole future aisle route.
- If a stored stop becomes stale because another robot is using the pallet, the active aisle is replanned from the robot's current position.
- Before leaving after the last stored stop, the solver rescans the same aisle against live pallet availability and remaining requirements. Newly released useful pallets can extend the visit in place.
- Refill remains a subroutine of the active aisle; the aisle commitment survives the refill trip.

### Priority-aware pallet availability

A small warehouse-specific rule prevents neighboring robots from selecting each other's just-finished pickup positions and creating an artificial goal swap.

- The rule applies only while choosing or replanning **future aisle stops**.
- For robot `R`, an undocked pallet currently adjacent to an active lower-ID robot is temporarily unavailable to `R`.
- Example: if R2 is beside P2 and R4 is beside P4, R2 may still choose P4 because R4 has lower priority. If R4 would otherwise choose P2, it temporarily skips P2 because R2 has higher priority and is already beside it.
- R4 then continues to another useful stop, which naturally clears P4 for R2 without any forced traffic maneuver.
- Once R2 moves away, P2 becomes available again. The existing final same-aisle rescan can add it back later if R4 still needs it.
- Adjacency is **not** a claim and does not reserve a pallet into the future.
- Most importantly, adjacency never preempts an already-active stop. If R4 is actively picking P4 and R2 merely drives past the other side of P4, R4 keeps its claim and continues picking.

This keeps the responsibility in the correct layer: aisle planning decides which pallet is sensible to service next; the fleet traffic layer only resolves the first physical move toward the chosen goal.

## Benchmark history

These numbers are historical references from earlier traffic implementations, not guarantees for the new one-step scheduler.

- Original full FIFO baseline: **161,470 timesteps**, **734,235 moves**, **703,548 collection moves**, **31,121 aisle visits**, **19,903 aisle re-entries**.
- First completed full aisle-aware 5-robot / 1,000-order run: **67,840 timesteps**, **263,139 moves**, **229,603 collection moves**, **8,491 aisle visits**, **25 aisle re-entries**.
- A later predictive-traffic run reached **998/1000 orders by timestep 72,000** and exposed a long rigid-robot replenishment oscillation. That failure, together with earlier traffic-specific patches, motivated replacing future trajectory prediction with the current one-step model.

A new performance baseline should be recorded only after the simplified scheduler and priority-aware aisle selection complete the normal unit/integration tests and a fresh full 1,000-order run.

## Regression coverage

`tests/test_multi_robot_solver.py` contains the focused fleet-traffic cases for:

- a lower-ID robot keeping its natural route and waiting while a higher-ID robot detours;
- the rigid replenishment-row case where the higher-ID robot clears laterally across multiple replans;
- both robots carrying south-docked pallets beside a pallet wall, proving the higher-ID complete footprint does not distort the lower-ID preferred route while the higher-ID planner still respects the lower-ID complete footprint;
- a lower-ID robot waiting while a higher-ID robot performs repeated picks, then proceeding after that robot starts clearing;
- two-robot and five-robot first-ten-order integration/replay checks.

`tests/test_aisle_solver.py` adds warehouse-specific coordination regressions for:

- R2 and R4 finishing beside neighboring pallets, R2 selecting R4's side while R4 skips the pallet beside higher-priority R2, moves on, and later reacquires the skipped pallet through the final aisle rescan;
- R4 retaining an active P4 claim and continuing to pick while higher-priority R2 drives onto the opposite side of that pallet.

`tests/test_scheduler.py` is limited to the one-step reservation primitives: rigid-footprint cell reservations, waits, and edge conflicts. The old timed-A* scheduler tests were removed with the old planner.

See `tests/README.md` for a file-by-file test guide.

## Current optimization boundary

Correctness first:

- preserve simulator legality;
- preserve full rigid footprints;
- preserve pallet-home invariants;
- preserve deterministic robot-ID priority;
- preserve active pallet claims;
- preserve FIFO assignment and exact fulfillment;
- change one optimization layer at a time and measure it independently.

The current traffic planner should remain simple unless a reproducible failure demonstrates that a new movement rule is actually required. Warehouse-specific task conflicts should be solved in the task/aisle layer when possible rather than by adding traffic exceptions.

## Planned next experiment: soft traffic guides

Soft directional traffic is intentionally deferred until the simple one-step scheduler has a measured baseline.

Candidate preferred lanes remain:

- north/up: `x=(8,15,22,29,36,43)`;
- south/down: `x=(13,20,27,34,41,48)`;
- east/right center lane: `y=21`;
- west/left center lane: `y=18`.

These should be **soft costs**, not hard one-way constraints. The experiment should answer one clean question: does a small preferred-direction cost reduce head-on encounters and makespan without materially increasing travel distance?

## Architecture summary

> Aisle planning decides what to service, including temporary priority-aware pallet availability. Spatial A* decides the preferred route through the current warehouse geometry. The fleet scheduler commits only one footprint-safe move per robot in ID order. The simulator decides whether the complete timestep is legal. Then everything traffic-related is recomputed from the new real world state.
