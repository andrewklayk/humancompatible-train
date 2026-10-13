# E0e/O2: API surface. Line counts are of this script's own four adapter functions, which do the same job for each library across all three step orderings -- including, on the Cooper side, the compute_violations hook its primal-dual roll needs to avoid a redundant loss and graph. The third column records what Cooper's larger surface buys, because a table without it would be a strawman.

| aspect | ours | cooper | what cooper's extra surface buys |
|---|---|---|---|
| user code to pose the problem + one step, all three step orderings (non-comment lines) | 22 | 34 | - |
| choosing the primal/dual step ordering | which entry point the loop calls: forward_update; split forward/update; split with the constraint closure called again after primal.step() | which optimizer class is constructed: AlternatingDualPrimal, Simultaneous, AlternatingPrimalDual | the ordering cannot be got wrong by writing the loop wrong, and compute_violations lets the primal-dual re-evaluation skip the loss and the primal graph |
| objects a user constructs | 1 (ALM) + own torch optimizer + own loop | CMP subclass, Constraint, Multiplier, CMPState, ConstraintState, ConstrainedOptimizer | formulation and multiplier are swappable independently of the optimizer |
| who owns the training loop | the user (dual layer is a torch.optim.Optimizer over duals) | the library (optimizer.roll drives forward, both backwards, both steps) | primal/dual step ordering is a class choice, not an idiom the user has to get right |
| sparse / per-sample multipliers | no | IndexedMultiplier, ImplicitMultiplier, constraint_features | one multiplier per constraint instance at dataset scale |
| separating the differentiable surrogate from the measurement that drives the multiplier | no | ConstraintState.strict_violation | duals can follow an exact or non-differentiable statistic |
| constraint groups with independent dual step / bounds / bound | add_constraint_group, named or positional | one Constraint attribute per group | - |
| data-parallel reduction of constraint values | process_group=..., all_reduce(AVG) before the dual update | none | - |
| dual restart (arXiv:2208.04425 Eq. 6) | restart=True | not in 1.0.1 (0.x had dual_restarts) | - |
| penalty coefficient schedules | iALM's sigma; PBM's six penalty rules | PenaltyCoefficientUpdater (multiplicative, additive, feasibility-driven) | schedule is composable with any formulation |
