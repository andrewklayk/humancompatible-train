# E0e/X9: the practical reading of X8. Each ordering runs from a common y0 = 0 until the relative KKT residual clears 1e-06 or to 20000 steps. Run on our side only, which X1/X2/X7 licenses: every ordering is bitwise its Cooper counterpart, so this is a statement about both libraries. The column that decides the question is 'constraint evals to tolerance', not 'solved at': primal-dual re-evaluates the constraints at the new iterate, so an equal iteration count is twice the work.

| problem | ordering | cooper counterpart | constraint evals/step | steps run | solved at | constraint evals to tolerance | relative KKT | ||y-y*||inf | cooper | torch |
|---|---|---|---|---|---|---|---|---|---|---|
| qp_active | dual_primal | AlternatingDualPrimal | 1 | 20000 | None | None | 2.238e-03 | 6.660e-03 | 1.0.1 | 2.10.0+cu128 |
| qp_active | simultaneous | Simultaneous | 1 | 20000 | None | None | 2.232e-03 | 6.644e-03 | 1.0.1 | 2.10.0+cu128 |
| qp_active | primal_dual | AlternatingPrimalDual | 2 | 20000 | None | None | 2.275e-03 | 6.771e-03 | 1.0.1 | 2.10.0+cu128 |
| qp_inactive | dual_primal | AlternatingDualPrimal | 1 | 6300 | 6300 | 6300 | 9.129e-07 | 2.174e-06 | 1.0.1 | 2.10.0+cu128 |
| qp_inactive | simultaneous | Simultaneous | 1 | 6300 | 6300 | 6300 | 8.865e-07 | 2.111e-06 | 1.0.1 | 2.10.0+cu128 |
| qp_inactive | primal_dual | AlternatingPrimalDual | 2 | 6300 | 6300 | 12600 | 9.195e-07 | 2.190e-06 | 1.0.1 | 2.10.0+cu128 |
| svm_iris | dual_primal | AlternatingDualPrimal | 1 | 17700 | 17700 | 17700 | 9.488e-07 | 1.335e-06 | 1.0.1 | 2.10.0+cu128 |
| svm_iris | simultaneous | Simultaneous | 1 | 17700 | 17700 | 17700 | 9.542e-07 | 1.342e-06 | 1.0.1 | 2.10.0+cu128 |
| svm_iris | primal_dual | AlternatingPrimalDual | 2 | 17700 | 17700 | 35400 | 9.488e-07 | 1.335e-06 | 1.0.1 | 2.10.0+cu128 |
| qp_equality | dual_primal | AlternatingDualPrimal | 1 | 3500 | 3500 | 3500 | 9.502e-07 | 1.935e-06 | 1.0.1 | 2.10.0+cu128 |
| qp_equality | simultaneous | Simultaneous | 1 | 3500 | 3500 | 3500 | 9.042e-07 | 1.841e-06 | 1.0.1 | 2.10.0+cu128 |
| qp_equality | primal_dual | AlternatingPrimalDual | 2 | 3500 | 3500 | 7000 | 9.756e-07 | 1.986e-06 | 1.0.1 | 2.10.0+cu128 |
