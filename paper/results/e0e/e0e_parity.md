# E0e/X: ALM(penalty=0) against Cooper's Lagrangian + SGD-ascent multiplier, 200 steps from a common start. Our forward_update uses post-update multipliers, so it pairs with AlternatingDualPrimalOptimizer; the split API pairs with SimultaneousOptimizer; and the split API with the constraints re-evaluated after primal.step() pairs with AlternatingPrimalDualOptimizer, the one Cooper's docs recommend -- reached without a new entry point, at the cost of a second constraint evaluation per step. The last two pairings are the controls that make the first three non-vacuous. Only the duals were expected to agree bitwise; that the parameters do as well says torch's einsum and matmul reductions coincide at these sizes, which is a fact about the kernels rather than about either library.

| pairing | problem | m | steps | constraint evals/step | max |dy|inf | max |dx|inf | duals bitwise identical | params bitwise identical | cooper | torch |
|---|---|---|---|---|---|---|---|---|---|---|
| forward_update <-> AlternatingDualPrimal | qp_active | 5 | 200 | 1 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| forward_update <-> AlternatingDualPrimal | qp_inactive | 6 | 200 | 1 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| forward_update <-> AlternatingDualPrimal | qp_equality_reduced | 6 | 200 | 1 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| forward_update <-> AlternatingDualPrimal | svm_iris | 100 | 200 | 1 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| forward_update <-> AlternatingDualPrimal | qp_equality | 3 | 200 | 1 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| split forward/update <-> Simultaneous | qp_active | 5 | 200 | 1 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| split forward/update <-> Simultaneous | qp_inactive | 6 | 200 | 1 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| split forward/update <-> Simultaneous | qp_equality_reduced | 6 | 200 | 1 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| split forward/update <-> Simultaneous | svm_iris | 100 | 200 | 1 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| split forward/update <-> Simultaneous | qp_equality | 3 | 200 | 1 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| split + re-evaluated c <-> AlternatingPrimalDual | qp_active | 5 | 200 | 2 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| split + re-evaluated c <-> AlternatingPrimalDual | qp_inactive | 6 | 200 | 2 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| split + re-evaluated c <-> AlternatingPrimalDual | qp_equality_reduced | 6 | 200 | 2 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| split + re-evaluated c <-> AlternatingPrimalDual | svm_iris | 100 | 200 | 2 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| split + re-evaluated c <-> AlternatingPrimalDual | qp_equality | 3 | 200 | 2 | 0.000e+00 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| forward_update <-> Simultaneous (control) | qp_active | 5 | 200 | 1 | 9.768e-03 | 3.301e-03 | False | False | 1.0.1 | 2.10.0+cu128 |
| forward_update <-> Simultaneous (control) | qp_inactive | 6 | 200 | 1 | 5.421e-03 | 1.906e-03 | False | False | 1.0.1 | 2.10.0+cu128 |
| forward_update <-> Simultaneous (control) | qp_equality_reduced | 6 | 200 | 1 | 9.796e-03 | 6.384e-03 | False | False | 1.0.1 | 2.10.0+cu128 |
| forward_update <-> Simultaneous (control) | svm_iris | 100 | 200 | 1 | 6.653e-02 | 5.620e-02 | False | False | 1.0.1 | 2.10.0+cu128 |
| forward_update <-> Simultaneous (control) | qp_equality | 3 | 200 | 1 | 9.826e-03 | 6.384e-03 | False | False | 1.0.1 | 2.10.0+cu128 |
| split + re-evaluated c <-> AlternatingDualPrimal (control) | qp_active | 5 | 200 | 2 | 1.929e-02 | 4.478e-03 | False | False | 1.0.1 | 2.10.0+cu128 |
| split + re-evaluated c <-> AlternatingDualPrimal (control) | qp_inactive | 6 | 200 | 2 | 1.111e-02 | 3.495e-03 | False | False | 1.0.1 | 2.10.0+cu128 |
| split + re-evaluated c <-> AlternatingDualPrimal (control) | qp_equality_reduced | 6 | 200 | 2 | 5.332e-02 | 1.371e-02 | False | False | 1.0.1 | 2.10.0+cu128 |
| split + re-evaluated c <-> AlternatingDualPrimal (control) | svm_iris | 100 | 200 | 2 | 0.000e+00 | 3.843e-02 | True | False | 1.0.1 | 2.10.0+cu128 |
| split + re-evaluated c <-> AlternatingDualPrimal (control) | qp_equality | 3 | 200 | 2 | 5.332e-02 | 1.372e-02 | False | False | 1.0.1 | 2.10.0+cu128 |
