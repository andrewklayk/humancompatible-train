# E0e/X8: AlternatingPrimalDualOptimizer started from y0 = [lr*c(x0)]+ against forward_update started from y0 = 0. Both orderings iterate y <- P(y + lr*c(x_t)) over the same constraint sequence and differ only in whether c(x_0) is folded in before or after the first primal step, so handing primal-dual that one term erases the difference: identical parameters, and duals that are bitwise identical at an offset of one step. The same-index column is the control -- were it also zero the offset would be doing no work. The recommended optimizer is therefore not a different algorithm but the same one lagging by a dual step, and it pays a second constraint evaluation per step for the lag.

| problem | m | steps | max |dy|inf (offset 1) | max |dy|inf (same index) | max |dx|inf | duals bitwise identical (offset 1) | params bitwise identical | cooper | torch |
|---|---|---|---|---|---|---|---|---|---|
| qp_active | 5 | 200 | 0.000e+00 | 2.509e-02 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| qp_inactive | 6 | 200 | 0.000e+00 | 2.472e-02 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| qp_equality_reduced | 6 | 200 | 0.000e+00 | 7.326e-02 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| svm_iris | 100 | 200 | 0.000e+00 | 1.371e-02 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
| qp_equality | 3 | 200 | 0.000e+00 | 7.326e-02 | 0.000e+00 | True | True | 1.0.1 | 2.10.0+cu128 |
