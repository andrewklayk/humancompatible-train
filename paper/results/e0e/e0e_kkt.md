# E0e/K: both libraries run until the relative KKT residual clears 1e-06 or to 20000 steps, scored against the exact (x*, y*). This is what separates 'agree with each other' from 'agree and are right'. The gate is agreement plus an identical convergence iteration, so no fixed budget has to be defended; the residual a given problem reaches under one untuned step size is E0a's subject, and the spread here is its finding #4 again -- svm_iris (m=100) gets further than qp_active (m=5), because progress tracks the dual step against ||J||^2 rather than problem size.

| problem | steps run | cap | solved at (ours) | solved at (cooper) | relative KKT (ours) | relative KKT (cooper) | |difference| | ||y-y*||inf (ours) | ||y-y*||inf (cooper) | duals bitwise identical | cooper | torch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qp_active | 20000 | 20000 | None | None | 2.238e-03 | 2.238e-03 | 0.000e+00 | 6.660e-03 | 6.660e-03 | True | 1.0.1 | 2.10.0+cu128 |
| qp_inactive | 6300 | 20000 | 6300 | 6300 | 9.129e-07 | 9.129e-07 | 0.000e+00 | 2.174e-06 | 2.174e-06 | True | 1.0.1 | 2.10.0+cu128 |
| svm_iris | 17700 | 20000 | 17700 | 17700 | 9.488e-07 | 9.488e-07 | 0.000e+00 | 1.335e-06 | 1.335e-06 | True | 1.0.1 | 2.10.0+cu128 |
| qp_equality | 3500 | 20000 | 3500 | 3500 | 9.502e-07 | 9.502e-07 | 0.000e+00 | 1.935e-06 | 1.935e-06 | True | 1.0.1 | 2.10.0+cu128 |
