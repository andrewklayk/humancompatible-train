# E0e/X5: started from y0=0.5 so the projection actually fires. Our clamp_ and Cooper's relu must pin the same multipliers to exactly 0.0.

| problem | y0 | exact zeros (ours) | exact zeros (cooper) | zero sets identical | duals bitwise identical | exact zeros in y* | cooper | torch |
|---|---|---|---|---|---|---|---|---|
| qp_inactive | 5.000e-01 | 4 | 4 | True | True | 4 | 1.0.1 | 2.10.0+cu128 |
| svm_iris | 5.000e-01 | 100 | 100 | True | True | 96 | 1.0.1 | 2.10.0+cu128 |
