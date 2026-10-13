# E0a/F: one forward_update from the exact KKT point. 'excess drift' is the multiplier movement a method is answerable for, once its own declared lower_bound is allowed.

| problem | method | grad_x | dual drift | lower_bound | unavoidable deviation | excess drift | tolerance | grad_x tolerance |
|---|---|---|---|---|---|---|---|---|
| qp_active | ALM (rho=0) | 7.105e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 4.230e-14 | 1.421e-11 |
| qp_active | ALM (rho=1) | 7.105e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 4.230e-14 | 1.421e-11 |
| qp_active | nuPI (rho=0) | 1.421e-14 | 9.770e-15 | 0 | 0.000e+00 | 9.770e-15 | 4.230e-14 | 1.421e-11 |
| qp_active | nuPI (rho=1) | 1.776e-14 | 9.770e-15 | 0 | 0.000e+00 | 9.770e-15 | 4.230e-14 | 1.421e-11 |
| qp_active | iALM | 1.066e-14 | 8.882e-16 | 0 | 0.000e+00 | 8.882e-16 | 4.230e-14 | 1.421e-11 |
| qp_active | PBM | 7.105e-15 | 4.441e-16 | 1.000e-09 | 0.000e+00 | 4.441e-16 | 4.230e-14 | 1.421e-11 |
| qp_inactive | ALM (rho=0) | 3.553e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 3.384e-14 | 1.421e-11 |
| qp_inactive | ALM (rho=1) | 3.553e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 3.384e-14 | 1.421e-11 |
| qp_inactive | nuPI (rho=0) | 9.326e-15 | 4.885e-15 | 0 | 0.000e+00 | 4.885e-15 | 3.384e-14 | 1.421e-11 |
| qp_inactive | nuPI (rho=1) | 9.770e-15 | 4.885e-15 | 0 | 0.000e+00 | 4.885e-15 | 3.384e-14 | 1.421e-11 |
| qp_inactive | iALM | 3.553e-15 | 2.220e-16 | 0 | 0.000e+00 | 2.220e-16 | 3.384e-14 | 1.421e-11 |
| qp_inactive | PBM | 1.003e-09 | 1.000e-09 | 1.000e-09 | 1.000e-09 | 0.000e+00 | 3.384e-14 | 4.943e-08 |
| svm_iris | ALM (rho=0) | 4.441e-16 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 1.999e-14 | 1.421e-11 |
| svm_iris | ALM (rho=1) | 4.441e-16 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 1.999e-14 | 1.421e-11 |
| svm_iris | nuPI (rho=0) | 1.998e-15 | 1.221e-15 | 0 | 0.000e+00 | 1.221e-15 | 1.999e-14 | 1.421e-11 |
| svm_iris | nuPI (rho=1) | 2.165e-15 | 1.221e-15 | 0 | 0.000e+00 | 1.221e-15 | 1.999e-14 | 1.421e-11 |
| svm_iris | iALM | 4.441e-16 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 1.999e-14 | 1.421e-11 |
| svm_iris | PBM | 3.188e-08 | 1.000e-09 | 1.000e-09 | 1.000e-09 | 0.000e+00 | 1.999e-14 | 1.745e-07 |
