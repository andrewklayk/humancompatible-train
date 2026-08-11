# E0a/F: one forward_update from the exact KKT point. 'excess drift' is the multiplier movement a method is answerable for, once its own declared lower_bound is allowed.

| problem | method | grad_x | dual drift | lower_bound | unavoidable deviation | excess drift | tolerance | grad_x tolerance |
|---|---|---|---|---|---|---|---|---|
| qp_active | ALM (rho=0) | 7.105e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 4.230e-14 | 1.421e-11 |
| qp_active | ALM (rho=1) | 7.105e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 4.230e-14 | 1.421e-11 |
| qp_active | nuPI (rho=0) | 7.105e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 4.230e-14 | 1.421e-11 |
| qp_active | nuPI (rho=1) | 7.105e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 4.230e-14 | 1.421e-11 |
| qp_active | iALM | 7.105e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 4.230e-14 | 1.421e-11 |
| qp_active | PBM | 7.105e-15 | 4.441e-16 | 1.000e-04 | 0.000e+00 | 4.441e-16 | 4.230e-14 | 1.421e-11 |
| qp_inactive | ALM (rho=0) | 3.553e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 3.384e-14 | 1.421e-11 |
| qp_inactive | ALM (rho=1) | 3.553e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 3.384e-14 | 1.421e-11 |
| qp_inactive | nuPI (rho=0) | 3.553e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 3.384e-14 | 1.421e-11 |
| qp_inactive | nuPI (rho=1) | 3.553e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 3.384e-14 | 1.421e-11 |
| qp_inactive | iALM | 3.553e-15 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 3.384e-14 | 1.421e-11 |
| qp_inactive | PBM | 1.003e-04 | 1.000e-04 | 1.000e-04 | 1.000e-04 | 0.000e+00 | 3.384e-14 | 4.943e-03 |
| svm_iris | ALM (rho=0) | 4.441e-16 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 1.999e-14 | 1.421e-11 |
| svm_iris | ALM (rho=1) | 4.441e-16 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 1.999e-14 | 1.421e-11 |
| svm_iris | nuPI (rho=0) | 4.441e-16 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 1.999e-14 | 1.421e-11 |
| svm_iris | nuPI (rho=1) | 4.441e-16 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 1.999e-14 | 1.421e-11 |
| svm_iris | iALM | 4.441e-16 | 0.000e+00 | 0 | 0.000e+00 | 0.000e+00 | 1.999e-14 | 1.421e-11 |
| svm_iris | PBM | 3.188e-03 | 1.000e-04 | 1.000e-04 | 1.000e-04 | 0.000e+00 | 1.999e-14 | 1.745e-02 |
