# E0a/I1: constraints scaled by 7 must scale the multipliers by 1/7 and move nothing else.

| problem | method | alpha | excess drift (plain) | excess drift (scaled) |
|---|---|---|---|---|
| qp_active | ALM (rho=0) | 7.000e+00 | 0.000e+00 | 5.551e-17 |
| qp_active | ALM (rho=1) | 7.000e+00 | 0.000e+00 | 5.551e-17 |
| qp_active | nuPI (rho=0) | 7.000e+00 | 9.770e-15 | 6.839e-14 |
| qp_active | nuPI (rho=1) | 7.000e+00 | 9.770e-15 | 6.839e-14 |
| qp_active | iALM | 7.000e+00 | 8.882e-16 | 6.217e-15 |
| qp_active | PBM | 7.000e+00 | 4.441e-16 | 5.829e-16 |
| qp_inactive | ALM (rho=0) | 7.000e+00 | 0.000e+00 | 2.776e-17 |
| qp_inactive | ALM (rho=1) | 7.000e+00 | 0.000e+00 | 2.776e-17 |
| qp_inactive | nuPI (rho=0) | 7.000e+00 | 4.885e-15 | 3.419e-14 |
| qp_inactive | nuPI (rho=1) | 7.000e+00 | 4.885e-15 | 3.419e-14 |
| qp_inactive | iALM | 7.000e+00 | 2.220e-16 | 2.220e-16 |
| qp_inactive | PBM | 7.000e+00 | 0.000e+00 | 0.000e+00 |
| svm_iris | ALM (rho=0) | 7.000e+00 | 0.000e+00 | 6.939e-18 |
| svm_iris | ALM (rho=1) | 7.000e+00 | 0.000e+00 | 6.939e-18 |
| svm_iris | nuPI (rho=0) | 7.000e+00 | 1.221e-15 | 8.549e-15 |
| svm_iris | nuPI (rho=1) | 7.000e+00 | 1.221e-15 | 8.549e-15 |
| svm_iris | iALM | 7.000e+00 | 0.000e+00 | 1.041e-17 |
| svm_iris | PBM | 7.000e+00 | 0.000e+00 | 0.000e+00 |
