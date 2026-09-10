# E0a/C: convergence certificates. Every step size is a fixed constant — primal SGD at 0.005, ALM at its shipped lr=0.01, the others at their published values — and each run stops when the relative KKT residual clears 1e-06 or at 25000 iterations. The question is whether convergence happens under a reasonable configuration, not how fast; the iteration counts are context, not a ranking.

| problem | method | primal lr | iterations | hit cap | status | relative KKT | ||y-y*||inf | max [c]+ | ||grad f + J'y||inf |
|---|---|---|---|---|---|---|---|---|---|
| qp_active | ALM (rho=0) | 5.000e-03 | 25000 | True | bounded | 6.967e-04 | 2.074e-03 | 4.838e-05 | 2.639e-06 |
| qp_active | ALM (rho=1) | 5.000e-03 | 25000 | True | bounded | 8.409e-04 | 2.503e-03 | 5.740e-05 | 7.589e-05 |
| qp_active | nuPI (rho=0) | 5.000e-03 | 550 | False | solved | 8.726e-07 | 2.597e-06 | 6.135e-08 | 5.236e-07 |
| qp_active | nuPI (rho=1) | 5.000e-03 | 550 | False | solved | 6.210e-07 | 1.848e-06 | 4.458e-08 | 3.694e-07 |
| qp_active | iALM | 5.000e-03 | 426 | False | solved | 7.461e-07 | 2.221e-06 | 9.038e-08 | 8.948e-07 |
| qp_active | PBM | 5.000e-03 | 1013 | False | solved | 9.400e-07 | 2.798e-06 | 1.028e-07 | 5.436e-07 |
| qp_inactive | ALM (rho=0) | 5.000e-03 | 6300 | False | solved | 9.129e-07 | 2.174e-06 | 4.600e-07 | 5.649e-08 |
| qp_inactive | ALM (rho=1) | 5.000e-03 | 7700 | False | solved | 8.754e-07 | 2.085e-06 | 3.624e-07 | 3.105e-07 |
| qp_inactive | nuPI (rho=0) | 5.000e-03 | 283 | False | solved | 8.870e-07 | 2.178e-08 | 2.124e-09 | 2.112e-06 |
| qp_inactive | nuPI (rho=1) | 5.000e-03 | 283 | False | solved | 8.936e-07 | 2.410e-08 | 2.371e-09 | 2.128e-06 |
| qp_inactive | iALM | 5.000e-03 | 330 | False | solved | 4.994e-07 | 1.189e-06 | 0.000e+00 | 1.105e-06 |
| qp_inactive | PBM | 5.000e-03 | 400 | False | solved | 8.898e-07 | 2.119e-06 | 1.176e-07 | 1.747e-06 |
| svm_iris | ALM (rho=0) | 5.000e-03 | 17700 | False | solved | 9.488e-07 | 1.335e-06 | 8.794e-08 | 5.036e-08 |
| svm_iris | ALM (rho=1) | 5.000e-03 | 18200 | False | solved | 9.959e-07 | 1.401e-06 | 8.924e-08 | 1.019e-07 |
| svm_iris | nuPI (rho=0) | 5.000e-03 | 2900 | False | solved | 3.529e-07 | 4.965e-07 | 1.125e-07 | 4.044e-07 |
| svm_iris | nuPI (rho=1) | 5.000e-03 | 3200 | False | solved | 2.789e-07 | 3.924e-07 | 4.010e-08 | 1.195e-07 |
| svm_iris | iALM | 5.000e-03 | 5700 | False | solved | 2.817e-07 | 3.963e-07 | 2.509e-07 | 3.532e-07 |
| svm_iris | PBM | 5.000e-03 | 8160 | False | solved | 5.465e-07 | 4.688e-07 | 5.465e-07 | 7.068e-07 |
| qp_nonconvex | ALM (rho=0) | 5.000e-03 | 2700 | False | diverged | 2.672e+08 | nan | 1.214e+08 | 2.672e+08 |
| qp_nonconvex | ALM (rho=1) | 5.000e-03 | 25000 | True | bounded | 1.967e+02 | nan | 5.414e+01 | 1.967e+02 |
| qp_nonconvex | nuPI (rho=0) | 5.000e-03 | 900 | False | solved | 1.668e-07 | nan | 3.536e-08 | 1.668e-07 |
| qp_nonconvex | nuPI (rho=1) | 5.000e-03 | 869 | False | solved | 5.378e-07 | nan | 5.814e-08 | 5.378e-07 |
| qp_nonconvex | iALM | 5.000e-03 | 25000 | True | bounded | 3.263e+00 | nan | 2.897e-01 | 3.263e+00 |
| qp_nonconvex | PBM | 5.000e-03 | 11000 | False | solved | 7.904e-07 | nan | 7.904e-07 | 5.997e-07 |
