# E0a/C: one untuned configuration per method — primal step 1/(L_f + rho||J||^2), dual step 1/||J||^2, and a per-problem iteration budget of 700/dual_step so that every problem gets equal progress rather than equal iterations.

| problem | method | primal lr | dual step | status | relative KKT | ||y-y*||inf | max [c]+ | ||grad f + J'y||inf |
|---|---|---|---|---|---|---|---|---|
| qp_active | ALM (rho=0) | 2.547e-02 | 5.219e-02 | solved | 1.914e-08 | 5.697e-08 | 1.328e-09 | 4.552e-11 |
| qp_active | ALM (rho=1) | 1.712e-02 | 5.219e-02 | solved | 2.440e-08 | 7.262e-08 | 1.666e-09 | 2.209e-09 |
| qp_active | nuPI (rho=0) | 2.547e-02 | 5.219e-02 | solved | 2.222e-08 | 6.613e-08 | 1.541e-09 | 5.277e-11 |
| qp_active | nuPI (rho=1) | 1.712e-02 | 5.219e-02 | solved | 2.738e-08 | 8.149e-08 | 1.870e-09 | 2.479e-09 |
| qp_active | iALM | 1.712e-02 | 5.219e-02 | solved | 1.645e-08 | 4.898e-08 | 1.142e-09 | 1.139e-10 |
| qp_active | PBM | 1.712e-02 | 5.219e-02 | solved | 3.133e-15 | 9.326e-15 | 0.000e+00 | 3.553e-15 |
| qp_inactive | ALM (rho=0) | 3.142e-02 | 4.092e-02 | solved | 6.154e-15 | 1.465e-14 | 2.665e-15 | 3.553e-15 |
| qp_inactive | ALM (rho=1) | 1.777e-02 | 4.092e-02 | solved | 5.408e-15 | 1.288e-14 | 2.442e-15 | 4.441e-15 |
| qp_inactive | nuPI (rho=0) | 3.142e-02 | 4.092e-02 | solved | 8.112e-15 | 1.932e-14 | 3.331e-15 | 3.553e-15 |
| qp_inactive | nuPI (rho=1) | 1.777e-02 | 4.092e-02 | solved | 9.790e-15 | 2.331e-14 | 3.997e-15 | 7.105e-15 |
| qp_inactive | iALM | 1.777e-02 | 4.092e-02 | solved | 5.501e-15 | 1.310e-14 | 2.665e-15 | 2.665e-15 |
| qp_inactive | PBM | 1.777e-02 | 4.092e-02 | bounded | 1.250e-04 | 1.000e-04 | 0.000e+00 | 2.978e-04 |
| svm_iris | ALM (rho=0) | 1.000e+00 | 3.283e-03 | solved | 9.134e-14 | 1.285e-13 | 8.438e-15 | 0.000e+00 |
| svm_iris | ALM (rho=1) | 3.272e-03 | 3.283e-03 | solved | 4.644e-14 | 6.534e-14 | 8.438e-15 | 2.431e-14 |
| svm_iris | nuPI (rho=0) | 1.000e+00 | 3.283e-03 | bounded | 2.512e-01 | 3.534e-01 | 0.000e+00 | 8.327e-17 |
| svm_iris | nuPI (rho=1) | 3.272e-03 | 3.283e-03 | solved | 9.461e-14 | 1.331e-13 | 1.266e-14 | 2.831e-14 |
| svm_iris | iALM | 3.272e-03 | 3.283e-03 | solved | 4.080e-14 | 5.740e-14 | 8.438e-15 | 1.676e-14 |
| svm_iris | PBM | 3.272e-03 | 3.283e-03 | bounded | 4.466e-03 | 3.378e-03 | 0.000e+00 | 6.284e-03 |
| qp_nonconvex | ALM (rho=0) | 4.544e-01 | 5.000e-01 | diverged | 6.167e+08 | nan | 2.803e+08 | 6.167e+08 |
| qp_nonconvex | ALM (rho=1) | 2.381e-01 | 5.000e-01 | bounded | 8.427e+01 | nan | 4.706e+01 | 8.427e+01 |
| qp_nonconvex | nuPI (rho=0) | 4.544e-01 | 5.000e-01 | diverged | 3.159e+08 | nan | 1.436e+08 | 3.159e+08 |
| qp_nonconvex | nuPI (rho=1) | 2.381e-01 | 5.000e-01 | bounded | 4.643e+01 | nan | 2.998e+01 | 4.643e+01 |
| qp_nonconvex | iALM | 2.381e-01 | 5.000e-01 | diverged | 3.771e+08 | nan | 1.714e+08 | 3.771e+08 |
| qp_nonconvex | PBM | 2.381e-01 | 5.000e-01 | diverged | 9.848e+08 | nan | 9.848e+08 | 8.195e+08 |
