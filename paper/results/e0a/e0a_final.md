# E0a: multiplier recovery after 20000 primal iterations

| problem | method | dual config | primal lr | status | ||y-y*||inf | ||y-y*||2/||y*||2 | max [c]+ | ||grad f + J'y||inf | f - f* |
|---|---|---|---|---|---|---|---|---|---|
| qp_active | ALM (rho=0) | lr=0.3 | 2.547e-02 | solved | 1.288e-14 | 4.956e-15 | 0.000e+00 | 3.553e-15 | -1.421e-14 |
| qp_active | ALM (rho=1) | lr=0.3 | 1.712e-02 | solved | 1.132e-14 | 4.530e-15 | 0.000e+00 | 3.553e-15 | -1.421e-14 |
| qp_active | ALM (rho=10) | lr=0.3 | 4.332e-03 | solved | 1.288e-14 | 5.369e-15 | 0.000e+00 | 2.132e-14 | -2.842e-14 |
| qp_active | nuPI (rho=0) | ki=0.3, kp=3 | 2.547e-02 | solved | 4.663e-15 | 1.520e-15 | 0.000e+00 | 3.553e-15 | -1.421e-14 |
| qp_active | nuPI (rho=1) | ki=0.3, kp=3 | 1.712e-02 | solved | 6.661e-15 | 2.238e-15 | 0.000e+00 | 3.553e-15 | -2.842e-14 |
| qp_active | PBM | gamma=0.3 | 1.712e-02 | solved | 5.107e-15 | 2.460e-15 | 0.000e+00 | 5.329e-15 | -1.421e-14 |
| qp_active | iALM (beta=0.1) | beta=0.1 | 2.429e-02 | solved | 5.884e-14 | 2.208e-14 | 8.882e-16 | 1.776e-15 | -2.842e-14 |
| qp_active | iALM (beta=1) | beta=1 | 1.712e-02 | solved | 1.110e-14 | 4.494e-15 | 0.000e+00 | 5.329e-15 | -1.421e-14 |
| qp_active | iALM (beta=10) | beta=10 | 4.332e-03 | solved | 9.326e-15 | 4.348e-15 | 0.000e+00 | 2.132e-14 | -1.421e-14 |
| qp_inactive | ALM (rho=0) | lr=0.3 | 3.142e-02 | solved | 1.998e-15 | 1.048e-15 | 4.441e-16 | 5.329e-15 | 7.105e-15 |
| qp_inactive | ALM (rho=1) | lr=0.3 | 1.777e-02 | solved | 1.776e-15 | 9.374e-16 | 0.000e+00 | 1.776e-15 | 1.421e-14 |
| qp_inactive | ALM (rho=10) | lr=1 | 3.620e-03 | solved | 4.219e-15 | 2.172e-15 | 0.000e+00 | 2.132e-14 | 7.105e-15 |
| qp_inactive | nuPI (rho=0) | ki=0.3, kp=0.3 | 3.142e-02 | solved | 1.332e-15 | 6.820e-16 | 0.000e+00 | 5.329e-15 | 7.105e-15 |
| qp_inactive | nuPI (rho=1) | ki=1, kp=1 | 1.777e-02 | solved | 1.110e-15 | 5.684e-16 | 0.000e+00 | 1.776e-15 | 0.000e+00 |
| qp_inactive | PBM | gamma=0.1 | 1.777e-02 | bounded | 1.000e-04 | 1.043e-04 | 0.000e+00 | 2.978e-04 | 7.652e-10 |
| qp_inactive | iALM (beta=0.1) | beta=0.1 | 2.918e-02 | solved | 3.331e-15 | 1.720e-15 | 6.661e-16 | 1.776e-15 | 0.000e+00 |
| qp_inactive | iALM (beta=1) | beta=1 | 1.777e-02 | solved | 2.887e-15 | 1.482e-15 | 2.220e-16 | 7.105e-15 | 0.000e+00 |
| qp_inactive | iALM (beta=10) | beta=10 | 3.620e-03 | solved | 4.441e-15 | 2.595e-15 | 2.220e-16 | 2.132e-14 | 7.105e-15 |
| svm_iris | ALM (rho=0) | lr=0.01 | 1.000e+00 | solved | 9.812e-07 | 2.192e-06 | 5.642e-08 | 5.776e-12 | -1.045e-08 |
| svm_iris | ALM (rho=1) | lr=0.1 | 3.272e-03 | solved | 3.042e-14 | 8.039e-14 | 3.175e-14 | 5.917e-14 | 1.932e-14 |
| svm_iris | ALM (rho=10) | lr=0.3 | 3.282e-04 | bounded | 2.882e-03 | 5.949e-03 | 4.125e-05 | 8.753e-04 | 7.392e-06 |
| svm_iris | nuPI (rho=0) | ki=0.01, kp=0 | 1.000e+00 | solved | 9.812e-07 | 2.192e-06 | 5.642e-08 | 5.776e-12 | -1.045e-08 |
| svm_iris | nuPI (rho=1) | ki=0.1, kp=0.1 | 3.272e-03 | solved | 5.745e-15 | 1.629e-14 | 4.441e-16 | 1.243e-14 | -4.441e-16 |
| svm_iris | PBM | gamma=0.3 | 3.272e-03 | bounded | 3.378e-03 | 6.952e-03 | 0.000e+00 | 6.284e-03 | 2.618e-07 |
| svm_iris | iALM (beta=0.1) | beta=0.1 | 3.178e-02 | solved | 2.054e-15 | 4.205e-15 | 0.000e+00 | 8.327e-16 | 1.110e-16 |
| svm_iris | iALM (beta=1) | beta=1 | 3.272e-03 | solved | 1.887e-14 | 3.557e-14 | 8.660e-15 | 2.559e-14 | -8.882e-16 |
| svm_iris | iALM (beta=10) | beta=10 | 3.282e-04 | bounded | 6.043e-03 | 1.296e-02 | 3.530e-05 | 1.345e-03 | 1.518e-06 |
| qp_nonconvex | ALM (rho=0) | lr=0.0001 | 4.544e-01 | diverged | nan | nan | 2.323e+28 | 5.111e+28 | nan |
| qp_nonconvex | ALM (rho=1) | lr=1 | 2.381e-01 | bounded | nan | nan | 3.568e+01 | 9.795e+01 | nan |
| qp_nonconvex | ALM (rho=10) | lr=1 | 4.504e-02 | solved | nan | nan | 2.220e-16 | 2.442e-15 | nan |
| qp_nonconvex | nuPI (rho=0) | ki=1, kp=1 | 4.544e-01 | bounded | nan | nan | 2.586e+00 | 3.910e+00 | nan |
| qp_nonconvex | nuPI (rho=1) | ki=0.3, kp=3 | 2.381e-01 | solved | nan | nan | 4.441e-16 | 8.882e-16 | nan |
| qp_nonconvex | PBM | gamma=0.1 | 2.381e-01 | diverged | nan | nan | 1.448e+109 | 3.177e+109 | nan |
| qp_nonconvex | iALM (beta=0.1) | beta=0.1 | 4.166e-01 | diverged | nan | nan | 2.058e+25 | 4.529e+25 | nan |
| qp_nonconvex | iALM (beta=1) | beta=1 | 2.381e-01 | diverged | nan | nan | 2.339e+20 | 5.148e+20 | nan |
| qp_nonconvex | iALM (beta=10) | beta=10 | 4.504e-02 | solved | nan | nan | 8.882e-16 | 8.660e-15 | nan |
