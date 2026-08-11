# E0a/R: three exact reductions among four independently written classes, one step and along a trajectory.

| problem | reduction | surrogate difference | dual difference | duals bitwise identical | bar | max |surrogate difference| | max |dual difference| | steps with preconditions holding | precondition first broke at step | note |
|---|---|---|---|---|---|---|---|---|---|---|
| qp_active | R1  nuPI(kp=0) == ALM(rho=0) | 0.000e+00 | 0.000e+00 | True | bitwise | 5.684e-14 | 1.998e-15 | 200 | None | unconditional |
| qp_active | R2  iALM(sigma=1, gamma>>) == ALM(lr=beta, rho=beta) | 0.000e+00 | 0.000e+00 | True | bitwise | 5.684e-14 | 8.882e-16 | 200 | None | requires gamma >= beta*||c||, i.e. the safeguard does not bind |
| qp_active | R3  PBM(penalty_update='alm') == ALM((1-g)/r, 1/r) | 0.000e+00 | 0.000e+00 | True | <= 1.4e-11 | 4.263e-14 | 1.332e-15 | 200 | None | requires c/p >= -0.5 (the quad_log branch) and no range clamps |
| qp_inactive | R1  nuPI(kp=0) == ALM(rho=0) | 0.000e+00 | 0.000e+00 | True | bitwise | 2.132e-14 | 4.441e-16 | 200 | None | unconditional |
| qp_inactive | R2  iALM(sigma=1, gamma>>) == ALM(lr=beta, rho=beta) | 0.000e+00 | 0.000e+00 | True | bitwise | 2.842e-14 | 2.220e-16 | 200 | None | requires gamma >= beta*||c||, i.e. the safeguard does not bind |
| qp_inactive | R3  PBM(penalty_update='alm') == ALM((1-g)/r, 1/r) | 1.665e-16 | 1.110e-16 | False | <= 1.4e-11 | 1.421e-14 | 2.220e-16 | 6 | 6 | requires c/p >= -0.5 (the quad_log branch) and no range clamps |
| svm_iris | R1  nuPI(kp=0) == ALM(rho=0) | 0.000e+00 | 0.000e+00 | True | bitwise | 0.000e+00 | 0.000e+00 | 200 | None | unconditional |
| svm_iris | R2  iALM(sigma=1, gamma>>) == ALM(lr=beta, rho=beta) | 0.000e+00 | 0.000e+00 | True | bitwise | 0.000e+00 | 0.000e+00 | 200 | None | requires gamma >= beta*||c||, i.e. the safeguard does not bind |
| svm_iris | R3  PBM(penalty_update='alm') == ALM((1-g)/r, 1/r) | 0.000e+00 | 0.000e+00 | True | <= 1.4e-11 | 7.105e-15 | 2.220e-16 | 3 | 3 | requires c/p >= -0.5 (the quad_log branch) and no range clamps |
| qp_nonconvex | R1  nuPI(kp=0) == ALM(rho=0) | 0.000e+00 | 0.000e+00 | True | bitwise | 9.134e+47 | 1.208e-13 | 200 | None | unconditional |
| qp_nonconvex | R2  iALM(sigma=1, gamma>>) == ALM(lr=beta, rho=beta) | 0.000e+00 | 0.000e+00 | True | bitwise | 4.027e+08 | 1.421e-14 | 117 | 117 | requires gamma >= beta*||c||, i.e. the safeguard does not bind |
| qp_nonconvex | R3  PBM(penalty_update='alm') == ALM((1-g)/r, 1/r) | 0.000e+00 | 1.110e-16 | False | <= 1.4e-11 | 0.000e+00 | 2.220e-16 | 2 | 2 | requires c/p >= -0.5 (the quad_log branch) and no range clamps |
