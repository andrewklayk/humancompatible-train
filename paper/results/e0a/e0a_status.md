# E0a/O: outcome counts. On qp_nonconvex no convergence is claimed — a fixed-penalty Lagrangian surrogate is unbounded below on an indefinite objective, so failure there is a limit of applicability, not a defect.

| problem | convex | solved | bounded (finite, not a KKT point) | diverged | did not solve |
|---|---|---|---|---|---|
| qp_active | True | 6 | 0 | 0 | - |
| qp_inactive | True | 5 | 1 | 0 | PBM |
| svm_iris | True | 4 | 2 | 0 | nuPI (rho=0), PBM |
| qp_nonconvex | False | 0 | 2 | 4 | ALM (rho=0), ALM (rho=1), nuPI (rho=0), nuPI (rho=1), iALM, PBM |
