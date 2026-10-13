# E0a/O: outcome counts. On qp_nonconvex no convergence is claimed — a fixed-penalty Lagrangian surrogate is unbounded below on an indefinite objective, so failure there is a limit of applicability, not a defect.

| problem | convex | solved | bounded (finite, not a KKT point) | diverged | did not solve |
|---|---|---|---|---|---|
| qp_active | True | 4 | 2 | 0 | ALM (rho=0), ALM (rho=1) |
| qp_inactive | True | 6 | 0 | 0 | - |
| svm_iris | True | 6 | 0 | 0 | - |
| qp_nonconvex | False | 3 | 2 | 1 | ALM (rho=0), ALM (rho=1), iALM |
