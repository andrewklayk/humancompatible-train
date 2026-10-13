# E0e/X6: E2a's income_pairwise problem, identical batches and identical initial weights by construction. Run in float64 here, unlike E2a's float32 pipeline, so the script keeps one precision policy and the bar can stay at rounding.

| problem | m | steps | batch size | max |dy|inf | max |dweights|inf | duals bitwise identical | cooper | torch | notes |
|---|---|---|---|---|---|---|---|---|---|
| income_pairwise | 30 | 200 | 48 | 0.000e+00 | 0.000e+00 | True | 1.0.1 | 2.10.0+cu128 | ACSIncome FL, sens=MARxSEX |
