# E0b: best value reached per direction at n = 50, against the published f*. The reference solvers' own final values (Curtis & Que Tables 2-4) are NOT yet transcribed - see REFERENCE_SOLVERS_N50 in paper/problems/nonsmooth.py.

| problem | convex | f(x0) | f* published | ours/SVANO-BFGS f | ours/SVANO-BFGS calls | ours/SVANO-Bundle f | ours/SVANO-Bundle calls | ours/SVANO-GS f | ours/SVANO-GS calls |
|---|---|---|---|---|---|---|---|---|---|
| maxq | True | 2.500e+03 | 0.000e+00 | 2.324e-05 | 896 | 2.324e-05 | 1152 | 2.679e-05 | 1018 |
| mxhilb | True | 4.499e+00 | 0.000e+00 | 5.392e-08 | 373 | 5.312e-08 | 465 | 3.595e-08 | 520 |
| chained_lq | True | 4.900e+01 | -6.930e+01 | -6.930e+01 | 414 | -6.930e+01 | 547 | -6.930e+01 | 1270 |
| chained_cb3_1 | True | 9.800e+02 | 9.800e+01 | 9.800e+01 | 765 | 9.800e+01 | 978 | 9.800e+01 | 1023 |
| chained_cb3_2 | True | 9.800e+02 | 9.800e+01 | 9.800e+01 | 219 | 9.800e+01 | 519 | 9.800e+01 | 654 |
| active_faces | False | 3.932e+00 | 0.000e+00 | 7.993e-11 | 187 | 7.735e-12 | 90 | 1.207e-10 | 283 |
| brown_2 | False | 9.800e+01 | 0.000e+00 | 1.730e-07 | 605 | 2.968e-07 | 957 | 4.549e-06 | 2977 |
| chained_mifflin_2 | False | 2.328e+02 | -3.480e+01 | -3.478e+01 | 866 | -3.479e+01 | 793 | -3.479e+01 | 1190 |
| chained_crescent_1 | False | 2.922e+02 | 0.000e+00 | 7.092e-09 | 304 | 3.533e-11 | 311 | 2.931e-10 | 276 |
| chained_crescent_2 | False | 2.922e+02 | 0.000e+00 | 2.179e-06 | 987 | 3.951e-06 | 993 | 2.733e-05 | 3095 |
