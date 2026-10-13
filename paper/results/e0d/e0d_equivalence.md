# E0d: 2 gloo ranks vs one process at the pooled batch, after 20 steps

| per-rank batch | pooled batch | steps | constraint | method | surrogate linear in c | duals identical across ranks | dual gap vs 1x pooled | param gap vs 1x pooled | dual at bound | state_dict round-trips |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 32 | 1 | mean | ALM (rho=0) | True | True | 0.000e+00 | 2.578e-17 | False | True |
| 16 | 32 | 1 | mean | ALM (rho=1) | False | True | 0.000e+00 | 4.157e-04 | False | True |
| 16 | 32 | 1 | mean | nuPI | True | True | 0.000e+00 | 2.578e-17 | False | True |
| 16 | 32 | 1 | mean | iALM | False | True | 0.000e+00 | 4.157e-04 | False | True |
| 16 | 32 | 1 | mean | PBM | False | True | 0.000e+00 | 4.144e-08 | False | True |
| 16 | 32 | 1 | ratio | ALM (rho=0) | True | True | 9.348e-04 | 2.614e-03 | False | True |
| 16 | 32 | 1 | ratio | ALM (rho=1) | False | True | 9.348e-04 | 2.614e-03 | False | True |
| 16 | 32 | 1 | ratio | nuPI | True | True | 9.348e-04 | 2.614e-03 | False | True |
| 16 | 32 | 1 | ratio | iALM | False | True | 1.521e-02 | 2.614e-03 | False | True |
| 16 | 32 | 1 | ratio | PBM | False | True | 0.000e+00 | 6.214e-07 | True | True |
| 16 | 32 | 20 | mean | ALM (rho=0) | True | True | 0.000e+00 | 9.219e-17 | False | True |
| 16 | 32 | 20 | mean | ALM (rho=1) | False | True | 3.477e-06 | 2.148e-03 | False | True |
| 16 | 32 | 20 | mean | nuPI | True | True | 0.000e+00 | 2.306e-17 | False | True |
| 16 | 32 | 20 | mean | iALM | False | True | 3.879e-05 | 1.753e-03 | False | True |
| 16 | 32 | 20 | mean | PBM | False | True | 1.915e-08 | 7.528e-07 | False | True |
| 16 | 32 | 20 | ratio | ALM (rho=0) | True | True | 0.000e+00 | 9.051e-03 | True | True |
| 16 | 32 | 20 | ratio | ALM (rho=1) | False | True | 0.000e+00 | 9.051e-03 | True | True |
| 16 | 32 | 20 | ratio | nuPI | True | True | 0.000e+00 | 8.726e-03 | True | True |
| 16 | 32 | 20 | ratio | iALM | False | True | 0.000e+00 | 1.546e-03 | True | True |
| 16 | 32 | 20 | ratio | PBM | False | True | 0.000e+00 | 5.450e-06 | True | True |
| 64 | 128 | 1 | mean | ALM (rho=0) | True | True | 0.000e+00 | 0.000e+00 | False | True |
| 64 | 128 | 1 | mean | ALM (rho=1) | False | True | 0.000e+00 | 4.000e-06 | False | True |
| 64 | 128 | 1 | mean | nuPI | True | True | 0.000e+00 | 0.000e+00 | False | True |
| 64 | 128 | 1 | mean | iALM | False | True | 1.800e-16 | 4.000e-06 | False | True |
| 64 | 128 | 1 | mean | PBM | False | True | 0.000e+00 | 3.993e-10 | False | True |
| 64 | 128 | 1 | ratio | ALM (rho=0) | True | True | 3.352e-05 | 6.547e-04 | False | True |
| 64 | 128 | 1 | ratio | ALM (rho=1) | False | True | 3.352e-05 | 6.547e-04 | False | True |
| 64 | 128 | 1 | ratio | nuPI | True | True | 3.352e-05 | 6.547e-04 | False | True |
| 64 | 128 | 1 | ratio | iALM | False | True | 5.013e-04 | 6.547e-04 | False | True |
| 64 | 128 | 1 | ratio | PBM | False | True | 0.000e+00 | 1.174e-07 | True | True |
| 64 | 128 | 20 | mean | ALM (rho=0) | True | True | 0.000e+00 | 6.090e-17 | False | True |
| 64 | 128 | 20 | mean | ALM (rho=1) | False | True | 2.167e-06 | 1.219e-04 | False | True |
| 64 | 128 | 20 | mean | nuPI | True | True | 3.333e-16 | 1.218e-16 | False | True |
| 64 | 128 | 20 | mean | iALM | False | True | 6.357e-06 | 1.179e-04 | False | True |
| 64 | 128 | 20 | mean | PBM | False | True | 1.795e-08 | 1.469e-07 | False | True |
| 64 | 128 | 20 | ratio | ALM (rho=0) | True | True | 9.133e-03 | 4.138e-03 | False | True |
| 64 | 128 | 20 | ratio | ALM (rho=1) | False | True | 9.133e-03 | 4.138e-03 | False | True |
| 64 | 128 | 20 | ratio | nuPI | True | True | 1.138e-02 | 4.073e-03 | False | True |
| 64 | 128 | 20 | ratio | iALM | False | True | 0.000e+00 | 6.281e-04 | True | True |
| 64 | 128 | 20 | ratio | PBM | False | True | 0.000e+00 | 1.380e-06 | True | True |
| 256 | 512 | 1 | mean | ALM (rho=0) | True | True | 0.000e+00 | 5.200e-17 | False | True |
| 256 | 512 | 1 | mean | ALM (rho=1) | False | True | 0.000e+00 | 2.293e-05 | False | True |
| 256 | 512 | 1 | mean | nuPI | True | True | 0.000e+00 | 5.200e-17 | False | True |
| 256 | 512 | 1 | mean | iALM | False | True | 1.677e-16 | 2.293e-05 | False | True |
| 256 | 512 | 1 | mean | PBM | False | True | 0.000e+00 | 2.282e-09 | False | True |
| 256 | 512 | 1 | ratio | ALM (rho=0) | True | True | 3.436e-05 | 6.956e-05 | False | True |
| 256 | 512 | 1 | ratio | ALM (rho=1) | False | True | 3.436e-05 | 6.956e-05 | False | True |
| 256 | 512 | 1 | ratio | nuPI | True | True | 3.436e-05 | 6.956e-05 | False | True |
| 256 | 512 | 1 | ratio | iALM | False | True | 6.208e-04 | 6.956e-05 | False | True |
| 256 | 512 | 1 | ratio | PBM | False | True | 0.000e+00 | 1.095e-08 | True | True |
| 256 | 512 | 20 | mean | ALM (rho=0) | True | True | 0.000e+00 | 1.890e-16 | False | True |
| 256 | 512 | 20 | mean | ALM (rho=1) | False | True | 1.243e-05 | 1.785e-04 | False | True |
| 256 | 512 | 20 | mean | nuPI | True | True | 0.000e+00 | 1.891e-16 | False | True |
| 256 | 512 | 20 | mean | iALM | False | True | 2.796e-05 | 1.060e-04 | False | True |
| 256 | 512 | 20 | mean | PBM | False | True | 2.259e-07 | 6.970e-07 | False | True |
| 256 | 512 | 20 | ratio | ALM (rho=0) | True | True | 0.000e+00 | 3.836e-04 | True | True |
| 256 | 512 | 20 | ratio | ALM (rho=1) | False | True | 0.000e+00 | 3.836e-04 | True | True |
| 256 | 512 | 20 | ratio | nuPI | True | True | 0.000e+00 | 3.777e-04 | True | True |
| 256 | 512 | 20 | ratio | iALM | False | True | 0.000e+00 | 8.293e-05 | True | True |
| 256 | 512 | 20 | ratio | PBM | False | True | 0.000e+00 | 2.663e-07 | True | True |
