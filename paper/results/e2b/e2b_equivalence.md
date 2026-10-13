# E2b: 2 gloo ranks vs one process at the pooled batch, after 10 steps

| constraint | m | bound | sharding | steps | method | surrogate linear in c | penalty live | duals identical across ranks | dual gap vs 1x pooled | param gap vs 1x pooled | dual at bound | state_dict round-trips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pairwise | 30 | 5.000e-02 | balanced | 1 | ALM (rho=0) | True | False | True | 0.000e+00 | 2.852e-17 | False | True |
| pairwise | 30 | 5.000e-02 | balanced | 1 | nuPI (rho=0) | True | False | True | 0.000e+00 | 2.852e-17 | False | True |
| pairwise | 30 | 5.000e-02 | balanced | 1 | ALM (rho=1) | False | False | True | 0.000e+00 | 2.852e-17 | False | True |
| pairwise | 30 | 5.000e-02 | balanced | 1 | iALM | False | False | True | 3.419e-16 | 5.706e-17 | False | True |
| pairwise | 30 | 5.000e-02 | balanced | 1 | PBM | False | False | True | 0.000e+00 | 1.665e-12 | True | True |
| pairwise | 30 | 5.000e-02 | balanced | 10 | ALM (rho=0) | True | True | True | 1.159e-16 | 1.105e-16 | False | True |
| pairwise | 30 | 5.000e-02 | balanced | 10 | nuPI (rho=0) | True | True | True | 2.312e-16 | 1.106e-16 | False | True |
| pairwise | 30 | 5.000e-02 | balanced | 10 | ALM (rho=1) | False | True | True | 8.550e-05 | 1.253e-03 | False | True |
| pairwise | 30 | 5.000e-02 | balanced | 10 | iALM | False | True | True | 1.198e-02 | 1.229e-03 | True | True |
| pairwise | 30 | 5.000e-02 | balanced | 10 | PBM | False | True | True | 0.000e+00 | 1.216e-11 | True | True |
| pairwise | 30 | 5.000e-02 | shuffled | 1 | ALM (rho=0) | True | False | True | 9.419e-04 | 8.320e-05 | False | True |
| pairwise | 30 | 5.000e-02 | shuffled | 1 | nuPI (rho=0) | True | False | True | 9.419e-04 | 8.320e-05 | False | True |
| pairwise | 30 | 5.000e-02 | shuffled | 1 | ALM (rho=1) | False | False | True | 9.419e-04 | 8.320e-05 | False | True |
| pairwise | 30 | 5.000e-02 | shuffled | 1 | iALM | False | False | True | 1.931e-02 | 1.664e-03 | False | True |
| pairwise | 30 | 5.000e-02 | shuffled | 1 | PBM | False | False | True | 0.000e+00 | 4.011e-12 | True | True |
| pairwise | 30 | 5.000e-02 | shuffled | 10 | ALM (rho=0) | True | True | True | 1.088e-03 | 2.525e-04 | False | True |
| pairwise | 30 | 5.000e-02 | shuffled | 10 | nuPI (rho=0) | True | True | True | 1.266e-03 | 2.941e-04 | False | True |
| pairwise | 30 | 5.000e-02 | shuffled | 10 | ALM (rho=1) | False | True | True | 1.071e-03 | 5.394e-04 | False | True |
| pairwise | 30 | 5.000e-02 | shuffled | 10 | iALM | False | True | True | 6.495e-02 | 4.388e-03 | True | True |
| pairwise | 30 | 5.000e-02 | shuffled | 10 | PBM | False | True | True | 0.000e+00 | 1.198e-11 | True | True |
| pairwise | 30 | -2.000e-02 | balanced | 1 | ALM (rho=0) | True | True | True | 2.208e-16 | 1.141e-16 | False | True |
| pairwise | 30 | -2.000e-02 | balanced | 1 | nuPI (rho=0) | True | True | True | 2.208e-16 | 1.141e-16 | False | True |
| pairwise | 30 | -2.000e-02 | balanced | 1 | ALM (rho=1) | False | True | True | 2.208e-16 | 1.525e-03 | False | True |
| pairwise | 30 | -2.000e-02 | balanced | 1 | iALM | False | True | True | 3.985e-16 | 1.525e-03 | False | True |
| pairwise | 30 | -2.000e-02 | balanced | 1 | PBM | False | True | True | 2.010e-16 | 1.665e-12 | True | True |
| pairwise | 30 | -2.000e-02 | balanced | 10 | ALM (rho=0) | True | True | True | 2.161e-16 | 1.105e-16 | False | True |
| pairwise | 30 | -2.000e-02 | balanced | 10 | nuPI (rho=0) | True | True | True | 6.463e-16 | 3.317e-16 | False | True |
| pairwise | 30 | -2.000e-02 | balanced | 10 | ALM (rho=1) | False | True | True | 1.029e-03 | 1.131e-02 | False | True |
| pairwise | 30 | -2.000e-02 | balanced | 10 | iALM | False | True | True | 1.050e-02 | 1.198e-02 | False | True |
| pairwise | 30 | -2.000e-02 | balanced | 10 | PBM | False | True | True | 9.726e-12 | 1.572e-11 | False | True |
| pairwise | 30 | -2.000e-02 | shuffled | 1 | ALM (rho=0) | True | True | True | 9.353e-04 | 8.320e-05 | False | True |
| pairwise | 30 | -2.000e-02 | shuffled | 1 | nuPI (rho=0) | True | True | True | 9.353e-04 | 8.320e-05 | False | True |
| pairwise | 30 | -2.000e-02 | shuffled | 1 | ALM (rho=1) | False | True | True | 9.353e-04 | 3.390e-03 | False | True |
| pairwise | 30 | -2.000e-02 | shuffled | 1 | iALM | False | True | True | 1.688e-02 | 3.913e-03 | False | True |
| pairwise | 30 | -2.000e-02 | shuffled | 1 | PBM | False | True | True | 4.573e-03 | 4.011e-12 | True | True |
| pairwise | 30 | -2.000e-02 | shuffled | 10 | ALM (rho=0) | True | True | True | 1.015e-03 | 2.525e-04 | False | True |
| pairwise | 30 | -2.000e-02 | shuffled | 10 | nuPI (rho=0) | True | True | True | 1.180e-03 | 2.941e-04 | False | True |
| pairwise | 30 | -2.000e-02 | shuffled | 10 | ALM (rho=1) | False | True | True | 7.657e-04 | 8.302e-03 | False | True |
| pairwise | 30 | -2.000e-02 | shuffled | 10 | iALM | False | True | True | 7.666e-03 | 9.986e-03 | False | True |
| pairwise | 30 | -2.000e-02 | shuffled | 10 | PBM | False | True | True | 5.258e-03 | 1.145e-11 | False | True |
| agg | 1 | 5.000e-02 | balanced | 1 | ALM (rho=0) | True | True | True | 3.176e-03 | 2.049e-02 | False | True |
| agg | 1 | 5.000e-02 | balanced | 1 | nuPI (rho=0) | True | True | True | 3.176e-03 | 2.049e-02 | False | True |
| agg | 1 | 5.000e-02 | balanced | 1 | ALM (rho=1) | False | True | True | 3.176e-03 | 2.281e-02 | False | True |
| agg | 1 | 5.000e-02 | balanced | 1 | iALM | False | True | True | 5.911e-02 | 2.509e-02 | False | True |
| agg | 1 | 5.000e-02 | balanced | 1 | PBM | False | True | True | 1.564e-02 | 4.305e-11 | False | True |
| agg | 1 | 5.000e-02 | balanced | 10 | ALM (rho=0) | True | True | True | 3.745e-02 | 5.407e-02 | False | True |
| agg | 1 | 5.000e-02 | balanced | 10 | nuPI (rho=0) | True | True | True | 3.555e-02 | 5.416e-02 | False | True |
| agg | 1 | 5.000e-02 | balanced | 10 | ALM (rho=1) | False | True | True | 3.689e-02 | 6.339e-02 | False | True |
| agg | 1 | 5.000e-02 | balanced | 10 | iALM | False | True | True | 3.931e-01 | 1.251e-01 | False | True |
| agg | 1 | 5.000e-02 | balanced | 10 | PBM | False | True | True | 4.532e-01 | 2.191e-10 | False | True |
| agg | 1 | 5.000e-02 | shuffled | 1 | ALM (rho=0) | True | True | True | 3.483e-03 | 3.173e-02 | False | True |
| agg | 1 | 5.000e-02 | shuffled | 1 | nuPI (rho=0) | True | True | True | 3.483e-03 | 3.173e-02 | False | True |
| agg | 1 | 5.000e-02 | shuffled | 1 | ALM (rho=1) | False | True | True | 3.483e-03 | 3.542e-02 | False | True |
| agg | 1 | 5.000e-02 | shuffled | 1 | iALM | False | True | True | 6.482e-02 | 3.893e-02 | False | True |
| agg | 1 | 5.000e-02 | shuffled | 1 | PBM | False | True | True | 1.715e-02 | 6.677e-11 | False | True |
| agg | 1 | 5.000e-02 | shuffled | 10 | ALM (rho=0) | True | True | True | 3.498e-02 | 4.948e-02 | False | True |
| agg | 1 | 5.000e-02 | shuffled | 10 | nuPI (rho=0) | True | True | True | 3.327e-02 | 4.969e-02 | False | True |
| agg | 1 | 5.000e-02 | shuffled | 10 | ALM (rho=1) | False | True | True | 3.514e-02 | 5.777e-02 | False | True |
| agg | 1 | 5.000e-02 | shuffled | 10 | iALM | False | True | True | 3.767e-01 | 1.161e-01 | False | True |
| agg | 1 | 5.000e-02 | shuffled | 10 | PBM | False | True | True | 3.546e-01 | 1.077e-10 | False | True |
| agg | 1 | -2.000e-02 | balanced | 1 | ALM (rho=0) | True | True | True | 3.154e-03 | 2.063e-02 | False | True |
| agg | 1 | -2.000e-02 | balanced | 1 | nuPI (rho=0) | True | True | True | 3.154e-03 | 2.063e-02 | False | True |
| agg | 1 | -2.000e-02 | balanced | 1 | ALM (rho=1) | False | True | True | 3.154e-03 | 2.581e-02 | False | True |
| agg | 1 | -2.000e-02 | balanced | 1 | iALM | False | True | True | 5.232e-02 | 3.080e-02 | False | True |
| agg | 1 | -2.000e-02 | balanced | 1 | PBM | False | True | True | 1.512e-02 | 4.590e-11 | False | True |
| agg | 1 | -2.000e-02 | balanced | 10 | ALM (rho=0) | True | True | True | 3.491e-02 | 5.733e-02 | False | True |
| agg | 1 | -2.000e-02 | balanced | 10 | nuPI (rho=0) | True | True | True | 3.292e-02 | 5.743e-02 | False | True |
| agg | 1 | -2.000e-02 | balanced | 10 | ALM (rho=1) | False | True | True | 3.404e-02 | 7.330e-02 | False | True |
| agg | 1 | -2.000e-02 | balanced | 10 | iALM | False | True | True | 2.049e-01 | 1.591e-01 | False | True |
| agg | 1 | -2.000e-02 | balanced | 10 | PBM | False | True | True | 7.539e-01 | 4.955e-10 | False | True |
| agg | 1 | -2.000e-02 | shuffled | 1 | ALM (rho=0) | True | True | True | 3.459e-03 | 3.195e-02 | False | True |
| agg | 1 | -2.000e-02 | shuffled | 1 | nuPI (rho=0) | True | True | True | 3.459e-03 | 3.195e-02 | False | True |
| agg | 1 | -2.000e-02 | shuffled | 1 | ALM (rho=1) | False | True | True | 3.459e-03 | 4.006e-02 | False | True |
| agg | 1 | -2.000e-02 | shuffled | 1 | iALM | False | True | True | 5.738e-02 | 4.777e-02 | False | True |
| agg | 1 | -2.000e-02 | shuffled | 1 | PBM | False | True | True | 1.658e-02 | 7.119e-11 | False | True |
| agg | 1 | -2.000e-02 | shuffled | 10 | ALM (rho=0) | True | True | True | 3.278e-02 | 5.183e-02 | False | True |
| agg | 1 | -2.000e-02 | shuffled | 10 | nuPI (rho=0) | True | True | True | 3.116e-02 | 5.204e-02 | False | True |
| agg | 1 | -2.000e-02 | shuffled | 10 | ALM (rho=1) | False | True | True | 3.317e-02 | 6.717e-02 | False | True |
| agg | 1 | -2.000e-02 | shuffled | 10 | iALM | False | True | True | 2.042e-01 | 1.840e-01 | False | True |
| agg | 1 | -2.000e-02 | shuffled | 10 | PBM | False | True | True | 5.932e-01 | 1.717e-10 | False | True |
