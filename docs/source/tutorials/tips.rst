Tips and Tricks
==================================================

Here, we discuss some miscellaneous tricks and tips for using the package, which are not specific to any particular method, but can be useful in general when working with constrained optimization problems.

Dealing with Noise
------------------

In the stochastic case, the gradients are estimated using mini-batches of data, which introduces additional noise into the optimization process. This can make convergence more challenging, but this can be mitigated.

**Momentum**: Just like in standard optimization, using momentum can help smooth out the updates and mitigate the noise. In `humancompatible-train`, the dual optimizers support momentum, which can be enabled by setting the `momentum` parameter to a non-zero value.
