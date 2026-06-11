Tips and Tricks
==================================================

Here, we discuss some miscellaneous tricks and tips for using the package, which are not specific to any particular method, but can be useful in general when working with constrained optimization problems.

Dealing with Noise
------------------

In the stochastic case, the gradients are estimated using mini-batches of data, which introduces additional noise into the optimization process. This can make convergence more challenging, but this can be mitigated.

**Momentum**: Just like in standard optimization, using momentum can help smooth out the updates and mitigate the noise. In ``humancompatible-train``, the ``ALM``, ``iALM``, and ``nuPI`` dual optimizers support momentum, which can be enabled by setting the ``momentum`` parameter to a non-zero value.
Some dual update strategies, such as ``nuPI``, explicitly rely on momentum.

Without momentum, the dual update at each step is a direct ascent step on the constraint values:

.. math::

    \pmb{\lambda}_{t+1} \leftarrow \text{clamp}\!\left(\pmb{\lambda}_t + \gamma\, \mathbf{c}_t(\theta_t),\; \lambda_{\min},\; \lambda_{\max}\right)

With momentum enabled, a running buffer :math:`\mathbf{b}_t` accumulates a weighted history of past constraint values before being used for the dual update:

.. math::

    \mathbf{b}_{t+1} &\leftarrow \mu\, \mathbf{b}_t + (1 - \delta)\, \mathbf{c}_t(\theta_t) \\
    \pmb{\lambda}_{t+1} &\leftarrow \text{clamp}\!\left(\pmb{\lambda}_t + \gamma\, \mathbf{b}_{t+1},\; \lambda_{\min},\; \lambda_{\max}\right)

where :math:`\mu` is the ``momentum`` coefficient, :math:`\delta` is the ``dampening`` coefficient, and :math:`\gamma` is the dual learning rate.

.. note::

    When ``momentum > 0`` and ``dampening`` is not explicitly provided, the library automatically sets ``dampening = momentum``.
    This conservative choice prioritises stability: the buffer update becomes

    .. math::

        \mathbf{b}_{t+1} \leftarrow \mu\, \mathbf{b}_t + (1 - \mu)\, \mathbf{c}_t(\theta_t)

    which is a standard exponential moving average of the constraint values with smoothing factor :math:`\mu`.