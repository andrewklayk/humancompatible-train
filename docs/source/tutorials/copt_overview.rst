Constrained Optimization Overview
=================================

This tutorial provides an overview of constrained optimization problems, and how this relates to Deep Learning. We will cover problem formulation, ....


Formulation
---------------

In `humancompatible-train`, and in Constrained Machine Learning more generally, we are interested in solving problems of the form:

.. math:: 
    \min_{x\in\mathbb{R}^n} \quad & \mathbb{E}[f(x,\xi)] \\
    \text{s.t.} \quad & \mathbb{E}[g(x,\xi)] \leq 0, \\
    & \mathbb{E}[h(x,\xi)] = 0, \\

where :math:`f` is the **objective function** we want to minimize, :math:`g` are the **inequality constraints**, and :math:`h` are the **equality constraints**. The expectation is taken over some random variable :math:`\xi`, which represents the data.

You may recognize the first line of the above formula as the standard formulation of an optimization problem in machine learning, where we want to minimize the expected loss over the data. \
We then introduce constraints -- they could express anything from some bounds on the parameters of the model, or a requirement on the model's predictions to satisfy some fairness criterion, to the boundary conditions of a physical system.


.. note::
    - As is standard in the field, we adopt the convention of writing the constraints as :math:`g(x) \leq 0`, and :math:`h(x) = 0`. This is just a notational choice, and does not affect the generality of the formulation. It is trivial to transform :math:`g(x) \geq 0` into :math:`-g(x) \leq 0`, or :math:`g(x) \leq \epsilon` into :math:`g(x) - \epsilon \leq 0` for some non-zero bound. 
    - It is also easy to switch between equality and inequality constraints: to achieve :math:`g(x) = 0`, one can set :math:`-g(x) \leq 0` and :math:`g(x) \leq 0` simultaneously. In fact, different algorithms are designed to handle either equality or inequality constraints natively, but, again, it is trivial to switch between the two. We shall see more concrete examples later on.


Solving Constrained Problems
--------------------------------

We all know how to solve an unconstrained optimization problem -- we can use gradient descent, or any of its variants. But how do we solve a constrained optimization problem?
The Constrained Machine Learning field, including us, seems to have converged on **Lagrangian-based methods**, which utilize the Lagrangian function to transform the **constrained** problem into an **unconstrained** one.

Going forward in this tutorial, we will focus on the **deterministic case** to simplify notation; the stochastic case is more complex, but utilizes the same principles (imagine Gradient Descent vs. SGD). For more rigorous mathematical treatment of the stochastic case, see **TODO**, as well as the references included in the documentation for each of the algorithms in the package.

In a deterministic case, the Lagrangian function is defined as follows:

.. math::
     \mathcal{L}(x, \lambda, \mu) = f(x) + \lambda^T g(x) + \mu^T h(x)

where :math:`\lambda` is the Lagrange multiplier associated with the constraint :math:`g(x) \leq 0`, and :math:`\mu` is the Lagrange multiplier associated with the constraint :math:`h(x) = 0`.

It is then possible to show that the original constrained optimization problem is equivalent to the following unconstrained optimization problem:

.. math::
    \min_{x\in\mathbb{R}^n} \max_{\lambda \geq 0, \mu} \mathcal{L}(x, \lambda, \mu)


We refer to the original problem as the **primal problem**, with :math:`x` as the **primal variables**, and to the transformed problem as the **dual problem**, with :math:`\lambda` and :math:`\mu` as the **dual variables**. The dual problem is unconstrained, and can be solved using a clever application of standard optimization techniques.

In particular, the most common approach is to use **alternating updates**: we fix the primal variables, and optimize the dual variables using gradient ascent; then we fix the dual variables, and optimize the primal variables using gradient descent. This process is repeated until convergence.

In `humancompatible-train`, we implement several variants of this approach, based on methods present in the literature. For more details, see the corresponding documentation; for now, it is important to understand that they are all based on the same principle of alternating updates to the primal and dual variables.

In the simplest case of the Lagrangian method, this gives us the following update rules:

.. math::
    \lambda_{t+1} & = \lambda_t + \beta \nabla_\lambda \mathcal{L}(x_{t}, \lambda_t, \mu_t) = \lambda_t + \beta g(x_{t}) \\
    \mu_{t+1} & = \mu_t + \gamma \nabla_\mu \mathcal{L}(x_{t}, \lambda_t, \mu_t) = \mu_t + \gamma h(x_{t}) \\
    x_{t+1} & = x_t - \alpha \nabla_x \mathcal{L}(x_t, \lambda_t, \mu_t)

where :math:`\alpha`, :math:`\beta`, and :math:`\gamma` are the learning rates for the primal and dual variables, respectively.

.. note::
    - The above update rules are for the simplest variant of the Lagrangian method. The methods implemented in this package are all more complex. Even beyond our implementation, one can (and sometimes should!) modify the update rules by e.g. tweaking the training loop code, as we show in the :doc:`tips` tutorial.
    - The above update rules are for the deterministic case. In the stochastic case, the gradients are estimated using mini-batches of data, which introduces additional noise into the optimization process. This can make convergence more challenging, but we have some tricks up our sleeves, such as momentum, LR scheduling, and so on.

In our package, the `dual optimizers` handle the updates to the dual variables, while the primal updates are handled by the standard PyTorch optimizers. This allows for seamless integration of constraints into the training loop, as we will see in the next tutorial.
