humancompatible-train documentation
===================================

Welcome to the **humancompatible-train** documentation.

What is **humancompatible-train**?

**humancompatible-train** is a PyTorch-based package for constrained optimization, aimed at constrained deep learning tasks.
We implement several first-order Lagrangian-based methods for constrained optimization with a PyTorch-based API that allow seamless integration of constraints into the training loop.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :titlesonly:

   install
   getting_started

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :titlesonly:

   Constrained Optimization Overview <tutorials/copt_overview>
   Basic usage: Fairness <tutorials/basic_usage>
   Handling inequality constraints <tutorials/inequality_constraints>
   Tips and Tricks <tutorials/tips>

.. toctree::
   :caption: API reference
   :titlesonly:

   Dual Optimizers <api_reference/dual_optimizers>
   Utils <api_reference/utils>