humancompatible-train documentation
===================================

Welcome to the **humancompatible-train** documentation.
<<<<<<< HEAD
=======

What is **humancompatible-train**?

**humancompatible-train** is a PyTorch-based package for constrained optimization, aimed at constrained deep learning tasks.
We implement several first-order Lagrangian-based methods for constrained optimization with a PyTorch-based API that allow seamless integration of constraints into the training loop.
>>>>>>> b54b13db327489065b3ec3c95872c84428cb2883

What is **humancompatible-train**?

**humancompatible-train** is a PyTorch-based package for constrained optimization, aimed at constrained deep learning tasks.
We implement several first-order Lagrangian-based methods for constrained optimization with a PyTorch-based API that allow seamless integration of constraints into the training loop.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   getting_started

.. toctree::
   :maxdepth: 2
   :caption: Usage Examples

   examples/basic_usage
   examples/advanced_usage
   examples/learn_DAG
   examples/api_reference

.. toctree::
   :maxdepth: 2
<<<<<<< HEAD
   :caption: Additional Resources

   troubleshooting
   support
=======
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
>>>>>>> b54b13db327489065b3ec3c95872c84428cb2883
