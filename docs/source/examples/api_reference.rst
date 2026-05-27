API Reference
=============

This page provides a comprehensive reference of the humancompatible-train API.

Core Idea
------------

Optimizer
~~~~~~~

Our Optimizer class 

.. code-block:: python

   class Trainer:
       """Initialize and train models with constraints."""
       
       def __init__(self, constraints=None, objective_weights=None):
           """
           Parameters
           ----------
           constraints : list, optional
               List of constraints to enforce
           objective_weights : dict, optional
               Weights for multi-objective optimization
           """
           pass
       
       def fit(self, X, y):
           """
           Fit the model.
           
           Parameters
           ----------
           X : array-like
               Training data
           y : array-like
               Training labels
           
           Returns
           -------
           model : TrainedModel
               The fitted model
           """
           pass
