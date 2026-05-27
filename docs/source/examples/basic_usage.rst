Basic Usage
===========

This page demonstrates the basic usage patterns of humancompatible-train.

Simple Training Example
-----------------------

Below is a minimal example showing how to train a model with constraints:

.. code-block:: python

   from humancompatible.train.dual_optim import ALM
   from sklearn.datasets import make_classification
   
   # Load your data
   X, y = make_classification(n_samples=100, n_features=20)
   
   # Initialize a trainer
   trainer = hc.Trainer()
   
   # Train your model
   model = trainer.fit(X, y)
   
   # Make predictions
   predictions = model.predict(X)

Working with Constraints
------------------------

To add constraints to your training:

.. code-block:: python

   # Define constraints
   constraints = [
       hc.Constraint(type='fairness', metric='demographic_parity'),
       hc.Constraint(type='robustness', threshold=0.95)
   ]
   
   # Train with constraints
   trainer = hc.Trainer(constraints=constraints)
   model = trainer.fit(X, y)

Evaluating Your Model
---------------------

After training, you can evaluate your model's performance:

.. code-block:: python

   metrics = model.evaluate(X_test, y_test)
   print(f"Accuracy: {metrics['accuracy']}")
   print(f"Constraint satisfaction: {metrics['constraint_satisfaction']}")

Common Patterns
---------------

.. note::

   These are placeholder examples. Refer to the actual API for the correct function signatures and parameters.

See Also
--------

- :doc:`Advanced Usage <advanced_usage>` for more complex examples
- :doc:`API Reference <api_reference>` for detailed documentation
