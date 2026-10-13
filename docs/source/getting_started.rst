Getting Started
===============

Quick Start
-----------

After installing humancompatible-train, you can import it in your Python code:

.. code-block:: python

   from humancompatible.train.dual_optim import *

Basic Example
--------------

This is an abstract code sample; you can find runnable examples in the :doc:`tutorials/basic_usage` section.

.. code-block:: python

    from humancompatible.train.dual_optim import ALM

    device = ...
    num_constraints = ...

    optimizer = torch.optim.Adam(model.parameters(), ...)
    dual_optimizer = ALM(m=num_constraints, ..., device=device)

    for inputs, labels in dataloader:
        # evaluate objective
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # evaluate tensor of constraints
        constraints = evaluate_constraints(inputs, labels, ...) 
        # evaluate lagrangian and update dual variables
        lagrangian = dual_optimizer.forward_update(loss, constraints)
        # backward pass and step
        lagrangian.backward()
        optimizer.step()
        optimizer.zero_grad()

.. note::

   For detailed examples (including inequality constraints), see the :doc:`tutorials/basic_usage` and :doc:`tutorials/inequality_constraints` sections.

Next Steps
----------

- Read the :doc:`Basic Usage <tutorials/basic_usage>` guide for a complete example
- If you encounter issues, visit the :doc:`Troubleshooting <troubleshooting>` page
