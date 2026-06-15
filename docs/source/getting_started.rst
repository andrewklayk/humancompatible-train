Getting Started
===============

Quick Start
-----------

After installing humancompatible-train, you can import it in your Python code:

.. code-block:: python

   from humancompatible.train.dual_optim import *

Basic Example
--------------

<<<<<<< HEAD
This is an abstract code sample; you can find runnable examples in the :doc:`Basic Usage <examples/basic_usage>` section.

.. code-block:: python
    
=======
This is an abstract code sample; you can find runnable examples in the :doc:`tutorials/basic_usage` section.

.. code-block:: python

>>>>>>> b54b13db327489065b3ec3c95872c84428cb2883
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

<<<<<<< HEAD
   For detailed examples (including inequality constraints), see the :doc:`Basic Usage <examples/basic_usage>` section.
=======
   For detailed examples (including inequality constraints), see the :doc:`tutorials/basic_usage` and :doc:`tutorials/inequality_constraints` sections.
>>>>>>> b54b13db327489065b3ec3c95872c84428cb2883

Next Steps
----------

<<<<<<< HEAD
- Read the :doc:`Basic Usage <examples/basic_usage>` guide for a complete example
- Check the :doc:`API Reference <examples/api_reference>` for detailed function documentation
=======
- Read the :doc:`Basic Usage <tutorials/basic_usage>` guide for a complete example
>>>>>>> b54b13db327489065b3ec3c95872c84428cb2883
- If you encounter issues, visit the :doc:`Troubleshooting <troubleshooting>` page
