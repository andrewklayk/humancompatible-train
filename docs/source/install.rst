Installation
============

Prerequisites
-------------

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

Basic Installation
------------------

Install the package using pip:

.. code-block:: bash

   pip install humancompatible-train

Installation from Source
------------------------

To install the development version from source:

.. code-block:: bash

   git clone https://github.com/humancompatible-train.git
   cd humancompatible-train
   pip install -e .

Using Virtual Environment (Recommended)
----------------------------------------

It's recommended to install in a virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install humancompatible-train

Verifying Installation
----------------------

To verify your installation was successful:

.. code-block:: python

   import humancompatible_train
   print(humancompatible_train.__version__)

Optional Dependencies
---------------------

For specific features, you may need additional packages:

.. code-block:: bash

   pip install humancompatible-train[dev]  # Development tools
   pip install humancompatible-train[docs]  # Documentation building