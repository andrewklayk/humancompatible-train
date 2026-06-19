Installation
============

Prerequisites
-------------

<<<<<<< HEAD
- Python 3.8 or higher
=======
- Python 3.11 or higher
>>>>>>> b54b13db327489065b3ec3c95872c84428cb2883
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

<<<<<<< HEAD
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

=======
>>>>>>> b54b13db327489065b3ec3c95872c84428cb2883
Optional Dependencies
---------------------

For specific features, you may need additional packages:

.. code-block:: bash

<<<<<<< HEAD
   pip install humancompatible-train[dev]  # Development tools
   pip install humancompatible-train[docs]  # Documentation building
=======
   pip install humancompatible-train[examples]  # Example notebooks
>>>>>>> b54b13db327489065b3ec3c95872c84428cb2883
