.. _Installation:

Installation and Quick Start
============================

This is a terminal application.

Installation
------------

#. Verify that a Python version 3.8 or higher is installed.

#. Verify that `git <https://github.com/>`_ is installed 

#. Install a solver such as ``cplex``, ``glpk``, ``gurobi``, ``ipopt`` or etc. so that is can be run from the command line.

#. Install `Pyomo <http://www.pyomo.org/>`_.

#. Install `mpi-sppy <https://github.com/Pyomo/mpi-sppy>`_ using a github clone.

#. Install `boot-sp <https://github.com/boot-sp/boot-sp>`_ using a github clone.

#. cd to the directory where you want to put `smais` and give these commands:


.. code-block:: bash

   $ git clone https://github.com/DLWoodruff/SMAIS
   $ cd SMAIS
   $ pip install -e .

   
For parallel operation, you will need to install `multiprocessing_on_dill <https://pypi.org/project/multiprocessing_on_dill/>`


Quick Start
-----------

To test your installation, assuming that you have Gurobi installed, 

.. code-block:: bash

   $ cd examples
   $ bash newsvendor.bash

If you don't have Gurobi Installed, edit `examples/newsvendor_is.json` and change the solver name.


