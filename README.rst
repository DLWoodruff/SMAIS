SMAIS
======
Adaptive Importance Sampling with Surrogate Modeling


Overview
========

Stochastic programming models can serve as indispensable tools for decision-making in uncertain environments.  A key aspect of optimizing such models is the effective evaluation of the recourse function for a given candidate solution, denoted as xhat. This software enhances this process by adeptly constructing importance sampling distributions through surrogate modeling. The primary aim is to minimize variance, thereby ensuring more efficient and accurate function value evaluations.

At first, the software is expected mainly to be used by researchers
and is only tested on Unix platforms. It might work on Windows, but is
not tested there. This documentation assumes you are using a Unix
shell such as Bash.


Basics
------

The ``smais`` software relies on a ``Pyomo`` model to define the underlying problem (i.e., a deterministic scenario) and relies
on ``mpi-sppy`` and ``boot-sp``for some low level functions. All of these packages must be installed.


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


