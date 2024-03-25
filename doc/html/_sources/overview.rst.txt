.. _Overview:

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

