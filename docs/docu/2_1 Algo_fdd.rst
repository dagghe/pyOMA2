The ``fdd`` module
==================

This module provides implementation of the Frequency Domain Decomposition (FDD) [BZA01]_, the Enhanced
Frequency Domain Decomposition (EFDD) [BVA01]_, and the Frequency Spatial Domain Decomposition (FSDD)
[ZWT10]_ algorithms, along with their adaptations for multi-setup experimental data [SARB21]_. These algorithms
are used in structural dynamics to identify modal parameters such as natural frequencies, damping ratios,
and mode shapes from ambient vibration measurements.

Classes:
   :class:`.FDD_algo`
      Implements the basic FDD algorithm for single setup modal analysis.
   :class:`.EFDD_algo`
      Extends ``FDD_algo`` to provide Enhanced FDD analysis.
   :class:`.FSDD_algo`
      Implements the Frequency-Spatial Domain Decomposition, a variant of EFDD.
   :class:`.FDD_algo_MS`
      Adapts ``FDD_algo`` for multi-setup modal analysis.
   :class:`.EFDD_algo_MS`
      Extends ``EFDD_algo`` for multi-setup scenarios.

.. Important::
   Each class contains methods for executing the respective algorithms, extracting modal parameters,
   and plotting results. The module also includes utility functions and classes for visualization
   and interactive analysis.

.. Note::
   Users should be familiar with the concepts of modal analysis and system identification to effectively use this module.


The ``FDD_algo`` class
----------------------

.. autoclass:: pyoma2.algorithm.fdd.FDD_algo
   :members:
   :inherited-members:
   :show-inheritance:

The ``EFDD_algo`` class
-----------------------

.. autoclass:: pyoma2.algorithm.fdd.EFDD_algo
   :members:
   :inherited-members:
   :show-inheritance:

The ``FSDD_algo`` class
-----------------------

.. autoclass:: pyoma2.algorithm.fdd.FSDD_algo
   :members:
   :inherited-members:
   :show-inheritance:

The ``FDD_algo_MS`` class
-------------------------

.. autoclass:: pyoma2.algorithm.fdd.FDD_algo_MS
   :members:
   :inherited-members:
   :show-inheritance:

The ``EFDD_algo_MS`` class
--------------------------

.. autoclass:: pyoma2.algorithm.fdd.EFDD_algo_MS
   :members:
   :inherited-members:
   :show-inheritance:
