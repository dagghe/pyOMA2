The ``fdd`` algorithm module
============================

This module provides implementation of the Frequency Domain Decomposition (FDD) algorithm [BZA01]_, the Enhanced
Frequency Domain Decomposition (EFDD) algorithm [BVA01]_, and the Frequency Spatial Domain Decomposition (FSDD)
algorithm [ZWT10]_, along with their adaptations for multi-setup experimental data [SARB21]_. These algorithms
are used in structural dynamics to identify modal parameters such as natural frequencies, damping ratios,
and mode shapes from ambient vibration measurements.

Classes:
   :class:`.FDD`
      Implements the basic FDD algorithm for single setup modal analysis.
   :class:`.EFDD`
      Extends ``FDD`` to provide Enhanced FDD analysis.
   :class:`.FSDD`
      Implements the Frequency-Spatial Domain Decomposition, a variant of EFDD.
   :class:`.FDD_MS`
      Adapts ``FDD`` for multi-setup modal analysis.
   :class:`.EFDD_MS`
      Extends ``EFDD`` for multi-setup scenarios.

.. Important::
   Each class contains methods for executing the respective Frequency Domain Decomposition based
   algorithm, extracting modal parameters, plotting results, and additional utilities relevant to
   the specific FDD approach.

.. Note::
   Users should be familiar with the concepts of modal analysis and system identification to effectively use this module.


The ``FDD`` class
----------------------

.. autoclass:: pyoma2.algorithms.fdd.FDD
   :members:
   :inherited-members:
   :show-inheritance:

The ``EFDD`` class
-----------------------

.. autoclass:: pyoma2.algorithms.fdd.EFDD
   :members:
   :inherited-members:
   :show-inheritance:

The ``FSDD`` class
-----------------------

.. autoclass:: pyoma2.algorithms.fdd.FSDD
   :members:
   :inherited-members:
   :show-inheritance:

The ``FDD_MS`` class
-------------------------

.. autoclass:: pyoma2.algorithms.fdd.FDD_MS
   :members:
   :inherited-members:
   :show-inheritance:

The ``EFDD_MS`` class
--------------------------

.. autoclass:: pyoma2.algorithms.fdd.EFDD_MS
   :members:
   :inherited-members:
   :show-inheritance:
