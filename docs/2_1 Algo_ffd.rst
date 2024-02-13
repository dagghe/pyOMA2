.. autosummary::
   :toctree: generated

The ``fdd`` module
------------------

This module provides implementation of the Frequency Domain Decomposition (FDD) and Enhanced
Frequency Domain Decomposition (EFDD) algorithms, along with their adaptations for multi-setup
experimental data. These algorithms are used in structural dynamics to identify modal parameters
such as natural frequencies, damping ratios, and mode shapes from ambient vibration measurements.

Classes:
   ``FDD_algo``
      Implements the basic FDD algorithm for single setup modal analysis.
   ``EFDD_algo``
      Extends ``FDD_algo`` to provide Enhanced FDD analysis.
   ``FSDD_algo``
      Implements the Frequency-Spatial Domain Decomposition, a variant of EFDD.
   ``FDD_algo_MS``
      Adapts ``FDD_algo`` for multi-setup modal analysis.
   ``EFDD_algo_MS``
      Extends ``EFDD_algo`` for multi-setup scenarios.

Each class contains methods for executing the respective algorithms, extracting modal parameters,
and plotting results. The module also includes utility functions and classes for visualization
and interactive analysis.

**Summary:**

.. autosummary::

    pyoma2.algorithm.fdd.FDD_algo
    pyoma2.algorithm.fdd.EFDD_algo
    pyoma2.algorithm.fdd.FSDD_algo
    pyoma2.algorithm.fdd.FDD_algo_MS
    pyoma2.algorithm.fdd.EFDD_algo_MS

.. Note::
   - The classes are designed to be used as part of the ``pyOMA2`` package and rely on its internal data structures and algorithms.
   - Users should be familiar with the concepts of modal analysis and system identification to effectively use this module.


.. [1] Brincker, R., Zhang, L., & Andersen, P. (2001). Modal identification of output-only
       systems using frequency domain decomposition. Smart Materials and Structures, 10(3), 441.

.. [2] Brincker, R., Ventura, C. E., & Andersen, P. (2001). Damping estimation by frequency
       domain decomposition. In Proceedings of IMAC 19: A Conference on Structural Dynamics,
       February 5-8, 2001, Hyatt Orlando, Kissimmee, Florida. Society for Experimental Mechanics.

.. [3] Zhang, L., Wang, T., & Tamura, Y. (2010). A frequency–spatial domain decomposition
       (FSDD) method for operational modal analysis. Mechanical Systems and Signal Processing,
       24(5), 1227-1239.


The ``FDD_algo`` class
######################

.. autoclass:: pyoma2.algorithm.fdd.FDD_algo
   :members:
   :show-inheritance:

The ``EFDD_algo`` class
######################

.. autoclass:: pyoma2.algorithm.fdd.EFDD_algo
   :members:
   :show-inheritance:

The ``FSDD_algo`` class
######################

.. autoclass:: pyoma2.algorithm.fdd.FSDD_algo
   :members:
   :show-inheritance:

The ``FDD_algo_MS`` class
#########################

.. autoclass:: pyoma2.algorithm.fdd.FDD_algo_MS
   :members:
   :show-inheritance:

The ``EFDD_algo_MS`` class
##########################

.. autoclass:: pyoma2.algorithm.fdd.EFDD_algo_MS
   :members:
   :show-inheritance:
