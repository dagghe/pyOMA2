.. autosummary::
   :toctree: generated

The ``ssi`` module
------------------

This module implements the Stochastic Subspace Identification (SSI) algorithm in various forms,
tailored for both single and multiple experimental setup scenarios. It includes classes and methods
for conducting data-driven and covariance-driven SSI analyses. The primary focus of this module is
on structural analysis and system identification, providing tools to process measurement data, extract
modal parameters, and perform comprehensive system dynamics analysis.

Classes:
   ``SSIdat_algo``
      Implements the Data-Driven SSI algorithm for single setup.
   ``SSIcov_algo``
      Implements the Covariance-Driven SSI algorithm for single setup.
   ``SSIdat_algo_MS``
      Extends ``SSIdat_algo`` for multi-setup experiments.
   ``SSIcov_algo_MS``
      Extends ``SSIdat_algo_MS`` for covariance-based analysis in multi-setup experiments.

Each class contains methods for executing the SSI algorithm, extracting modal parameters,
plotting results, and additional utilities relevant to the specific SSI approach.

**Summary:**

.. autosummary::

    pyoma2.algorithm.ssi.SSIdat_algo
    pyoma2.algorithm.ssi.SSIcov_algo
    pyoma2.algorithm.ssi.SSIdat_algo_MS
    pyoma2.algorithm.ssi.SSIcov_algo_MS

.. Note::
   - The classes are designed to be used as part of the ``pyOMA2`` package and rely on its internal data structures and algorithms.
   - Users should be familiar with the concepts of modal analysis and system identification to effectively use this module.


.. [1] Peeters, B., & De Roeck, G. (1999). Reference-based stochastic subspace
       identification for output-only modal analysis. Mechanical Systems and
       Signal Processing, 13(6), 855-878.
.. [2] Döhler, M. (2011). Subspace-based system identification and fault detection:
       Algorithms for large systems and application to structural vibration analysis.
       Diss. Université Rennes 1.


The ``SSIdat_algo`` class
#########################

.. autoclass:: pyoma2.algorithm.ssi.SSIdat_algo
   :members:
   :show-inheritance:

The ``SSIcov_algo`` class
#########################

.. autoclass:: pyoma2.algorithm.ssi.SSIcov_algo
   :members:
   :show-inheritance:

The ``SSIdat_algo_MS`` class
############################

.. autoclass:: pyoma2.algorithm.ssi.SSIdat_algo_MS
   :members:
   :show-inheritance:


The ``SSIcov_algo_MS`` class
############################

.. autoclass:: pyoma2.algorithm.ssi.SSIcov_algo_MS
   :members:
   :show-inheritance:
