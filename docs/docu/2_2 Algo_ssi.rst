The ``ssi`` algorithm module
============================

This module provides a comprehensive implementation of the Stochastic Subspace Identification (SSI) algorithm family
for modal analysis and system identification [BPDG99]_, [MiDo11]_, [MDM16]_. It supports both single‐setup and
multi‐setup experimental configurations, as well as input-output scenarios using OMAX (Operational Modal Analysis with
eXogenous inputs).
Both data‐driven and covariance‐driven algorithms are included, with optional uncertainty quantification for the
modal parameters [DLM13]_.

Classes:
   :class:`.SSI`
      Implements the SSI algorithm for single setup experiments.
   :class:`.SSI_MS`
      Extends ``SSI`` for multi-setup experiments.


.. Important::
   Each class contains methods for executing the SSI algorithm, extracting modal parameters,
   plotting results, and additional utilities relevant to the specific SSI approach.

.. Note::
   Users should be familiar with the concepts of modal analysis and system identification to effectively use this module.


The ``SSI`` class
-------------------------

.. autoclass:: pyoma2.algorithms.ssi.SSI
   :members:
   :inherited-members:
   :show-inheritance:
   :no-index:

The ``SSI_MS`` class
----------------------------

.. autoclass:: pyoma2.algorithms.ssi.SSI_MS
   :members:
   :inherited-members:
   :show-inheritance:
   :no-index:
