The ``plscf`` module
====================

Bla bla bla

Classes:
   :class:`.pLSCF`
      Implements the Data-Driven SSI algorithm for single setup.
   :class:`.pLSCF_MS`
      Implements the Covariance-Driven SSI algorithm for single setup.

.. Important::
   Each class contains methods for executing the pLSCF algorithm, extracting modal parameters,
   plotting results, and additional utilities relevant to the specific approach.

.. Note::
   Users should be familiar with the concepts of modal analysis and system identification to effectively use this module.


The ``pLSCF`` class
-------------------------

.. autoclass:: pyoma2.algorithm.plscf.pLSCF
   :members:
   :inherited-members:
   :show-inheritance:

The ``pLSCF_MS`` class
-------------------------

.. autoclass:: pyoma2.algorithm.plscf.pLSCF_MS
   :members:
   :inherited-members:
   :show-inheritance:
