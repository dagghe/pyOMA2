The ``result`` module
---------------------

These classes are were the results of the analyses are stored.

Classes:
   :class:`.BaseResult`
      Base class for storing results data.
   :class:`.FDDResult`
      Class for storing Frequency Domain Decomposition (FDD) results data.
   :class:`.EFDDResult`
      Class for storing results data from Enhanced Frequency Domain Decomposition (EFDD)
      and Frequency Spatial Domain Decomposition (FSDD).
   :class:`.SSIResult`
      Class for storing results data from Stochastic Subspace Identification (SSI) methods.
   :class:`.pLSCFResult`
      Class for storing results data from the poly-reference Least Square Complex Frequency (pLSCF) method.
   :class:`.MsPoserResult`
      Base class for MultiSetup Poser result data.
   :class:`.AutoSSIResult`
      Result class for automated Structural System Identification (SSI) with clustering.
   :class:`.ClusteringResult`
      Class to store clustering results and related metadata.

.. Warning::
    The module is designed to be used as part of the pyOMA2 package and relies on its
    internal data structures and algorithms.

.. automodule:: pyoma2.algorithms.data.result
   :members:
   :show-inheritance:
