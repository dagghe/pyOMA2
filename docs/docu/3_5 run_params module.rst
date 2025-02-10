The ``run_params`` module
-------------------------

These classes are were the parameters used to run the analyses are stored.

Classes:
   :class:`.BaseRunParams`
      Base class for storing run parameters for modal analysis algorithms.
   :class:`.FDDRunParams`
      Class for storing Frequency Domain Decomposition (FDD) run parameters.
   :class:`.EFDDRunParams`
      Class for storing Enhanced Frequency Domain Decomposition (EFDD) run parameters.
   :class:`.SSIRunParams`
      Parameters for the Stochastic Subspace Identification (SSI) method.
   :class:`.pLSCFRunParams`
      Parameters for the poly-reference Least Square Complex Frequency (pLSCF) method.
   :class:`.AutoSSIRunParams`
      Run parameters for automated SSI.
   :class:`.Step1`
      Parameters for the first step of clustering analysis.
   :class:`.Step2`
      Parameters for the second step of clustering analysis.
   :class:`.Step3`
      Parameters for the third step of clustering analysis.
   :class:`.Clustering`
      Main class for clustering analysis.

.. Warning::
    The module is designed to be used as part of the pyOMA2 package and relies on its
    internal data structures and algorithms.

.. automodule:: pyoma2.algorithms.data.run_params
   :members:
   :show-inheritance:
