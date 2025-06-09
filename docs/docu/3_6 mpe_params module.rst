The ``mpe_params`` module
-------------------------

These classes are were the parameters used to extract the modal parameters are stored.

Classes:
   :class:`.BaseMPEParams`
      Base class for storing mpe parameters for modal analysis algorithms.
   :class:`.FDDMPEParams`
      Class for storing Frequency Domain Decomposition (FDD) MPE parameters.
   :class:`.EFDDMPEParams`
      Class for storing Enhanced Frequency Domain Decomposition (EFDD) MPE parameters.
   :class:`.SSIMPEParams`
      Class for storing Stochastic Subspace Identification (SSI) MPE parameters.
   :class:`.pLSCFMPEParams`
      Class for storing poly-reference Least Square Complex Frequency (pLSCF) MPE parameters.

.. Warning::
    The module is designed to be used as part of the pyOMA2 package and relies on its
    internal data structures and algorithms.

.. automodule:: pyoma2.algorithms.data.mpe_params
   :members:
   :show-inheritance:
