The ``fdd`` module
------------------

This module is a part of the pyOMA2 package and provides utility functions for conducting
Operational Modal Analysis (OMA) using Frequency Domain Decomposition (FDD) algorithm [BZA01]_, Enhanced
Frequency Domain Decomposition (EFDD) algorithm [BVA01]_ and Frequency Spatial Domain Decomposition (FSDD)
algorithm [ZWT10]_.

Functions:
    - :func:`.SD_PreGER`: Estimates Power Spectral Density matrices for multi-setup experiments.
    - :func:`.SD_est`: Computes Cross-Spectral Density using correlogram or periodogram methods.
    - :func:`.SD_svalsvec`: Calculates singular values and vectors for Cross-Spectral Density matrices.
    - :func:`.FDD_mpe`: Extracts modal parameters using the FDD method.
    - :func:`.SDOF_bellandMS`: Utility function for EFDD and FSDD methods.
    - :func:`.EFDD_mpe`: Extracts modal parameters using EFDD and FSDD methods.

.. automodule:: pyoma2.functions.fdd
   :members:
