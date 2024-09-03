The ``plscf`` module
--------------------

This module is a part of the pyOMA2 package and provides utility functions for conducting
Operational Modal Analysis (OMA) using the poly-reference Least Square Complex Frequency Domain (pLSCFD)
identification method, also known as polymax [PAGL04]_.

Functions:
    - :func:`.pLSCF`: Perform the poly-reference Least Square Complex Frequency (pLSCF) algorithm.
    - :func:`.pLSCF_Poles`: Extract poles from the pLSCF algorithm results.
    - :func:`.rmfd2AC`: Convert Right Matrix Fraction Description to state-space representation.
    - :func:`.AC2MP_poly`: Convert state-space representation to modal parameters.
    - :func:`.pLSCF_MPE`: Extract modal parameters using the pLSCF method for selected frequencies.

.. automodule:: pyoma2.functions.plscf
   :members:
