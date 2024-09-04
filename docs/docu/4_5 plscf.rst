The ``plscf`` module
--------------------

This module is a part of the pyOMA2 package and provides utility functions for conducting
Operational Modal Analysis (OMA) using the poly-reference Least Square Complex Frequency Domain (pLSCFD)
identification method, also known as polymax [PAGL04]_.

Functions:
    - :func:`.pLSCF`: Perform the poly-reference Least Square Complex Frequency (pLSCF) algorithm.
    - :func:`.pLSCF_poles`: Extract poles from the pLSCF algorithm results.
    - :func:`.rmfd2ac`: Convert Right Matrix Fraction Description to state-space representation.
    - :func:`.ac2mp_poly`: Convert state-space representation to modal parameters.
    - :func:`.pLSCF_mpe`: Extract modal parameters using the pLSCF method for selected frequencies.

.. automodule:: pyoma2.functions.plscf
   :members:
