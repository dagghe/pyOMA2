The ``ssi`` module
------------------

This module provides a collection of utility functions to support the implementation
of Stochastic Subspace Identification (SSI) algorithms [BPDG99]_, [MiDo11]_, [DOME13]_.
It includes functions for building Hankel matrices with various methods, converting
state-space representations to modal parameters,performing system identification using
SSI, and extracting modal parameters from identified systems.

Functions:
    - :func:`.build_hank`: Constructs a Hankel matrix from time series data.
    - :func:`.SSI`: Performs system identification using the SSI method.
    - :func:`.SSI_fast`: Efficient implementation of the SSI system identification.
    - :func:`.SSI_poles`: Computes modal parameters from identified state-space models.
    - :func:`.SSI_multi_setup`: SSI for multiple setup measurements.
    - :func:`.SSI_mpe`: Extracts modal parameters for selected frequencies.

.. automodule:: pyoma2.functions.ssi
   :members:
