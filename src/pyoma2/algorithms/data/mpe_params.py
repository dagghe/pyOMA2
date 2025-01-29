"""
This module provides classes for storing mpe parameters for various modal analysis
algorithms included in the pyOMA2 module.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


class BaseMPEParams(BaseModel):
    """
    Base class for storing mpe parameters for modal analysis algorithms.
    """

    model_config = ConfigDict(
        from_attributes=True, arbitrary_types_allowed=True, extra="forbid"
    )


class FDDMPEParams(BaseMPEParams):
    """
    Class for storing Frequency Domain Decomposition (FDD) MPE parameters.

    Attributes
    ----------
    sel_freq : numpy.ndarray
        Array of selected frequencies for modal parameter estimation,.
    DF : float, optional
        Frequency resolution for estimation, default is 0.1.
    """

    sel_freq: Optional[npt.NDArray[np.float64]] = None
    DF: float = 0.1


class EFDDMPEParams(BaseMPEParams):
    """
    Class for storing Enhanced Frequency Domain Decomposition (EFDD) MPE parameters.

    Attributes
    ----------
    sel_freq : numpy.ndarray
        Array of selected frequencies for modal parameter estimation,.
    DF1 : float, optional
        Frequency resolution for estimation, default is 0.1.
    DF2 : float
        Frequency resolution for the second stage of EFDD, default is 1.0.
    cm : int
        Number of closely spaced modes, default is 1.
    MAClim : float
        Minimum acceptable Modal Assurance Criterion value, default is 0.85.
    sppk : int
        Number of peaks to skip for the fit, default is 3.
    npmax : int
        Maximum number of peaks to use in the fit, default is 20.
    """

    sel_freq: Optional[npt.NDArray[np.float64]] = None
    DF1: float = 0.1
    DF2: float = 1.0
    cm: int = 1
    MAClim: float = 0.95
    sppk: int = 3
    npmax: int = 20


class SSIMPEParams(BaseMPEParams):
    """
    Class for storing Stochastic Subspace Identification (SSI) MPE parameters.

    Attributes
    ----------
    sel_freq : list of float or None, optional
        List of selected frequencies for modal parameter extraction. Default is None.
    order_in : int, list of int, or str
        Specified model order(s) for which the modal parameters are to be extracted.
        If 'find_min', the function attempts to find the minimum model order that provides
        stable poles for each mode of interest.
    rtol : float, optional
        Relative tolerance for comparing identified frequencies with the selected ones.
        Default is 5e-2.
    """

    sel_freq: Optional[List[float]] = None
    order_in: Union[int, List[int], str] = "find_min"
    rtol: float = 5e-2


class pLSCFMPEParams(BaseMPEParams):
    """
    Class for storing poly-reference Least Square Complex Frequency (pLSCF) MPE parameters.

    Attributes
    ----------
    sel_freq : list of float or None, optional
        List of selected frequencies for modal parameter extraction. Default is None.
    order_in : int or str, optional
        Specified model order for extraction. Can be an integer or 'find_min'. Default
        is 'find_min'.
    rtol : float, optional
        Relative tolerance for comparing identified frequencies with the selected ones.
        Default is 1e-2.
    """

    sel_freq: Optional[List[float]] = None
    order_in: Union[int, List[int], str] = "find_min"
    rtol: float = 5e-2
