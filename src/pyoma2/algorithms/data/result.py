"""
This module provides classes for handling and storing various types of results data
related to the pyOMA2 module.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


class BaseResult(BaseModel):
    """
    Base class for storing results data.

    Attributes
    ----------
    Fn : numpy.NDArray
        Array of natural frequencies obtained from modal analysis.
    Phi : numpy.NDArray
        Array of mode shape vectors obtained from modal analysis.
    """

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # dopo mpe o mpe_from_plot
    Fn: Optional[npt.NDArray[np.float64]] = None  # array of natural frequencies
    Phi: Optional[npt.NDArray[np.float64]] = None  # array of Mode shape vectors


class FDDResult(BaseResult):
    """
    Class for storing Frequency Domain Decomposition (FDD) results data.

    Attributes
    ----------
    freq : numpy.NDArray
        Array of frequencies.
    Sy : numpy.NDArray
        PSD obtained from the FDD analysis.
    S_val : numpy.NDArray
        Singular values of the PSD.
    S_vec : numpy.NDArray
        Singular vectors of the PSD.
    """

    freq: Optional[npt.NDArray[np.float64]] = None
    Sy: Optional[npt.NDArray[np.float64]] = None
    S_val: Optional[npt.NDArray[np.float64]] = None
    S_vec: Optional[npt.NDArray[np.float64]] = None


class EFDDResult(FDDResult):
    """
    Class for storing results data from Enhanced Frequency Domain Decomposition (EFDD)
    and Frequency Spatial Domain Decomposition (FSDD).

    Attributes
    ----------
    freq : numpy.NDArray
        Array of frequencies.
    Sy : numpy.NDArray
        PSD obtained from the analysis.
    S_val : numpy.NDArray
        Singular values of the PSD.
    S_vec : numpy.NDArray
        Singular vectors of the PSD.
    Xi : numpy.NDArray
        Array of damping ratios obtained from modal analysis.
    forPlot : list
        A list to store data for plotting purposes.
    """

    # dopo mpe, MPE_forPlot
    Xi: Optional[npt.NDArray[np.float64]] = None  # array of damping ratios
    forPlot: Optional[List] = None


class SSIResult(BaseResult):
    """
    Class for storing results data from Stochastic Subspace Identification (SSI) methods.

    Attributes
    ----------
    Obs : numpy.NDArray, optional
        Observability matrix obtained from the SSI analysis.
    A : list of numpy.NDArray, optional
        List of system matrices A from the SSI analysis.
    C : list of numpy.NDArray, optional
        List of system matrices C from the SSI analysis.
    H : numpy.NDArray, optional
        Hankel matrix used in SSI analysis.
    Lambds : numpy.NDArray, optional
        Array of eigenvalues from the SSI analysis.
    Fn_poles : numpy.NDArray, optional
        Array of all natural frequencies.
    Xi_poles : numpy.NDArray, optional
        Array of all damping ratios.
    Phi_poles : numpy.NDArray, optional
        Array of all mode shape vectors.
    Lab : numpy.NDArray, optional
        Array of labels for all the poles.
    Fn_poles_std : numpy.NDArray, optional
        Covariance of all natural frequencies.
    Xi_poles_std : numpy.NDArray, optional
        Covariance of all damping ratios.
    Phi_poles_std : numpy.NDArray, optional
        Covariance of all mode shape vectors.
    Xi : numpy.NDArray, optional
        Array of damping ratios.
    order_out : Union[list[int], int], optional
        Output order after modal parameter estimation. Can be a list of integers or a single integer.
    Fn_std : numpy.NDArray, optional
        Covariance of natural frequencies obtained from the analysis.
    Xi_std : numpy.NDArray, optional
        Covariance of damping ratios obtained from the analysis.
    Phi_std : numpy.NDArray, optional
        Covariance of mode shape vectors obtained from the analysis.
    """

    Obs: Optional[npt.NDArray[np.float64]] = None
    A: Optional[List[npt.NDArray[np.float64]]] = None
    C: Optional[List[npt.NDArray[np.float64]]] = None
    H: Optional[npt.NDArray[np.float64]] = None

    Lambds: Optional[npt.NDArray[np.float64]] = None
    Fn_poles: Optional[npt.NDArray[np.float64]] = None
    Xi_poles: Optional[npt.NDArray[np.float64]] = None
    Phi_poles: Optional[npt.NDArray[np.float64]] = None
    Lab: Optional[npt.NDArray[np.float64]] = None
    Fn_poles_std: Optional[npt.NDArray[np.float64]] = None
    Xi_poles_std: Optional[npt.NDArray[np.float64]] = None
    Phi_poles_std: Optional[npt.NDArray[np.float64]] = None
    # dopo mpe, MPE_forPlot
    Xi: Optional[npt.NDArray[np.float64]] = None  # array of damping ratios
    order_out: Optional[Union[int, List[int]]] = None
    Fn_std: Optional[npt.NDArray[np.float64]] = None  # covariance of natural frequencies
    Xi_std: Optional[npt.NDArray[np.float64]] = None  # covariance of damping ratios
    Phi_std: Optional[npt.NDArray[np.float64]] = None  # covariance of mode shapes


class pLSCFResult(BaseResult):
    """
    Class for storing results data from the poly-reference Least Square Complex Frequency (pLSCF) method.

    Attributes
    ----------
    freq : numpy.NDArray
        Array of frequencies.
    Sy : numpy.NDArray
        PSD obtained from the analysis.
    Ad : list of numpy.NDArray
        Denominator polynomial coefficients from pLSCF analysis.
    Bn : list of numpy.NDArray
        Numerator polynomial coefficients from pLSCF analysis.
    Fn_poles : numpy.NDArray
        Array of identified natural frequencies (poles) from pLSCF analysis.
    xi_poles : numpy.NDArray
        Array of damping ratios corresponding to identified poles.
    Phi_poles : numpy.NDArray
        Array of mode shape vectors corresponding to identified poles.
    Lab : numpy.NDArray
        Array of labels for the identified poles.
    Xi : numpy.NDArray
        Array of damping ratios obtained after modal parameter estimation.
    order_out : Union[list[int], int]
        Output order after modal parameter estimation. Can be a list of integers, or a single integer.
    """

    freq: Optional[npt.NDArray[np.float64]] = None
    Sy: Optional[npt.NDArray[np.float64]] = None
    Ad: Optional[List[npt.NDArray[np.float64]]] = None
    Bn: Optional[List[npt.NDArray[np.float64]]] = None
    Fn_poles: Optional[npt.NDArray[np.float64]] = None
    Xi_poles: Optional[npt.NDArray[np.float64]] = None
    Phi_poles: Optional[npt.NDArray[np.float64]] = None
    Lab: Optional[npt.NDArray[np.float64]] = None
    # dopo mpe, MPE_forPlot
    Xi: Optional[npt.NDArray[np.float64]] = None  # array of damping ratios
    order_out: Optional[Union[List[int], int]] = None


class MsPoserResult(BaseResult):
    """
    Base class for MultiSetup Poser result data.

    Attributes
    ----------
    Phi : numpy.NDArray
        Array of mode shape vectors obtained from MultiSetup Poser analysis.
    Fn : numpy.NDArray
        Array of natural frequencies obtained from MultiSetup Poser analysis (mean value).
    Fn_std : numpy.NDArray
        Standard deviation of natural frequencies between setups.
    Xi : numpy.NDArray
        Array of damping ratios obtained from MultiSetup Poser analysis (mean value).
    Xi_std : numpy.NDArray
        Standard deviation of damping ratios.
    """

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    Phi: npt.NDArray[np.float64]
    Fn: npt.NDArray[np.float64]
    Fn_std: npt.NDArray[np.float64]
    Xi: npt.NDArray[np.float64]
    Xi_std: npt.NDArray[np.float64]
