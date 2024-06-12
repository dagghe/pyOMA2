"""
This module provides classes for handling and storing various types of results data
related to the pyOMA2 module.
"""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

"""TODO fix type"""


class BaseResult(BaseModel):
    """
    Base class for storing results data.

    Attributes
    ----------
    model_config : ConfigDict
        Configuration dictionary containing model attributes, allowing for arbitrary types.
    Fn : numpy.typing.NDArray
        Array of natural frequencies obtained from modal analysis.
    Phi : numpy.typing.NDArray
        Array of mode shape vectors obtained from modal analysis.
    """

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # dopo MPE o MPE_fromPlot
    Fn: typing.Optional[npt.NDArray[np.float64]] = None  # array of natural frequencies
    Phi: typing.Optional[npt.NDArray[np.float64]] = None  # array of Mode shape vectors


class FDDResult(BaseResult):
    """
    Class for storing Frequency Domain Decomposition (FDD) results data.

    Attributes
    ----------
    freq : numpy.typing.NDArray
        Array of frequencies.
    Sy : numpy.typing.NDArray
        PSD obtained from the FDD analysis.
    S_val : numpy.typing.NDArray
        Singular values of the PSD.
    S_vec : numpy.typing.NDArray
        Singular vectors of the PSD.
    """

    freq: typing.Optional[npt.NDArray[np.float64]] = None
    Sy: typing.Optional[npt.NDArray[np.float64]] = None
    S_val: typing.Optional[npt.NDArray[np.float64]] = None
    S_vec: typing.Optional[npt.NDArray[np.float64]] = None


class EFDDResult(BaseResult):
    """
    Class for storing results data from Enhanced Frequency Domain Decomposition (EFDD)
    and Frequency Spatial Domain Decomposition (FSDD).

    Attributes
    ----------
    freq : numpy.typing.NDArray
        Array of frequencies.
    Sy : numpy.typing.NDArray
        PSD obtained from the analysis.
    S_val : numpy.typing.NDArray
        Singular values of the PSD.
    S_vec : numpy.typing.NDArray
        Singular vectors of the PSD.
    Xi : numpy.typing.NDArray
        Array of damping ratios obtained from modal analysis.
    forPlot : list
        A list to store data for plotting purposes.
    """

    freq: typing.Optional[npt.NDArray[np.float64]] = None
    Sy: typing.Optional[npt.NDArray[np.float64]] = None
    S_val: typing.Optional[npt.NDArray[np.float64]] = None
    S_vec: typing.Optional[npt.NDArray[np.float64]] = None
    # dopo MPE, MPE_forPlot
    Xi: typing.Optional[npt.NDArray[np.float64]] = None  # array of damping ratios
    forPlot: typing.Optional[typing.List] = None


class SSIResult(BaseResult):
    """
    Class for storing results data from Stochastic Subspace Identification (SSI) methods.

    Attributes
    ----------
    A : list of numpy.typing.NDArray
        List of system matrices A from the SSI analysis.
    C : list of numpy.typing.NDArray
        List of system matrices C from the SSI analysis.
    H : numpy.typing.NDArray
        Hankel matrix used in SSI analysis.
    Fn_poles : numpy.typing.NDArray
        Array of identified natural frequencies (poles) from SSI analysis
    xi_poles : numpy.typing.NDArray
        Array of damping ratios corresponding to identified poles
    Phi_poles : numpy.typing.NDArray
        Array of mode shape vectors corresponding to identified poles
    Lab : numpy.typing.NDArray
        Array of labels for the identified poles
    Xi : numpy.typing.NDArray
        Array of damping ratios obtained after modal parameter estimation
    order_out : Union[list[int], int]
        Output order after modal parameter estimation. Can be a list of integers, or a single integer
    """

    A: typing.Optional[typing.List[npt.NDArray[np.float64]]] = None
    C: typing.Optional[typing.List[npt.NDArray[np.float64]]] = None
    H: typing.Optional[npt.NDArray[np.float64]] = None

    Fn_poles: typing.Optional[npt.NDArray[np.float64]] = None
    xi_poles: typing.Optional[npt.NDArray[np.float64]] = None
    Phi_poles: typing.Optional[npt.NDArray[np.float64]] = None
    # lam_poles: npt.NDArray[np.float64]
    Lab: typing.Optional[npt.NDArray[np.float64]] = None
    # dopo MPE, MPE_forPlot
    Xi: typing.Optional[npt.NDArray[np.float64]] = None  # array of damping ratios
    order_out: typing.Union[typing.List[int], int, None] = None


class pLSCFResult(BaseResult):
    """
    Class for storing results data from the poly-reference Least Square Complex Frequency (pLSCF) method.

    Attributes
    ----------
    freq : numpy.typing.NDArray
        Array of frequencies.
    Sy : numpy.typing.NDArray
        PSD obtained from the analysis.
    Ad : list of numpy.typing.NDArray
        Denominator polynomial coefficients from pLSCF analysis.
    Bn : list of numpy.typing.NDArray
        Numerator polynomial coefficients from pLSCF analysis.
    Fn_poles : numpy.typing.NDArray
        Array of identified natural frequencies (poles) from pLSCF analysis.
    xi_poles : numpy.typing.NDArray
        Array of damping ratios corresponding to identified poles.
    Phi_poles : numpy.typing.NDArray
        Array of mode shape vectors corresponding to identified poles.
    Lab : numpy.typing.NDArray
        Array of labels for the identified poles.
    Xi : numpy.typing.NDArray
        Array of damping ratios obtained after modal parameter estimation.
    order_out : Union[list[int], int]
        Output order after modal parameter estimation. Can be a list of integers, or a single integer.
    """

    freq: typing.Optional[npt.NDArray[np.float64]] = None
    Sy: typing.Optional[npt.NDArray[np.float64]] = None
    Ad: typing.Optional[typing.List[npt.NDArray[np.float64]]] = None
    Bn: typing.Optional[typing.List[npt.NDArray[np.float64]]] = None
    Fn_poles: typing.Optional[npt.NDArray[np.float64]] = None
    xi_poles: typing.Optional[npt.NDArray[np.float64]] = None
    Phi_poles: typing.Optional[npt.NDArray[np.float64]] = None
    # lam_poles: npt.NDArray[np.float64]
    Lab: typing.Optional[npt.NDArray[np.float64]] = None
    # dopo MPE, MPE_forPlot
    Xi: typing.Optional[npt.NDArray[np.float64]] = None  # array of damping ratios
    order_out: typing.Union[typing.List[int], int, None] = None


class MsPoserResult(BaseModel):
    # FIXME non molto corretto che sia sotto la cartella algorithms.. ???
    # valutare se creare una cartella apposita per i setups
    """
    Base class for MultiSetup Poser result data.

    Attributes
    ----------
    Phi : numpy.typing.NDArray
        Array of mode shape vectors obtained from MultiSetup Poser analysis.
    Fn : numpy.typing.NDArray
        Array of natural frequencies obtained from MultiSetup Poser analysis (mean value).
    Fn_cov : numpy.typing.NDArray
        Covariance of natural frequencies between setups.
    Xi : numpy.typing.NDArray
        Array of damping ratios obtained from MultiSetup Poser analysis (mean value).
    Xi_cov : numpy.typing.NDArray
        Covariance of damping ratios.
    """

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    Phi: npt.NDArray[np.float64]
    Fn: npt.NDArray[np.float64]
    Fn_cov: npt.NDArray[np.float64]
    Xi: npt.NDArray[np.float64]
    Xi_cov: npt.NDArray[np.float64]
