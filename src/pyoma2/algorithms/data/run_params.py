"""
This module provides classes for storing run parameters for various modal analysis
algorithms included in the pyOMA2 module.
"""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


class BaseRunParams(BaseModel):
    """
    Base class for storing run parameters for modal analysis algorithms.

    Attributes
    ----------
    model_config : ConfigDict
        Configuration dictionary containing model attributes, allowing for arbitrary types.
    """

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)


class FDDRunParams(BaseRunParams):
    """
    Class for storing Frequency Domain Decomposition (FDD) run parameters.

    Attributes
    ----------
    nxseg : int, optional
        Number of points per segment, default is 1024.
    method_SD : str, optional ["per", "cor"]
        Method used for spectral density estimation, default is "per".
    pov : float, optional
        Percentage of overlap between segments (only for "per"), default is 0.5.
    sel_freq : numpy.ndarray
        Array of selected frequencies for modal parameter estimation,.
    DF : float, optional
        Frequency resolution for estimation, default is 0.1.

    Notes
    -----
    ``sel_freq`` and ``DF`` are used in the ``MPE`` method.
    """

    # METODO 1: run
    nxseg: int = 1024
    method_SD: str = "per"
    pov: float = 0.5
    # METODO 2: MPE e MPE_fromPlot
    sel_freq: typing.Optional[npt.NDArray[np.float64]] = None
    DF: float = 0.1


class EFDDRunParams(BaseRunParams):
    """
    Class for storing Enhanced Frequency Domain Decomposition (EFDD) run parameters.

    Attributes
    ----------
    nxseg : int, optional
        Number of points per segment, default is 1024.
    method_SD : str, optional ["per", "cor"]
        Method used for spectral density estimation, default is "per".
    pov : float, optional
        Percentage of overlap between segments (only for "per"), default is 0.5.
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
    Notes
    -----
    ``sel_freq``, ``DF1``, ``DF2``, ``cm``, ``MAClim``, ``sppk`` and ``npmax``
    are used in the ``MPE`` method.
    """

    # METODO 1: run
    nxseg: int = 1024
    method_SD: str = "per"
    pov: float = 0.5
    # METODO 2: MPE e MPE_fromPlot
    sel_freq: typing.Optional[npt.NDArray[np.float64]] = None
    DF1: float = 0.1
    DF2: float = 1.0
    cm: int = 1
    MAClim: float = 0.85
    sppk: int = 3
    npmax: int = 20


class SSIRunParams(BaseRunParams):
    """
    Class for storing Stochastic Subspace Identification (SSI) run parameters.

    Attributes
    ----------
    br : int
        Block rows in the Hankel matrix.
    method : str or None
        Method used in the SSI algorithm, one of
        ["data", "cov_mm", "cov_unb", "cov_bias].
    ref_ind : list of int or None
        List of reference indices, default is None.
    ordmin : int
        Minimum order of the model, default is 0.
    ordmax : int or None
        Maximum order of the model, default is None.
    step : int
        Step size for iterating through model orders, default is 1.
    err_fn : float
        Threshold for relative frequency difference, default is 0.01.
    err_xi : float
        Threshold for relative damping ratio difference, default is 0.05.
    err_phi : float
        Threshold for Modal Assurance Criterion (MAC), default is 0.03.
    xi_max : float
        Maximum allowed damping ratio, default is 0.1.
    mpc_lim : float
        xxx, default is 0.7.
    mpd_lim : float
        xxx, default is 0.3.
    sel_freq : list of float or None
        List of selected frequencies for modal parameter extraction.
    order_in : int or str
        Specified model order for extraction, default is 'find_min'.
    deltaf : float
        Frequency bandwidth around each selected frequency, default is 0.05.
    rtol : float
        Relative tolerance for comparing frequencies, default is 1e-2.
    Notes
    -----
    ``sel_freq``, ``order_in``, ``deltaf`` and ``rtol`` are used in the ``MPE`` method.
    """

    # METODO 1: run
    br: int
    method: str = None
    ref_ind: typing.Optional[typing.List[int]] = None
    ordmin: int = 0
    ordmax: typing.Optional[int] = None
    step: int = 1
    err_fn: float = 0.01
    err_xi: float = 0.05
    err_phi: float = 0.03
    xi_max: float = 0.1
    mpc_lim: typing.Optional[typing.Tuple] = 0.7
    mpd_lim: typing.Optional[typing.Tuple] = 0.3
    # METODO 2: MPE e MPE_fromPlot
    sel_freq: typing.Optional[typing.List[float]] = None
    order_in: typing.Union[int, str] = "find_min"
    rtol: float = 5e-2


class pLSCFRunParams(BaseRunParams):
    """
    Class for storing poly-reference Least Square Complex Frequency (pLSCF) run parameters.

    Attributes
    ----------
    ordmax : int
        Maximum order for the analysis.
    ordmin : int
        Minimum order for the analysis, default is 0.
    nxseg : int
        Number of segments for the PSD estimation, default is 1024.
    method_SD : str
        Method used for spectral density estimation, default is 'per'.
    pov : float
        Percentage of overlap between the segments (only for "per"), default is 0.5.
    sgn_basf : int
        Sign of the basis function, default is -1.
    step : int
        Step size for iterating through model orders, default is 1.
    err_fn : float
        Threshold for relative frequency difference, default is 0.01.
    err_xi : float
        Threshold for relative damping ratio difference, default is 0.05.
    err_phi : float
        Threshold for Modal Assurance Criterion (MAC), default is 0.03.
    xi_max : float
        Maximum allowed damping ratio, default is 0.1.
    mpc_lim : float
        xxx, default is 0.7.
    mpd_lim : float
        xxx, default is 0.3.
    sel_freq : list of float or None
        List of selected frequencies for modal parameter extraction.
    order_in : int or str
        Specified model order for extraction, default is 'find_min'.
    deltaf : float
        Frequency bandwidth around each selected frequency, default is 0.05.
    rtol : float
        Relative tolerance for comparing frequencies, default is 1e-2.
    Notes
    -----
    ``sel_freq``, ``order_in``, ``deltaf`` and ``rtol`` are used in the ``MPE`` method.
    """

    # METODO 1: run
    ordmax: int
    ordmin: int = 0
    nxseg: int = 1024
    method_SD: str = "per"
    pov: float = 0.5
    sgn_basf: int = -1
    step: int = 1
    err_fn: float = 0.01
    err_xi: float = 0.05
    err_phi: float = 0.03
    xi_max: float = 0.1
    mpc_lim: typing.Optional[typing.Tuple] = 0.7
    mpd_lim: typing.Optional[typing.Tuple] = 0.3
    # METODO 2: MPE e MPE_fromPlot
    sel_freq: typing.Optional[typing.List[float]] = None
    order_in: typing.Union[int, str] = "find_min"
    rtol: float = 5e-2
