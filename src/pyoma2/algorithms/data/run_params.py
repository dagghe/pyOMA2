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
    `sel_freq` and `DF` are used in the ``mpe`` method.
    """

    # METODO 1: run
    nxseg: int = 1024
    method_SD: typing.Literal["per", "cor"] = "per"
    pov: float = 0.5
    # METODO 2: mpe e mpe_from_plot
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
    `sel_freq`, `DF1`, `DF2`, `cm`, `MAClim`, `sppk` and `npmax`
    are used in the ``mpe`` method.
    """

    # METODO 1: run
    nxseg: int = 1024
    method_SD: typing.Literal["per", "cor"] = "per"
    pov: float = 0.5
    # METODO 2: mpe e mpe_from_plot
    sel_freq: typing.Optional[npt.NDArray[np.float64]] = None
    DF1: float = 0.1
    DF2: float = 1.0
    cm: int = 1
    MAClim: float = 0.95
    sppk: int = 3
    npmax: int = 20


class SSIRunParams(BaseRunParams):
    """
    Parameters for the Stochastic Subspace Identification (SSI) method.

    Attributes
    ----------
    br : int
        Number of block rows in the Hankel matrix.
    method_hank : str or None, optional
        Method used in the SSI algorithm. Options are ['data', 'cov', 'cov_R'].
        Default is None.
    ref_ind : list of int or None, optional
        List of reference indices used for subspace identification. Default is None.
    ordmin : int, optional
        Minimum model order for the analysis. Default is 0.
    ordmax : int or None, optional
        Maximum model order for the analysis. Default is None.
    step : int, optional
        Step size for iterating through model orders. Default is 1.
    sc : dict, optional
        Soft criteria for the SSI analysis, including thresholds for relative
        frequency difference (`err_fn`), damping ratio difference (`err_xi`), and
        Modal Assurance Criterion (`err_phi`). Default values are {'err_fn': 0.01,
        'err_xi': 0.05, 'err_phi': 0.03}.
    hc : dict, optional
        Hard criteria for the SSI analysis, including settings for presence of
        complex conjugates (`conj`), maximum damping ratio (`xi_max`),
        Modal Phase Collinearity (`mpc_lim`), and Mean Phase Deviation (`mpd_lim`)
        and maximum covariance (`cov_max`). Default values are {'conj': True,
        'xi_max': 0.1, 'mpc_lim': 0.7, 'mpd_lim': 0.3, 'cov_max': 0.2}.
    calc_unc : bool, optional
        Whether to calculate uncertainty. Default is False.
    nb : int, optional
        Number of bootstrap samples to use for uncertainty calculations (default is 100).
    sel_freq : list of float or None, optional
        List of selected frequencies for modal parameter extraction. Default is None.
    order_in : int, list of int, or str
        Specified model order(s) for which the modal parameters are to be extracted.
        If 'find_min', the function attempts to find the minimum model order that provides
        stable poles for each mode of interest.
    rtol : float, optional
        Relative tolerance for comparing identified frequencies with the selected ones.
        Default is 5e-2.

    Notes
    -----
    `sel_freq`, `order_in`, and `rtol` are used in the ``mpe`` method to extract
    modal parameters.
    """

    # METODO 1: run
    br: int
    method: str = None
    ref_ind: typing.Optional[typing.List[int]] = None
    ordmin: int = 0
    ordmax: typing.Optional[int] = None
    step: int = 1
    sc: dict = dict(err_fn=0.01, err_xi=0.05, err_phi=0.03)  # soft criteria
    hc: dict = dict(xi_max=0.1, mpc_lim=0.7, mpd_lim=0.3, CoV_max=0.02)  # hard criteria
    calc_unc: bool = False  # uncertainty calculations
    nb: int = 50  # number of dataset blocks
    # METODO 2: mpe e mpe_from_plot
    sel_freq: typing.Optional[typing.List[float]] = None
    order_in: typing.Union[int, list, str] = "find_min"
    rtol: float = 5e-2


class pLSCFRunParams(BaseRunParams):
    """
    Parameters for the poly-reference Least Square Complex Frequency (pLSCF) method.

    Attributes
    ----------
    ordmax : int
        Maximum order for the analysis.
    ordmin : int, optional
        Minimum order for the analysis. Default is 0.
    nxseg : int, optional
        Number of segments for the Power Spectral Density (PSD) estimation.
        Default is 1024.
    method_SD : str, optional
        Method used for spectral density estimation. Options are ['per', 'cor'].
        Default is 'per'.
    pov : float, optional
        Percentage of overlap between segments for PSD estimation (only applicable
        for 'per' method). Default is 0.5.
    sc : dict, optional
        Soft criteria for the SSI analysis, including thresholds for relative
        frequency difference (`err_fn`), damping ratio difference (`err_xi`), and
        Modal Assurance Criterion (`err_phi`). Default values are {'err_fn': 0.01,
        'err_xi': 0.05, 'err_phi': 0.03}.
    hc : dict, optional
        Hard criteria for the SSI analysis, including settings for presence of
        complex conjugates (`conj`), maximum damping ratio (`xi_max`),
        Modal Phase Collinearity (`mpc_lim`), and Mean Phase Deviation (`mpd_lim`)
        and maximum covariance (`cov_max`). Default values are {'conj': True,
        'xi_max': 0.1, 'mpc_lim': 0.7, 'mpd_lim': 0.3, 'cov_max': 0.2}.
    sel_freq : list of float or None, optional
        List of selected frequencies for modal parameter extraction. Default is None.
    order_in : int or str, optional
        Specified model order for extraction. Can be an integer or 'find_min'. Default
        is 'find_min'.
    deltaf : float, optional
        Frequency bandwidth around each selected frequency. Default is 0.05.
    rtol : float, optional
        Relative tolerance for comparing identified frequencies with the selected ones.
        Default is 1e-2.

    Notes
    -----
    `sel_freq`, `order_in`, `deltaf`, and `rtol` are used in the ``mpe`` method to
    extract modal parameters.
    """

    # METODO 1: run
    ordmax: int
    ordmin: int = 0
    nxseg: int = 1024
    method_SD: typing.Literal["per", "cor"] = "per"
    pov: float = 0.5
    # sgn_basf: int = -1
    # step: int = 1
    sc: typing.Dict = dict(err_fn=0.01, err_xi=0.05, err_phi=0.03)
    hc: typing.Dict = dict(conj=True, xi_max=0.1, mpc_lim=0.7, mpd_lim=0.3)
    # METODO 2: mpe e mpe_from_plot
    sel_freq: typing.Optional[typing.List[float]] = None
    order_in: typing.Union[int, str] = "find_min"
    rtol: float = 5e-2
