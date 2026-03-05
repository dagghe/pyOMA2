"""
This module provides classes for handling and storing various types of results data
related to the pyOMA2 module.
"""

from __future__ import annotations

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
    Fn: npt.NDArray[np.float64] | None = None  # array of natural frequencies
    Phi: npt.NDArray[np.float64] | None = None  # array of Mode shape vectors


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

    freq: npt.NDArray[np.float64] | None = None
    Sy: npt.NDArray[np.float64] | None = None
    S_val: npt.NDArray[np.float64] | None = None
    S_vec: npt.NDArray[np.float64] | None = None


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
    Xi: npt.NDArray[np.float64] | None = None  # array of damping ratios
    forPlot: list | None = None


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

    freq: npt.NDArray[np.float64] | None = None
    Sy: npt.NDArray[np.float64] | None = None
    Ad: list[npt.NDArray[np.float64]] | None = None
    Bn: list[npt.NDArray[np.float64]] | None = None
    Fn_poles: npt.NDArray[np.float64] | None = None
    Xi_poles: npt.NDArray[np.float64] | None = None
    Phi_poles: npt.NDArray[np.float64] | None = None
    Lab: npt.NDArray[np.float64] | None = None
    Xi: npt.NDArray[np.float64] | None = None
    order_out: list[int] | int | None = None


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
    Phi: npt.NDArray[np.float64] = None
    Fn: npt.NDArray[np.float64] = None
    Fn_std: npt.NDArray[np.float64] = None
    Xi: npt.NDArray[np.float64] = None
    Xi_std: npt.NDArray[np.float64] = None


class ClusteringResult(BaseResult):
    """
    Class to store clustering results and related metadata.

    Attributes
    ----------
    Fn : ndarray of shape (n_modes,)
        Natural frequencies of the modes.
    Xi : ndarray of shape (n_modes,)
        Damping ratios of the modes.
    Phi : ndarray of shape (n_channels, n_modes)
        Mode shapes.
    Fn_fl : ndarray of shape (n_samples,)
        Flattened array of natural frequencies from clustering.
    Xi_fl : ndarray of shape (n_samples,)
        Flattened array of damping ratios from clustering.
    Phi_fl : ndarray of shape (n_samples, n_channels)
        Flattened array of mode shapes from clustering.
    Fn_std_fl : ndarray of shape (n_samples,), optional
        Standard deviation of natural frequencies.
    Xi_std_fl : ndarray of shape (n_samples,), optional
        Standard deviation of damping ratios.
    Phi_std_fl : ndarray of shape (n_samples, n_channels), optional
        Standard deviation of mode shapes.
    order_fl : ndarray of shape (n_samples,)
        Orders corresponding to the flattened data.
    labels : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    dtot : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    medoid_distances : ndarray of shape (n_clusters,)
        Distance of each sample to its cluster medoid.
    order_out : ndarray of shape (n_modes,), optional
        Final order values of the selected modes.
    """

    Fn: npt.NDArray[np.float64] | None = None
    Xi: npt.NDArray[np.float64] | None = None
    Phi: npt.NDArray[np.float64] | None = None
    Fn_fl: npt.NDArray[np.float64] | None = None
    Xi_fl: npt.NDArray[np.float64] | None = None
    Phi_fl: npt.NDArray[np.float64] | None = None
    Fn_std_fl: npt.NDArray[np.float64] | None = None
    Xi_std_fl: npt.NDArray[np.float64] | None = None
    Phi_std_fl: npt.NDArray[np.float64] | None = None
    order_fl: npt.NDArray[np.int64] | None = None
    labels: npt.NDArray[np.int64] | None = None
    dtot: npt.NDArray[np.float64] | None = None
    medoid_distances: npt.NDArray[np.float64] | None = None
    order_out: npt.NDArray[np.int64] | None = None
    Fn_std: npt.NDArray[np.float64] | None = None
    Xi_std: npt.NDArray[np.float64] | None = None
    Phi_std: npt.NDArray[np.float64] | None = None


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

    Obs: npt.NDArray[np.float64] | None = None
    A: list[npt.NDArray[np.float64]] | None = None
    C: list[npt.NDArray[np.float64]] | None = None
    H: npt.NDArray[np.float64] | None = None

    Lambds: npt.NDArray[np.float64] | None = None
    Fn_poles: npt.NDArray[np.float64] | None = None
    Xi_poles: npt.NDArray[np.float64] | None = None
    Phi_poles: npt.NDArray[np.float64] | None = None
    Lab: npt.NDArray[np.float64] | None = None
    Fn_poles_std: npt.NDArray[np.float64] | None = None
    Xi_poles_std: npt.NDArray[np.float64] | None = None
    Phi_poles_std: npt.NDArray[np.float64] | None = None

    Xi: npt.NDArray[np.float64] | None = None
    Fn_std: npt.NDArray[np.float64] | None = None
    Xi_std: npt.NDArray[np.float64] | None = None
    Phi_std: npt.NDArray[np.float64] | None = None
    order_out: list[int] | int | None = None

    clustering_results: dict | None = {}
