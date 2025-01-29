"""
This module provides classes for storing run parameters for various modal analysis
algorithms included in the pyOMA2 module.
"""

from __future__ import annotations

import typing
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import TypedDict


class HCDictType(TypedDict):
    xi_max: Optional[float] = None
    mpc_lim: Optional[float] = None
    mpd_lim: Optional[float] = None
    CoV_max: Optional[float] = None


class SCDictType(TypedDict):
    err_fn: float
    err_xi: float
    err_phi: float


class BaseRunParams(BaseModel):
    """
    Base class for storing run parameters for modal analysis algorithms.
    """

    model_config = ConfigDict(
        from_attributes=True, arbitrary_types_allowed=True, extra="forbid"
    )


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
    """

    nxseg: int = 1024
    method_SD: Literal["per", "cor"] = "per"
    pov: float = 0.5


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
    """

    nxseg: int = 1024
    method_SD: Literal["per", "cor"] = "per"
    pov: float = 0.5


class SSIRunParams(BaseRunParams):
    """
    Parameters for the Stochastic Subspace Identification (SSI) method.

    Attributes
    ----------
    br : int
        Number of block rows in the Hankel matrix.
    method_hank : str or None, optional
        Method used in the SSI algorithm. Options are ['data', 'cov', 'cov_R'].
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
    """

    br: int = 20
    method: str = None
    ref_ind: Optional[List[int]] = None
    ordmin: int = 0
    ordmax: Optional[int] = None
    step: int = 1
    sc: SCDictType = dict(err_fn=0.05, err_xi=0.05, err_phi=0.05)
    hc: HCDictType = dict(xi_max=0.1, mpc_lim=0.5, mpd_lim=0.5, CoV_max=0.05)
    calc_unc: bool = False  # uncertainty calculations
    nb: int = 50  # number of dataset blocks


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
    """

    # METODO 1: run
    ordmax: int
    ordmin: int = 0
    nxseg: int = 1024
    method_SD: Literal["per", "cor"] = "per"
    pov: float = 0.5
    # sgn_basf: int = -1
    # step: int = 1
    sc: SCDictType = dict(err_fn=0.05, err_xi=0.05, err_phi=0.05)
    hc: HCDictType = dict(xi_max=0.1, mpc_lim=0.7, mpd_lim=0.3)


class AutoSSIRunParams(BaseRunParams):
    """
    Run parameters for automated SSI.

    Attributes
    ----------
    br : int
        Number of block rows.
    method : {'cov', 'cov_R', 'dat'}
        Method for assembling the Hankel/subspace matrix.
    ref_ind : list of int, optional
        Indices of reference/projection channels.
    ordmin : int
        Minimum model order.
    ordmax : int
        Maximum model order.
    step : int
        Step size for model order increment.
    calc_unc : bool
        Whether to calculate uncertainty in results.
    nb : int
        Number of data blocks for uncertainty calculations.
    """

    # METODO 1: run
    br: Optional[int] = None
    method: Optional[Literal["cov", "cov_R", "dat"]] = None
    ref_ind: Optional[List[int]] = None
    ordmin: int = 0
    ordmax: typing.Optional[int] = None
    step: int = 1
    calc_unc: bool = False  # uncertainty calculations
    nb: int = 50  # number of dataset blocks
    # hc_dict: HCDictType = dict(xi_max=0.1, mpc_lim=0.7, mpd_lim=0.3, CoV_max=0.05)
    # sc_dict: SCDictType = dict(err_fn=0.05, err_xi=0.05, err_phi=0.05)


# =============================================================================
# CLUSTERING
# =============================================================================


class Step1(BaseRunParams):
    """
    Parameters for the first step of clustering analysis.

    Attributes
    ----------
    hc : {bool, 'after'}
        Whether to apply hard validation criteria (HC) to the poles,
        ('after' delays it until after the pre clustering has been executed).
    hc_dict : HCDictType
        Dictionary of hard criteria (HC).
    sc : {bool, 'after'}
        Whether to apply soft validation criteria (SC) to the poles,
        ('after' delays it until after the pre clustering has been executed).
    sc_dict : SCDictType
        Dictionary of soft criteria (HC).
    pre_cluster : bool
        Whether to pre-cluster data before analysis.
    pre_clus_typ : {'GMM', 'kmeans'}
        Type of pre-clustering algorithm.
    pre_clus_dist : list of {'dfn', 'dxi', 'dlambda', 'dMAC', 'dMPC', 'dMPD', 'MPC', 'MPD'}
        Distance metrics used for pre-clustering.
    transform : {'box-cox'}, optional
        Data transformation method.

    Notes
    -----
    The `hc_dict` and `sc_dict` attributes are used only when `hc` and `sc`
    are set to `True`.
    The `pre_clus_typ`, `pre_clus_dist` and `transform` attributes are used
    only when `pre_cluster` is set to `True`.
    """

    hc: Union[bool, Literal["after"]] = True  # True, False, "after"
    hc_dict: HCDictType = dict(xi_max=0.1, mpc_lim=0.5, mpd_lim=0.5, CoV_max=0.05)
    sc: Union[bool, Literal["after"]] = True  # True, False, "after"
    sc_dict: SCDictType = dict(err_fn=0.03, err_xi=0.1, err_phi=0.05)
    pre_cluster: bool = False
    pre_clus_typ: Literal["GMM", "kmeans"] = "GMM"
    pre_clus_dist: List[
        Literal["dfn", "dxi", "dlambda", "dMAC", "dMPC", "dMPD", "MPC", "MPD"]
    ] = ["dlambda", "dMAC"]
    transform: Optional[Literal["box-cox"]] = None


class Step2(BaseRunParams):
    """
    Parameters for the second step of clustering analysis.

    Attributes
    ----------
    distance : list of {'dfn', 'dxi', 'dlambda', 'dMAC', 'dMPC', 'dMPD'}
        Distance metrics for clustering.
    weights : {'tot_one', None} or list, optional
        Weighting scheme for distance metrics.
    sqrtsqr : bool
        Whether to apply square-root to the sum of squares.
    algo : {'hdbscan', 'hierarc', 'optics', 'spectral', 'affinity'}
        Clustering algorithm to use.
    dc : {float, 'auto', 'mu+2sig', '95weib'}, optional
        Distance threshold for hierarchical clustering.
    linkage : {'average', 'complete', 'single'}
        Linkage criterion for hierarchical clustering.
    min_size : {int, 'auto'}, optional
        Minimum cluster size.
    n_clusters : {int, 'auto'}, optional
        Number of clusters.

    Notes
    -----
    The `dc` and `linkage` parameters are used exclusively for the 'hierarc'
    clustering algorithm. The `min_size` parameter is relevant only for 'hdbscan'
    and 'optics' (but could be used in step3), while `n_clusters` applies to
    'spectral', and hierarc' when 'dc' is None.
    """

    distance: List[Literal["dfn", "dxi", "dlambda", "dMAC", "dMPC", "dMPD"]] = [
        "dlambda",
        "dMAC",
    ]
    weights: Union[Optional[Literal["tot_one"]], list] = (
        None  # None, "tot_one", list with len == len(distance)
    )
    sqrtsqr: bool = False  # bool
    algo: Literal["hdbscan", "hierarc", "optics", "spectral", "affinity"] = "hierarc"
    dc: Optional[Union[float, Literal["auto", "mu+2sig", "95weib"]]] = "auto"
    linkage: Literal["average", "complete", "single"] = "average"
    min_size: Union[Optional[int], Literal["auto"]] = "auto"
    n_clusters: Union[Optional[int], Literal["auto"]] = None


class Step3(BaseRunParams):
    """
    Parameters for the third step of clustering analysis (post-processing).

    Attributes
    ----------
    post_proc : list of {'merge_similar', 'damp_IQR', 'fn_IQR', 'fn_med', '1xorder',
                         'min_size', 'min_size_pctg', 'min_size_kmeans',
                         'min_size_gmm', 'MTT'}
        Post-processing steps to apply to clustering results.
    merge_dist : {float, 'auto', 'deder'}
        Threshold for merging similar clusters.
    min_pctg : float
        Minimum cluster size as a percentage of the largest cluster.
    select : {'avg', 'fn_med_close', 'xi_med_close', 'medoid'}
        Method for selecting final clustering results.
    freq_lim : tuple of float, optional
        Frequency range limits for filtering clusters.

    Warnings
    --------
    The `post_proc` list is applied sequentially, and the order of operations
    affects the results. Steps listed earlier in the sequence are applied before
    later ones. Additionally, the same step can appear multiple times in the list,
    and it will be applied each time it is encountered. Ensure the order and
    repetition are appropriate for your intended analysis.
    """

    post_proc: List[
        Literal[
            "merge_similar",
            "damp_IQR",
            "fn_IQR",
            "fn_med",
            "1xorder",
            "min_size",
            "min_size_pctg",
            "min_size_kmeans",
            "min_size_gmm",
            "MTT",
        ]
    ] = ["merge_similar", "damp_IQR", "fn_IQR", "1xorder", "min_size", "MTT"]
    merge_dist: Union[Literal["auto", "deder"], float] = "auto"
    min_pctg: float = 0.3
    select: Literal["avg", "fn_mean_close", "xi_med_close", "medoid"] = "medoid"
    freq_lim: Optional[tuple] = None


class Clustering(BaseModel):
    """
    Main class for clustering analysis.

    Attributes
    ----------
    name : str
        Name of the clustering instance.
    steps : tuple of (Step1, Step2, Step3), optional
        Steps defining the clustering process.
    quick : {'Magalhaes', 'Reynders', 'Neu', 'Kvaale', 'Dederichs'}, optional
        Predefined configurations for specific clustering strategies.

    Methods
    -------
    assemble_steps()
        Automatically assembles steps based on the `quick` parameter if `steps` is not provided.
    """

    name: str
    steps: Optional[Tuple[Step1, Step2, Step3]] = None
    quick: Optional[str] = None

    @model_validator(mode="after")
    def assemble_steps(self):
        # This logic runs after the model is initialized and validated
        if self.steps is None and self.quick is not None:
            if self.quick == "Magalhaes":
                step1 = Step1(hc=False, sc=False)
                step2 = Step2(
                    distance=["dfn", "dMAC"], algo="hierarc", dc=0.02, linkage="single"
                )
                step3 = Step3(post_proc=["damp_IQR"], select="avg")
                self.steps = (step1, step2, step3)

            elif self.quick == "Reynders":
                step1 = Step1(
                    hc="after",
                    hc_dict=dict(xi_max=0.2, mpc_lim=0.0, mpd_lim=1.0, CoV_max=np.inf),
                    sc=False,
                    pre_cluster=True,
                    pre_clus_typ="kmeans",
                    pre_clus_dist=["dfn", "dxi", "dlambda", "dMAC", "MPC", "MPD"],
                )
                step2 = Step2(algo="hierarc", dc="mu+2sig", linkage="average")
                step3 = Step3(post_proc=["min_size_kmeans"], select="xi_med_close")
                self.steps = (step1, step2, step3)

            elif self.quick == "Neu":
                step1 = Step1(
                    hc=True,
                    hc_dict=dict(xi_max=0.2, mpc_lim=0.0, mpd_lim=1.0, CoV_max=np.inf),
                    sc=False,
                    pre_cluster=True,
                    pre_clus_typ="kmeans",
                    pre_clus_dist=["dfn", "dxi", "dlambda", "dMAC", "dMPD"],
                    transform="box-cox",
                )
                step2 = Step2(algo="hierarc", dc="95weib", linkage="average")
                step3 = Step3(
                    post_proc=("min_size_pctg", "MTT"), min_pctg=0.5, select="avg"
                )
                self.steps = (step1, step2, step3)

            elif self.quick == "Kvaale":
                step1 = Step1(
                    hc=False, sc=True, sc_dict=dict(err_fn=0.04, err_xi=0.2, err_phi=0.1)
                )
                step2 = Step2(algo="hdbscan", min_size=20)
                step3 = Step3(post_proc=["1xorder"], select="avg")
                self.steps = (step1, step2, step3)

            elif self.quick == "Dederichs":
                step1 = Step1(
                    hc=True,
                    hc_dict=dict(xi_max=0.2, mpc_lim=0.0, mpd_lim=1.0, CoV_max=np.inf),
                    sc=False,
                    pre_cluster=True,
                    pre_clus_typ="GMM",
                    pre_clus_dist=["dfn", "dMAC", "dMPC"],
                )
                step2 = Step2(algo="hierarc", dc=None, n_clusters="auto")
                step3 = Step3(
                    post_proc=("merge_similar", "1xorder", "min_size_gmm"),
                    merge_dist="deder",
                    select="avg",
                )
                self.steps = (step1, step2, step3)

        return self
