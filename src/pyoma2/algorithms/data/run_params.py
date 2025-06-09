"""
This module provides classes for storing run parameters for various modal
analysis algorithms included in the pyOMA2 package. Each class defines
parameters for a specific algorithm or step in a clustering workflow.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import TypedDict


class HCDictType(TypedDict):
    """
    Hard validation criteria (HC) dictionary.

    Attributes
    ----------
    xi_max : float, optional
        Maximum allowable damping ratio. Defaults to None (no limit).
    mpc_lim : float, optional
        Modal Phase Collinearity (MPC) limit. Defaults to None.
    mpd_lim : float, optional
        Mean Phase Deviation (MPD) limit. Defaults to None.
    CoV_max : float, optional
        Maximum coefficient of variation (CoV) of the frequencies.
        Only used when `calc_unc=True`(i.e., when uncertainty
        bounds of modal parameters are calculated).
        Defaults to None.
    """

    xi_max: Optional[float]
    mpc_lim: Optional[float]
    mpd_lim: Optional[float]
    CoV_max: Optional[float]


class SCDictType(TypedDict):
    """
    Soft validation criteria (SC) dictionary.

    Attributes
    ----------
    err_fn : float
        Maximum threshold for relative natural frequency difference. Default: 0.05.
    err_xi : float
        Maximum threshold for relative damping difference. Default: 0.1.
    err_phi : float
        Maximum threshold for Modal Assurance Criterion (MAC) difference.
        Default: 0.05.
    """

    err_fn: float
    err_xi: float
    err_phi: float


class BaseRunParams(BaseModel):
    """
    Base class for storing run parameters for modal analysis algorithms.

    This class configures Pydantic to:
    - Allow attributes to be set from class defaults or keyword arguments.
    - Forbid any extra fields not defined in subclasses.
    """

    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )


class FDDRunParams(BaseRunParams):
    """
    Run parameters for the Frequency Domain Decomposition (FDD) method.

    Attributes
    ----------
    nxseg : int
        Number of points per segment used in spectral density estimation.
        Default: 1024.
    method_SD : {'per', 'cor'}
        Method for spectral density estimation:
        - 'per': Periodogram
        - 'cor': Correlation-based
        Default: 'per'.
    pov : float
        Percentage of overlap between segments when `method_SD='per'`.
        Must be between 0.0 and 1.0. Default: 0.5.
    """

    nxseg: int = 1024
    method_SD: Literal["per", "cor"] = "per"
    pov: float = 0.5


class EFDDRunParams(BaseRunParams):
    """
    Run parameters for the Enhanced Frequency Domain Decomposition (EFDD) method.

    Attributes
    ----------
    nxseg : int
        Number of points per segment used in spectral density estimation.
        Default: 1024.
    method_SD : {'per', 'cor'}
        Method for spectral density estimation:
        - 'per': Periodogram
        - 'cor': Correlation-based
        Default: 'per'.
    pov : float
        Percentage of overlap between segments when `method_SD='per'`.
        Must be between 0.0 and 1.0. Default: 0.5.
    """

    nxseg: int = 1024
    method_SD: Literal["per", "cor"] = "per"
    pov: float = 0.5


class pLSCFRunParams(BaseRunParams):
    """
    Run parameters for the poly-reference Least Square Complex Frequency (pLSCF) method.

    Attributes
    ----------
    ordmax : int
        Maximum model order for the analysis. (Required)
    ordmin : int
        Minimum model order for the analysis. Default: 0.
    nxseg : int
        Number of points per segment for Power Spectral Density (PSD) estimation.
        Default: 1024.
    method_SD : {'per', 'cor'}
        Method for PSD estimation:
        - 'per': Periodogram
        - 'cor': Correlation-based
        Default: 'per'.
    pov : float
        Percentage of overlap between segments when `method_SD='per'`.
        Must be between 0.0 and 1.0. Default: 0.5.
    sc : SCDictType
        Soft validation criteria dictionary. Default:
        `{'err_fn': 0.05, 'err_xi': 0.05, 'err_phi': 0.05}`.
    hc : HCDictType
        Hard validation criteria dictionary. Default:
        `{'xi_max': 0.1, 'mpc_lim': 0.7, 'mpd_lim': 0.3}`.
    """

    ordmax: int
    ordmin: int = 0
    nxseg: int = 1024
    method_SD: Literal["per", "cor"] = "per"
    pov: float = 0.5
    sc: SCDictType = {"err_fn": 0.05, "err_xi": 0.05, "err_phi": 0.05}
    hc: HCDictType = {"xi_max": 0.1, "mpc_lim": 0.7, "mpd_lim": 0.3}


class SSIRunParams(BaseRunParams):
    """
    Run parameters for the Stochastic Subspace Identification (SSI) method.

    Attributes
    ----------
    br : int
        Number of block rows in the Hankel matrix. Default: 20.
    method : {'cov', 'cov_R', 'dat', 'IOcov'}, optional
        Variant of SSI algorithm to use:
        - 'cov': Covariance-driven
        - 'cov_R': Covariance-driven with autocorrelation
        - 'dat': Data-driven
        - 'IOcov': Input-Output covariance-driven
        Default: 'cov'.
    ref_ind : list[int], optional
        List of reference sensor indices for subspace identification.
        Default: None.
    ordmin : int
        Minimum model order to evaluate. Default: 0.
    ordmax : int, optional
        Maximum model order to evaluate. Default: None (no upper limit).
    step : int
        Step size for iterating through model orders. Default: 1.
    sc : SCDictType
        Soft validation criteria dictionary. Default:
        `{'err_fn': 0.05, 'err_xi': 0.1, 'err_phi': 0.05}`.
    hc : HCDictType
        Hard validation criteria dictionary. Default:
        `{'xi_max': 0.2, 'mpc_lim': 0.5, 'mpd_lim': 0.5, 'CoV_max': 0.2}`.
    calc_unc : bool
        Whether to calculate uncertainty bounds for modal parameters.
        Default: False.
    nb : int
        Number of bootstrap samples for uncertainty calculation. Default: 50.
    U : npt.NDArray[np.float64], optional
        Array of input time series (if using input-output SSI). Default: None.
    spetrum : bool
        Whether to compute the frequency spectrum. Default: False.
    fdd_run_params : FDDRunParams, optional
        Instance of FDDRunParams to pass FDD parameters to SSI. Default: None.
    """

    br: int = 20
    method: Optional[Literal["cov", "cov_R", "dat", "IOcov"]] = "cov"
    ref_ind: Optional[List[int]] = None
    ordmin: int = 0
    ordmax: Optional[int] = None
    step: int = 1
    sc: SCDictType = {"err_fn": 0.05, "err_xi": 0.1, "err_phi": 0.05}
    hc: HCDictType = {"xi_max": 0.2, "mpc_lim": 0.5, "mpd_lim": 0.5, "CoV_max": 0.2}
    calc_unc: bool = False
    nb: int = 50
    U: Optional[npt.NDArray[np.float64]] = None
    spetrum: bool = False
    fdd_run_params: Optional[FDDRunParams] = None


# =============================================================================
# CLUSTERING
# =============================================================================


class Step1(BaseRunParams):
    """
    Parameters for the first step of clustering analysis.

    Attributes
    ----------
    hc : bool or {'after'}
        Whether to apply hard validation criteria (HC) to the poles.
        If 'after', HC is applied after pre-clustering. Default: True.
    hc_dict : HCDictType
        Hard validation criteria dictionary. Used only if `hc=True`.
        Default: `{'xi_max': 0.15, 'mpc_lim': 0.8, 'mpd_lim': 0.3, 'CoV_max': 0.15}`.
    sc : bool or {'after'}
        Whether to apply soft validation criteria (SC) to the poles.
        If 'after', SC is applied after pre-clustering. Default: False.
    sc_dict : SCDictType
        Soft validation criteria dictionary. Used only if `sc=True`.
        Default: `{'err_fn': 0.03, 'err_xi': 0.05, 'err_phi': 0.05}`.
    pre_cluster : bool
        Whether to perform a pre-clustering step before validation. Default: False.
    pre_clus_typ : {'GMM', 'kmeans', 'FCMeans'}
        Type of pre-clustering algorithm to use. Default: 'GMM'.
    pre_clus_dist : list of {
        'dfn', 'dxi', 'dlambda', 'dMAC', 'dMPC', 'dMPD', 'MPC', 'MPD'
    }
        Distance metrics to use for pre-clustering. Default: ['dlambda', 'dMAC'].
    transform : {'box-cox'}, optional
        Data transformation method to apply before clustering.
        Default: None.
    """

    hc: Union[bool, Literal["after"]] = True
    hc_dict: HCDictType = {
        "xi_max": 0.15,
        "mpc_lim": 0.8,
        "mpd_lim": 0.3,
        "CoV_max": 0.15,
    }
    sc: Union[bool, Literal["after"]] = False
    sc_dict: SCDictType = {"err_fn": 0.03, "err_xi": 0.05, "err_phi": 0.05}
    pre_cluster: bool = False
    pre_clus_typ: Literal["GMM", "kmeans", "FCMeans"] = "GMM"
    pre_clus_dist: List[
        Literal["dfn", "dxi", "dlambda", "dMAC", "dMPC", "dMPD", "MPC", "MPD"]
    ] = ["dlambda", "dMAC"]
    transform: Optional[Literal["box-cox"]] = None


class Step2(BaseRunParams):
    """
    Parameters for the second step of clustering analysis.

    Attributes
    ----------
    distance : list of {
        'dfn', 'dxi', 'dlambda', 'dMAC', 'dMPC', 'dMPD'
    }
        Distance metrics for clustering. Default: ['dlambda', 'dMAC'].
    weights : {'tot_one'} or list[float], optional
        Weighting scheme for distance metrics. If 'tot_one', all distances sum to one.
        If a list, its length must equal `len(distance)`. Default: None.
    sqrtsqr : bool
        Whether to apply square-root to the sum of squares of distances.
        Default: False.
    algo : {'hdbscan', 'hierarc', 'optics', 'spectral', 'affinity'}
        Clustering algorithm to use. Default: 'hierarc'.
    dc : float or {'auto', 'mu+2sig', '95weib'}, optional
        Distance threshold for hierarchical clustering (only if `algo='hierarc'`).
        - 'auto': Automatic threshold
        - 'mu+2sig': Mean plus two standard deviations
        - '95weib': 95th percentile of Weibull fit
        If None, `n_clusters` must be specified. Default: 'auto'.
    linkage : {'average', 'complete', 'single'}
        Linkage criterion for hierarchical clustering (only if `algo='hierarc'`).
        Default: 'average'.
    min_size : int or {'auto'}, optional
        Minimum cluster size for 'hdbscan' or 'optics'. Default: 'auto'.
    n_clusters : int or {'auto'}, optional
        Number of clusters for 'spectral' or 'hierarc' (if `dc=None`).
        Default: None.
    """

    distance: List[Literal["dfn", "dxi", "dlambda", "dMAC", "dMPC", "dMPD"]] = [
        "dlambda",
        "dMAC",
    ]
    weights: Union[Optional[Literal["tot_one"]], List[float]] = None
    sqrtsqr: bool = False
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
    post_proc : list of post-processing steps to apply to clustering results:
        - 'merge_similar': Merge similar clusters.
        - 'damp_IQR': Damp clusters based on Interquartile Range (IQR) of damping.
        - 'fn_IQR': Filter clusters based on IQR of natural frequencies.
        - 'fn_med': Filter clusters based on median natural frequencies.
        - '1xorder': Filter clusters allowing 1 pole per order.
        - 'min_size': Filter clusters based on minimum size (from Step2).
        - 'min_size_pctg': Filter clusters based on minimum size as a percentage of the largest cluster.
        - 'min_size_kmeans': Filter clusters based on minimum size using k-means.
        - 'min_size_gmm': Filter clusters based on minimum size using Gaussian Mixture Models.
        - 'MTT': Filter clusters based on Modified Thompson Tau technique.
        - 'ABP': Filter clusters based on Adjusted boxplot technique.
    merge_dist : float or {'auto', 'deder'}
        Threshold for merging similar clusters. Default: 'auto'.
    min_pctg : float
        Minimum cluster size expressed as a fraction of the largest cluster.
        Default: 0.3.
    select : {'avg', 'fn_mean_close', 'xi_med_close', 'medoid'}
        Method for selecting the final representative cluster:
        - 'avg': Average of each cluster.
        - 'fn_mean_close': Cluster with mean natural frequency closest to overall mean.
        - 'xi_med_close': Cluster with median damping ratio closest to overall median.
        - 'medoid': Actual medoid of the cluster.
        Default: 'medoid'.
    freqlim : tuple[float, float], optional
        Frequency range limits (min, max) to filter clusters. Default: None.

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
            "ABP",
        ]
    ] = ["merge_similar", "damp_IQR", "fn_IQR", "1xorder", "min_size_pctg", "MTT"]
    merge_dist: Union[Literal["auto", "deder"], float] = "auto"
    min_pctg: float = 0.3
    select: Literal["avg", "fn_mean_close", "xi_med_close", "medoid"] = "medoid"
    freqlim: Optional[Tuple[float, float]] = None


class Clustering(BaseModel):
    """
    Main class for clustering analysis orchestration.

    Attributes
    ----------
    name : str
        Name of the clustering instance.
    steps : tuple[Step1, Step2, Step3], optional
        Tuple of Step1, Step2, and Step3 instances defining the clustering workflow.
        If None, steps are assembled automatically based on `quick`.
    quick : str, optional
        Predefined clustering configuration. Supported values:
        ['Magalhaes', 'Reynders', 'Neu', 'Kvaale', 'Dederichs',
         'hdbscan', 'affinity', 'spectral', 'optics',
         'hier_avg', 'hier_sing', 'hier_sing_nodc'].
        If provided and `steps` is None, `assemble_steps` will populate `steps`.
    freqlim : tuple[float, float], optional
        Global frequency range limits (min, max) for clustering. Passed to
        Step3 if using a `quick` configuration.
    """

    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )
    name: str
    steps: Optional[Tuple[Step1, Step2, Step3]] = None
    quick: Optional[str] = None
    freqlim: Optional[Tuple[float, float]] = None

    @model_validator(mode="after")
    def assemble_steps(self, freqlim):
        """
        Populate `steps` automatically when `quick` is specified and `steps` is None.

        This method runs after model initialization and validation. It reads `self.quick`
        and constructs Step1, Step2, and Step3 instances according to predefined
        strategies. If `quick` is not recognized, raises AttributeError.
        """
        if self.steps is None and self.quick is not None:
            if self.quick == "Magalhaes":
                step1 = Step1(hc=False, sc=False)
                step2 = Step2(
                    distance=["dfn", "dMAC"], algo="hierarc", dc=0.02, linkage="single"
                )
                step3 = Step3(post_proc=["damp_IQR"], select="avg", freqlim=self.freqlim)
                self.steps = (step1, step2, step3)

            elif self.quick == "Reynders":
                step1 = Step1(
                    hc="after",
                    hc_dict={
                        "xi_max": 0.2,
                        "mpc_lim": 0.0,
                        "mpd_lim": 1.0,
                        "CoV_max": np.inf,
                    },
                    sc=False,
                    pre_cluster=True,
                    pre_clus_typ="kmeans",
                    pre_clus_dist=[
                        "dfn",
                        "dxi",
                        "dlambda",
                        "dMAC",
                        "MPC",
                        "MPD",
                    ],
                )
                step2 = Step2(algo="hierarc", dc="mu+2sig", linkage="average")
                step3 = Step3(
                    post_proc=["min_size_kmeans"],
                    select="xi_med_close",
                    freqlim=self.freqlim,
                )
                self.steps = (step1, step2, step3)

            elif self.quick == "Neu":
                step1 = Step1(
                    hc=True,
                    hc_dict={
                        "xi_max": 0.2,
                        "mpc_lim": 0.0,
                        "mpd_lim": 1.0,
                        "CoV_max": np.inf,
                    },
                    sc=False,
                    pre_cluster=True,
                    pre_clus_typ="kmeans",
                    pre_clus_dist=["dfn", "dxi", "dlambda", "dMAC", "dMPD"],
                    transform="box-cox",
                )
                step2 = Step2(algo="hierarc", dc="95weib", linkage="average")
                step3 = Step3(
                    post_proc=["min_size_pctg", "MTT"],
                    min_pctg=0.5,
                    select="avg",
                    freqlim=self.freqlim,
                )
                self.steps = (step1, step2, step3)

            elif self.quick == "Kvaale":
                step1 = Step1(
                    hc=False,
                    sc=True,
                    sc_dict={"err_fn": 0.04, "err_xi": 0.2, "err_phi": 0.1},
                )
                step2 = Step2(algo="hdbscan", min_size=20)
                step3 = Step3(post_proc=["1xorder"], select="avg", freqlim=self.freqlim)
                self.steps = (step1, step2, step3)

            elif self.quick == "Dederichs":
                step1 = Step1(
                    hc=True,
                    hc_dict={
                        "xi_max": 0.2,
                        "mpc_lim": 0.0,
                        "mpd_lim": 1.0,
                        "CoV_max": np.inf,
                    },
                    sc=False,
                    pre_cluster=True,
                    pre_clus_typ="GMM",
                    pre_clus_dist=["dfn", "dMAC", "dMPC"],
                )
                step2 = Step2(algo="hierarc", dc=None, n_clusters="auto")
                step3 = Step3(
                    post_proc=["merge_similar", "1xorder", "min_size_gmm"],
                    merge_dist="deder",
                    select="avg",
                    freqlim=self.freqlim,
                )
                self.steps = (step1, step2, step3)

            elif self.quick == "hdbscan":
                step1 = Step1(sc=False, pre_cluster=True, pre_clus_typ="GMM")
                step2 = Step2(algo="hdbscan")
                step3 = Step3(
                    post_proc=[
                        # "merge_similar",
                        "damp_IQR",
                        "fn_IQR",
                        "1xorder",
                        "min_size_pctg",
                        "MTT",
                    ],
                    freqlim=self.freqlim,
                )
                self.steps = (step1, step2, step3)

            elif self.quick == "affinity":
                step1 = Step1(sc=False, pre_cluster=True, pre_clus_typ="GMM")
                step2 = Step2(algo="affinity")
                step3 = Step3(
                    post_proc=[
                        # "merge_similar",
                        "damp_IQR",
                        "fn_IQR",
                        "1xorder",
                        "min_size_pctg",
                        "MTT",
                    ],
                    freqlim=self.freqlim,
                )
                self.steps = (step1, step2, step3)

            elif self.quick == "spectral":
                step1 = Step1(sc=False, pre_cluster=True, pre_clus_typ="GMM")
                step2 = Step2(algo="spectral")
                step3 = Step3(
                    post_proc=[
                        # "merge_similar",
                        "damp_IQR",
                        "fn_IQR",
                        "1xorder",
                        "min_size_pctg",
                        "MTT",
                    ],
                    freqlim=self.freqlim,
                )
                self.steps = (step1, step2, step3)

            elif self.quick == "optics":
                step1 = Step1(sc=False, pre_cluster=True, pre_clus_typ="GMM")
                step2 = Step2(algo="optics")
                step3 = Step3(
                    post_proc=[
                        "merge_similar",
                        "damp_IQR",
                        "fn_IQR",
                        "1xorder",
                        "min_size_pctg",
                        "MTT",
                    ],
                    freqlim=self.freqlim,
                )
                self.steps = (step1, step2, step3)

            elif self.quick == "hier_avg":
                step1 = Step1(sc=False, pre_cluster=True, pre_clus_typ="GMM")
                step2 = Step2(algo="hierarc", linkage="average")
                step3 = Step3(
                    post_proc=[
                        "damp_IQR",
                        "fn_IQR",
                        "1xorder",
                        "min_size_pctg",
                        "MTT",
                    ],
                    freqlim=self.freqlim,
                )
                self.steps = (step1, step2, step3)

            elif self.quick == "hier_sing":
                step1 = Step1(sc=False, pre_cluster=True, pre_clus_typ="GMM")
                step2 = Step2(algo="hierarc", linkage="single", dc="auto")
                step3 = Step3(
                    post_proc=[
                        "damp_IQR",
                        "fn_IQR",
                        "1xorder",
                        "min_size_pctg",
                        "MTT",
                    ],
                    freqlim=self.freqlim,
                )
                self.steps = (step1, step2, step3)

            elif self.quick == "hier_sing_nodc":
                step1 = Step1(sc=False, pre_cluster=True, pre_clus_typ="GMM")
                step2 = Step2(
                    algo="hierarc", linkage="single", dc=None, n_clusters="auto"
                )
                step3 = Step3(
                    post_proc=[
                        "merge_similar",
                        "damp_IQR",
                        "fn_IQR",
                        "1xorder",
                        "min_size_pctg",
                        "MTT",
                    ],
                    freqlim=self.freqlim,
                )
                self.steps = (step1, step2, step3)

            else:
                raise AttributeError(
                    f"Unknown quick option: {self.quick}\n"
                    "Possible values are:\n"
                    "['Magalhaes', 'Reynders', 'Neu', 'Kvaale', 'Dederichs',\n"
                    "'hdbscan', 'affinity', 'spectral', 'optics',\n"
                    "'hier_avg', 'hier_sing', 'hier_sing_nodc']"
                )
        return self
