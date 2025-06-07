"""
Stochastic Subspace Identification (SSI) Algorithm Module.
Part of the pyOMA2 package.

Authors:
    Dag Pasca
    Diego Margoni
"""

from __future__ import annotations

import logging
import typing

import numpy as np
from scipy import signal, stats
from tqdm import trange

from pyoma2.algorithms.base import BaseAlgorithm
from pyoma2.algorithms.data.mpe_params import SSIMPEParams
from pyoma2.algorithms.data.result import ClusteringResult, SSIResult
from pyoma2.algorithms.data.run_params import Clustering, FDDRunParams, SSIRunParams
from pyoma2.functions import clus, fdd, gen, plot, ssi
from pyoma2.support.sel_from_plot import SelFromPlot

logger = logging.getLogger(__name__)


# =============================================================================
# SINGLE SETUP
# =============================================================================
# (REF)DATA-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSI(BaseAlgorithm[SSIRunParams, SSIMPEParams, SSIResult, typing.Iterable[float]]):
    """
    Perform Stochastic Subspace Identification (SSI) on single-setup measurement data.

    This class implements the SSI-ref algorithm to identify system modal
    parameters (natural frequencies, damping ratios, mode shapes, etc.) from a single
    setup experiment. It estimates the state-space matrices, constructs Hankel matrices,
    computes poles, applies hard and soft criteria, and optionally estimates the power
    spectral density.

    Attributes
    ----------
    RunParamCls : Type[SSIRunParams]
        Class for algorithm run parameters.
    MPEParamCls : Type[SSIMPEParams]
        Class for modal parameter extraction parameters.
    ResultCls : Type[SSIResult]
        Class for storing algorithm results.
    method : Literal["dat", "cov", "cov_R", "IOcov"]
        Default SSI method. Set to 'cov' by default.
    """

    RunParamCls = SSIRunParams
    MPEParamCls = SSIMPEParams
    ResultCls = SSIResult
    method: typing.Literal["dat", "cov", "cov_R", "IOcov"] = "cov"

    def run(self) -> SSIResult:
        """
        Execute the SSI-ref algorithm on the provided measurement data.

        This method builds the required Hankel matrix (optionally using input U if provided),
        computes the state-space matrices (observability, A, C), estimates poles (natural
        frequencies, damping ratios, mode shapes), applies validation criteria (hard and soft),
        and returns the results encapsulated in an SSIResult object.

        Returns
        -------
        SSIResult
            Contains the observability matrix (Obs), state matrix (A), output matrix (C),
            Hankel matrix (H), eigenvalues (Lambds), identified poles (Fn_poles, Xi_poles, Phi_poles),
            pole labels (Lab), and associated uncertainties (Fn_poles_std, Xi_poles_std, Phi_poles_std).

        Raises
        ------
        ValueError
            If mandatory run parameters are missing or invalid.
        """
        # Transpose measurement data: rows are sensors, columns are time samples
        Y = self.data.T

        # Transpose Input matrix U if provided
        U = self.run_params.U.T if self.run_params.U is not None else None

        br = self.run_params.br
        method_hank = self.run_params.method or self.method
        ordmin = self.run_params.ordmin
        ordmax = self.run_params.ordmax
        step = self.run_params.step
        sc = self.run_params.sc
        hc = self.run_params.hc
        calc_unc = self.run_params.calc_unc
        nb = self.run_params.nb

        # Determine reference data indices for spectrum estimation or SSI
        if self.run_params.ref_ind is not None:
            ref_ind = self.run_params.ref_ind
            Yref = Y[ref_ind, :]
        else:
            Yref = Y

        # Estimate spectrum (PSD) if requested
        if self.run_params.spetrum is True:
            if self.run_params.fdd_run_params is not None:
                fdd_run_params = self.run_params.fdd_run_params
            else:
                fdd_run_params = FDDRunParams()  # Use default FDD parameters
            self.freq, self.Sy = fdd.SD_est(
                Y,
                Yref,
                self.dt,
                fdd_run_params.nxseg,
                fdd_run_params.method_SD,
                fdd_run_params.pov,
            )

        # Build Hankel matrix H and time vector T
        H, T = ssi.build_hank(
            Y=Y, Yref=Yref, br=br, method=method_hank, calc_unc=calc_unc, nb=nb, U=U
        )

        # Compute observability matrix (Obs), state transition (A), and output (C), and innovation covariance G
        Obs, A, C, self.G = ssi.SSI_fast(H, br, ordmax, step=step)

        # Maximum damping ratio for hard criterion
        hc_xi_max = hc["xi_max"]
        # Compute poles (frequencies, damping ratios, mode shapes, eigenvalues, and their uncertainties)
        Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = ssi.SSI_poles(
            Obs,
            A,
            C,
            ordmax,
            self.dt,
            step=step,
            calc_unc=calc_unc,
            H=H,
            T=T,
            xi_max=hc_xi_max,
            HC=True,
        )

        # Hard Criterion: magnitude of modal phase collinearity collinearity (mpc), modal phase deviation (mpd), coefficient of variation (CoV) of the frequencies
        hc_mpc_lim = hc["mpc_lim"]
        hc_mpd_lim = hc["mpd_lim"]
        hc_CoV_max = hc["CoV_max"]

        if hc_mpc_lim is not None:
            mask3 = gen.HC_MPC(Phis, hc_mpc_lim)
            to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                to_mask, mask3, Phis.shape[2]
            )

        if hc_mpd_lim is not None:
            mask4 = gen.HC_MPD(Phis, hc_mpd_lim)
            to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                to_mask, mask4, Phis.shape[2]
            )

        if calc_unc and hc_CoV_max is not None:
            Fn_std, mask5 = gen.HC_CoV(Fns, Fn_std, hc_CoV_max)
            to_mask = [Fns, Xis, Phis, Lambds, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Xi_std, Phi_std = gen.applymask(
                to_mask, mask5, Phis.shape[2]
            )

        # Apply Soft Criteria to label poles
        Lab = gen.SC_apply(
            Fns,
            Xis,
            Phis,
            ordmin,
            ordmax,
            step,
            sc["err_fn"],
            sc["err_xi"],
            sc["err_phi"],
        )

        return SSIResult(
            Obs=Obs,
            A=A,
            C=C,
            H=H,
            Lambds=Lambds,
            Fn_poles=Fns,
            Xi_poles=Xis,
            Phi_poles=Phis,
            Lab=Lab,
            Fn_poles_std=Fn_std,
            Xi_poles_std=Xi_std,
            Phi_poles_std=Phi_std,
        )

    def est_spectrum(self, run_params: typing.Optional[FDDRunParams] = None) -> None:
        """
        Estimate the power spectral density (PSD) of the measurement data using FDD.

        If run_params is not provided, default FDD parameters (FDDRunParams()) are used.

        Parameters
        ----------
        run_params : FDDRunParams, optional
            Configuration parameters for Frequency Domain Decomposition (FDD).

        Returns
        -------
        None

        Notes
        -----
        The computed frequency vector (self.freq) and spectral matrix (self.Sy) are stored
        as attributes of the algorithm instance.
        """
        if run_params is None:
            run_params = FDDRunParams()

        Y = self.data.T
        Yref = Y[self.run_params.ref_ind, :] if self.run_params.ref_ind is not None else Y
        self.freq, self.Sy = fdd.SD_est(
            Y,
            Yref,
            self.dt,
            run_params.nxseg,
            run_params.method_SD,
            run_params.pov,
        )

    def plot_sSy_VS_mSy(
        self,
        order: int,
        freqlim: typing.Optional[tuple[float, float]] = None,
        nSv: typing.Union[int, str] = "all",
    ) -> tuple:
        """
        Plot comparison between synthetic and measured singular value spectra.

        This method computes synthetic spectral singular values from the identified state-space
        matrices (A, C, G) at a specified model order and compares them with the measured spectral
        singular values obtained via FDD.

        Parameters
        ----------
        order : int
            The model order at which to compute the synthetic spectrum. Internally, the order
            is divided by the run_params.step.
        freqlim : tuple(float, float), optional
            Frequency limits for plotting. If None, determined automatically.
        nSv : int or 'all', optional
            Number of singular values to plot. Default is 'all'.

        Returns
        -------
        tuple
            (fig, ax) where fig is the matplotlib Figure and ax is the Axes of the plot.

        Raises
        ------
        ValueError
            If spectrum (self.Sy) or result (self.result) is not available.
        """
        if self.Sy is None:
            raise ValueError("Spectrum not estimated. Call est_spectrum() first.")
        if self.result is None:
            raise ValueError("Run SSI algorithm first. Call run() before plotting.")

        order_index = int(order / self.run_params.step)
        AA = self.result.A
        CC = self.result.C
        GG = self.G

        # Compute sample covariance matrix R0
        R0 = (1 / self.data.shape[0]) * np.dot(self.data.T, self.data)

        # Compute synthetic spectral matrix at the specified order
        sSy = ssi.synt_spctr(
            AA[order_index],
            CC[order_index],
            GG[order_index],
            R0,
            omega=self.freq * 2 * np.pi,
            dt=self.dt,
        )

        # Compute singular values of measured and synthetic spectra
        Sval, _ = fdd.SD_svalsvec(self.Sy)
        sSval, _ = fdd.SD_svalsvec(sSy)

        # Create comparison plot
        fig, ax = plot.spectra_comparison(Sval, sSval, self.freq, freqlim, nSv)
        return fig, ax

    def add_clustering(self, *clusterings: Clustering) -> None:
        """
        Add clustering configuration(s) to the algorithm.

        Stores one or more Clustering objects (algorithm names and steps) so that clustering
        can be run after SSI identification.

        Parameters
        ----------
        *clusterings : Clustering
            One or more Clustering instances containing clustering parameters.

        Returns
        -------
        None
        """
        self.clusterings = {
            **getattr(self, "clusterings", {}),
            **{alg.name: alg.steps for alg in clusterings},
        }

    def run_all_clustering(self) -> None:
        """
        Execute all added clustering configurations sequentially.

        Iterates through each stored clustering configuration and calls run_clustering()
        for each algorithm name in self.clusterings. Results are stored in self.result.

        Returns
        -------
        None
        """
        for i in trange(len(self.clusterings.keys())):
            name = list(self.clusterings.keys())[i]
            self.run_clustering(name=name)
        logger.info("All clustering configurations executed.")

    def run_clustering(self, name: str) -> None:
        """
        Perform clustering on identified poles using a specified configuration.

        Applies hard and soft criteria (HC and SC), pre-clustering filters, distance-based
        clustering (e.g., HDBSCAN, k-means, etc.), and post-processing to group identified poles
        into clusters. The final clustering result is stored in self.result.clustering_results.

        Parameters
        ----------
        name : str
            The name of the clustering configuration to run (must match a key added via add_clustering()).

        Raises
        ------
        ValueError
            If try to run clustering before running the main SSI algorithm (self.result is None).
        AttributeError
            If the provided name is not in self.clusterings.
        """
        if self.result is None:
            raise ValueError("SSI algorithm must be run before clustering (call run()).")

        try:
            steps = self.clusterings[name]
        except KeyError as e:
            raise AttributeError(
                f"'{name}' is not a valid clustering configuration. "
                f"Valid names: {list(self.clusterings.keys())}"
            ) from e

        logger.info("Running clustering '%s'...", name)

        # Extract data from SSI result
        Fns = self.result.Fn_poles
        Xis = self.result.Xi_poles
        Lambds = self.result.Lambds
        Phis = self.result.Phi_poles
        Fn_std = self.result.Fn_poles_std
        Xi_std = self.result.Xi_poles_std
        Phi_std = self.result.Phi_poles_std

        calc_unc = self.run_params.calc_unc
        ordmin = self.run_params.ordmin
        ordmax = self.run_params.ordmax
        step = self.run_params.step

        # Unpack clustering steps
        step1, step2, step3 = steps
        freq_lim = step3.freqlim

        # STEP 1: Hard Criterion (HC) and Soft Criterion (SC) pre-filtering
        hc_dict = step1.hc_dict
        sc_dict = step1.sc_dict
        pre_cluster = step1.pre_cluster
        pre_clus_typ = step1.pre_clus_typ
        pre_clus_dist = step1.pre_clus_dist
        transform = step1.transform

        # Apply frequency limit filter if specified
        if freq_lim is not None:
            Fns, mask_flim = gen.HC_freqlim(Fns, freq_lim)
            to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                to_mask, mask_flim, Phis.shape[2]
            )

        if step1.hc:
            # HC: Damping ratio filter
            if hc_dict["xi_max"] is not None:
                Xis, mask2 = gen.HC_damp(Xis, hc_dict["xi_max"])
                to_mask = [Fns, Lambds, Phis, Fn_std, Xi_std, Phi_std]
                Fns, Lambds, Phis, Fn_std, Xi_std, Phi_std = gen.applymask(
                    to_mask, mask2, Phis.shape[2]
                )
            # HC: Modal phase collinearity (MPC) and modal phase deviation (MPD)
            if hc_dict["mpc_lim"] is not None:
                mask3 = gen.HC_MPC(Phis, hc_dict["mpc_lim"])
                to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                    to_mask, mask3, Phis.shape[2]
                )
            if hc_dict["mpd_lim"] is not None:
                mask4 = gen.HC_MPD(Phis, hc_dict["mpd_lim"])
                to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                    to_mask, mask4, Phis.shape[2]
                )
            # HC: Uncertainty-based CoV filter
            if calc_unc and hc_dict["CoV_max"] is not None:
                Fn_std, mask5 = gen.HC_CoV(Fns, Fn_std, hc_dict["CoV_max"])
                to_mask = [Fns, Xis, Phis, Lambds, Xi_std, Phi_std]
                Fns, Xis, Phis, Lambds, Xi_std, Phi_std = gen.applymask(
                    to_mask, mask5, Phis.shape[2]
                )

        if step1.sc:
            # SC: Label poles based on soft-criteria thresholds
            Lab = gen.SC_apply(
                Fns,
                Xis,
                Phis,
                ordmin,
                ordmax,
                step,
                sc_dict["err_fn"],
                sc_dict["err_xi"],
                sc_dict["err_phi"],
            )
            to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                to_mask, Lab, Phis.shape[2]
            )

        # Compute order grid for each pole
        order = np.full(Lambds.shape, np.nan)
        for kk in range(0, Lambds.shape[1]):
            order[: kk * step, kk] = kk * step

        # Compute MPC and MPD arrays for all poles
        MPC = np.apply_along_axis(gen.MPC, axis=2, arr=Phis)
        MPD = np.apply_along_axis(gen.MPD, axis=2, arr=Phis)

        # Vectorize features across non-masked poles
        non_nan_index = np.argwhere(~np.isnan(Fns.flatten(order="f")))
        features = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std, MPC, MPD, order]
        (
            Fn_fl,
            Xi_fl,
            Phi_fl,
            Lambd_fl,
            Fn_std_fl,
            Xi_std_fl,
            Phi_std_fl,
            MPC_fl,
            MPD_fl,
            order_fl,
        ) = clus.vectorize_features(features, non_nan_index)

        # Pre-clustering: optionally filter stable poles before main clustering
        if pre_cluster:
            data_dict = {
                "Fns": Fns,
                "Xis": Xis,
                "Phis": Phis,
                "Lambdas": Lambds,
                "MPC": MPC,
                "MPD": MPD,
            }
            feat_arr = clus.build_feature_array(
                pre_clus_dist, data_dict, ordmax, step, transform
            )

            if pre_clus_typ == "GMM":
                if step3.merge_dist == "deder":
                    labels_all, dlim = clus.GMM(feat_arr, dist=True)
                else:
                    labels_all = clus.GMM(feat_arr)
            elif pre_clus_typ == "kmeans":
                labels_all = clus.kmeans(feat_arr)
            elif pre_clus_typ == "FCMeans":
                labels_all = clus.FCMeans(feat_arr)
            else:
                raise ValueError(f"Unsupported pre-clustering type: {pre_clus_typ}")

            # Keep only stable cluster (label == 0)
            stab_lab = np.argwhere(labels_all == 0)
            filtered = clus.filter_fl_list(
                [
                    Fn_fl,
                    Xi_fl,
                    Phi_fl,
                    Lambd_fl,
                    Fn_std_fl,
                    Xi_std_fl,
                    Phi_std_fl,
                    MPC_fl,
                    MPD_fl,
                    order_fl,
                ],
                stab_lab,
            )
            (
                Fn_fl,
                Xi_fl,
                Phi_fl,
                Lambd_fl,
                Fn_std_fl,
                Xi_std_fl,
                Phi_std_fl,
                MPC_fl,
                MPD_fl,
                order_fl,
            ) = filtered

        # If Hard or Soft criteria to be applied after initial clustering
        if step1.hc == "after" or step1.sc == "after":
            # Reconstruct 2D arrays from vectorized features
            list_array1d = [Fn_fl, Xi_fl, Lambd_fl, MPC_fl, MPD_fl, order_fl]
            Fns, Xis, Lambds, MPC, MPD, order = clus.oned_to_2d(
                list_array1d, order_fl, Fns.shape, step
            )
            Phis = clus.oned_to_2d([Phi_fl], order_fl, Phis.shape, step)[0]

            if step1.hc == "after":
                if hc_dict["xi_max"] is not None:
                    Xis, mask2 = gen.HC_damp(Xis, hc_dict["xi_max"])
                    to_mask = [Fns, Lambds, Phis, Fn_std, Xi_std, Phi_std]
                    Fns, Lambds, Phis, Fn_std, Xi_std, Phi_std = gen.applymask(
                        to_mask, mask2, Phis.shape[2]
                    )
                if hc_dict["mpc_lim"] is not None:
                    mask3 = gen.HC_MPC(Phis, hc_dict["mpc_lim"])
                    to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                    Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                        to_mask, mask3, Phis.shape[2]
                    )
                if hc_dict["mpd_lim"] is not None:
                    mask4 = gen.HC_MPD(Phis, hc_dict["mpd_lim"])
                    to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                    Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                        to_mask, mask4, Phis.shape[2]
                    )
                if calc_unc and hc_dict["CoV_max"] is not None:
                    Fn_std, mask5 = gen.HC_CoV(Fns, Fn_std, hc_dict["CoV_max"])
                    to_mask = [Fns, Xis, Phis, Lambds, Xi_std, Phi_std]
                    Fns, Xis, Phis, Lambds, Xi_std, Phi_std = gen.applymask(
                        to_mask, mask5, Phis.shape[2]
                    )

            if step1.sc == "after":
                Lab = gen.SC_apply(
                    Fns,
                    Xis,
                    Phis,
                    ordmin,
                    ordmax,
                    step,
                    sc_dict["err_fn"],
                    sc_dict["err_xi"],
                    sc_dict["err_phi"],
                )
                to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                    to_mask, Lab, Phis.shape[2]
                )

            # Re-vectorize features after filtering
            non_nan_index = np.argwhere(~np.isnan(Fns.flatten(order="f")))
            features = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std, MPC, MPD, order]
            (
                Fn_fl,
                Xi_fl,
                Phi_fl,
                Lambd_fl,
                Fn_std_fl,
                Xi_std_fl,
                Phi_std_fl,
                MPC_fl,
                MPD_fl,
                order_fl,
            ) = clus.vectorize_features(features, non_nan_index)

        # STEP 2: Distance-based clustering
        dist_feat = step2.distance
        weights = step2.weights
        sqrtsqr = step2.sqrtsqr
        method = step2.algo
        min_size = step2.min_size
        if min_size == "auto":
            min_size = int(0.1 * ordmax / step)
        dc = step2.dc
        n_clusters = step2.n_clusters

        data_dict = {
            "Fn_fl": Fn_fl,
            "Xi_fl": Xi_fl,
            "Phi_fl": Phi_fl,
            "Lambda_fl": Lambd_fl,
            "MPC_fl": MPC_fl,
            "MPD_fl": MPD_fl,
        }

        dtot = clus.build_tot_dist(dist_feat, data_dict, len(Fn_fl), weights, sqrtsqr)
        dsim = clus.build_tot_simil(dist_feat, data_dict, len(Fn_fl), weights)

        if method == "hdbscan":
            labels_clus = clus.hdbscan(dtot, min_size)
        elif method == "optics":
            labels_clus = clus.optics(dtot, min_size)
        elif method == "hierarc":
            linkage = step2.linkage
            labels_clus = clus.hierarc(
                dtot, dc, linkage, n_clusters, ordmax, step, Fns, Phis
            )
        elif method == "spectral":
            labels_clus = clus.spectral(dsim, n_clusters, ordmax)
        elif method == "affinity":
            labels_clus = clus.affinity(dsim)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        # STEP 3: Post-processing (filter clusters, merge, remove outliers, etc.)
        post_proc = step3.post_proc
        merge_dist = step3.merge_dist
        if merge_dist == "auto":
            x = np.triu_indices_from(dtot, k=0)
            x = dtot[x]
            xs = np.linspace(dtot.min(), dtot.max(), 500)
            kde = stats.gaussian_kde(x)
            pdf = kde(xs)
            maxima_in = signal.argrelmax(pdf)
            dc2_ind = maxima_in[0][0]
            merge_dist = xs[dc2_ind]
        elif merge_dist == "deder":
            merge_dist = dlim

        select = step3.select

        labels = labels_clus.copy()
        unique_labels = set(labels)
        unique_labels.discard(-1)
        clusters = {label: np.where(labels == label)[0] for label in unique_labels}

        for post_i in post_proc:
            if post_i == "fn_med" and calc_unc:
                clusters, labels = clus.post_fn_med(clusters, labels, (Fn_fl, Fn_std_fl))
            if post_i == "fn_IQR":
                clusters, labels = clus.post_fn_IQR(clusters, labels, Fn_fl)
            if post_i == "damp_IQR":
                clusters, labels = clus.post_xi_IQR(clusters, labels, Xi_fl)
            if post_i == "min_size":
                clusters, labels = clus.post_min_size(clusters, labels, min_size)
            if post_i == "min_size_pctg":
                min_pctg = step3.min_pctg
                clusters, labels = clus.post_min_size_pctg(clusters, labels, min_pctg)
            if post_i == "min_size_kmeans":
                clusters, labels = clus.post_min_size_kmeans(labels)
            if post_i == "min_size_gmm":
                clusters, labels = clus.post_min_size_gmm(labels)
            if post_i == "merge_similar":
                clusters, labels = clus.post_merge_similar(
                    clusters, labels, dtot, merge_dist
                )
            if post_i == "1xorder":
                clusters, labels = clus.post_1xorder(clusters, labels, dtot, order_fl)
            if post_i == "MTT":
                clusters, labels = clus.post_MTT(clusters, labels, (Fn_fl, Xi_fl))
            if post_i == "ABP":
                clusters, labels = clus.post_adjusted_boxplot(
                    clusters, labels, (Fn_fl, Xi_fl)
                )

        # Reorder clusters by ascending frequency
        clusters, labels = clus.reorder_clusters(clusters, labels, Fn_fl)

        # Enforce frequency limit on final clusters if needed
        if freq_lim is not None:
            clusters, labels = clus.post_freq_lim(clusters, labels, freq_lim, Fn_fl)

        # Compute medoids for each cluster
        medoids = {}
        for label, indices in clusters.items():
            if len(indices) == 0:
                continue
            submatrix = dtot[np.ix_(indices, indices)]
            total_distances = submatrix.sum(axis=1)
            medoid_index = indices[np.argmin(total_distances)]
            medoids[label] = medoid_index

        medoid_indices = list(medoids.values())
        medoid_distances = dtot[np.ix_(medoid_indices, medoid_indices)]

        # Select final outputs: frequencies, damping, mode shapes, and uncertainties
        flattened_results = (Fn_fl, Xi_fl, Phi_fl.squeeze(), order_fl)
        Fn_out, Xi_out, Phi_out, order_out = clus.output_selection(
            select, clusters, flattened_results, medoid_indices
        )

        if calc_unc:
            flattened_unc = (Fn_std_fl, Xi_std_fl, Phi_std_fl.squeeze(), order_fl)
            Fn_std_out, Xi_std_out, Phi_std_out, order_out = clus.output_selection(
                select, clusters, flattened_unc, medoid_indices
            )
            Phi_std_out = Phi_std_out.T
        else:
            Fn_std_out = None
            Xi_std_out = None
            Phi_std_out = None

        logger.debug("Saving clustering '%s' result.", name)

        risultati = dict(
            Fn=Fn_out,
            Xi=Xi_out,
            Phi=Phi_out.T,
            Fn_fl=Fn_fl,
            Xi_fl=Xi_fl,
            Phi_fl=Phi_fl.squeeze(),
            Fn_std_fl=Fn_std_fl,
            Xi_std_fl=Xi_std_fl,
            Phi_std_fl=Phi_std_fl,
            order_fl=order_fl,
            labels=labels,
            dtot=dtot,
            medoid_distances=medoid_distances,
            order_out=order_out,
            Fn_std=Fn_std_out,
            Xi_std=Xi_std_out,
            Phi_std=Phi_std_out,
        )
        self.result.clustering_results[name] = ClusteringResult(**risultati)

    def plot_silhouette(self, name: str) -> tuple:
        """
        Plot silhouette scores for clustering results (Stabilization Diagram).

        For a given clustering name, this method computes and plots silhouette scores
        from the distance matrix (dtot) and cluster labels, which indicate how well
        each data point fits within its assigned cluster.

        Parameters
        ----------
        name : str
            Name of the clustering result to plot. Use 'all' to plot for every stored clustering.

        Returns
        -------
        tuple or list of tuples
            If name is a single clustering, returns (fig, ax) for that clustering.
            If name == 'all', returns two lists: ([fig1, fig2, ...], [ax1, ax2, ...]).

        Raises
        ------
        ValueError
            If there are no clustering results stored.
        AttributeError
            If the specified clustering name does not exist.
        """
        if name == "all":
            if not self.result.clustering_results:
                raise ValueError("No clustering results available.")
            names = list(self.result.clustering_results.keys())
            figs, axs = [], []
            for nm in names:
                clus_res = self.result.clustering_results.get(nm)
                if clus_res is None:
                    raise AttributeError(
                        f"'{nm}' is not a valid clustering algorithm name. "
                        f"Valid names: {list(self.result.clustering_results.keys())}"
                    )
                labels = clus_res.labels
                dtot = clus_res.dtot
                # Correct minor negative values due to numerical issues
                dtot_c = np.where((dtot < 0) & (np.abs(dtot) < 1e-10), np.abs(dtot), dtot)
                fig, ax = plot.plot_silhouette(dtot_c, labels, nm)
                figs.append(fig)
                axs.append(ax)
            return figs, axs
        else:
            clus_res = self.result.clustering_results.get(name)
            if clus_res is None:
                raise AttributeError(
                    f"'{name}' is not a valid clustering algorithm name. "
                    f"Valid names: {list(self.result.clustering_results.keys())}"
                )
            labels = clus_res.labels
            dtot = clus_res.dtot
            dtot_c = np.where((dtot < 0) & (np.abs(dtot) < 1e-10), np.abs(dtot), dtot)
            fig, ax = plot.plot_silhouette(dtot_c, labels, name)
            return fig, ax

    def plot_stab_cluster(
        self,
        name: str,
        plot_noise: bool = True,
        freqlim: typing.Optional[tuple[float, float]] = None,
    ) -> tuple:
        """
        Plot stabilization diagram overlaid with clustering results.

        Visualizes mode stability (frequency vs. model order) and colors each point
        according to its cluster label. Optionally include noise points.

        Parameters
        ----------
        name : str
            Name of the clustering result to plot. Use 'all' to create figures for all clusterings.
        plot_noise : bool, optional
            If True, noise points (label == -1) are included in the plot. Default is True.
        freqlim : tuple(float, float), optional
            Frequency axis limits as (min_freq, max_freq). If None, auto-scale is used.

        Returns
        -------
        tuple or list of tuples
            If name is single clustering, returns (fig, ax).
            If name == 'all', returns lists: ([fig1, fig2, ...], [ax1, ax2, ...]).

        Raises
        ------
        ValueError
            If there are no clustering results stored.
        AttributeError
            If the specified clustering name does not exist.
        """
        if name == "all":
            if not self.result.clustering_results:
                raise ValueError("No clustering results available.")
            names = list(self.result.clustering_results.keys())
            figs, axs = [], []
            for nm in names:
                clus_res = self.result.clustering_results.get(nm)
                if clus_res is None:
                    raise AttributeError(
                        f"'{nm}' is not a valid clustering algorithm name. "
                        f"Valid names: {list(self.result.clustering_results.keys())}"
                    )
                Fn_fl = clus_res.Fn_fl
                Fn_std_fl = clus_res.Fn_std_fl
                order_fl = clus_res.order_fl
                labels = clus_res.labels

                ordmax = self.run_params.ordmax
                # step = self.run_params.step

                fig, ax = plot.stab_clus_plot(
                    Fn_fl,
                    order_fl,
                    labels,
                    ordmax=ordmax,
                    plot_noise=plot_noise,
                    freqlim=freqlim,
                    Fn_std=Fn_std_fl,
                    name=nm,
                )
                figs.append(fig)
                axs.append(ax)
            return figs, axs
        else:
            clus_res = self.result.clustering_results.get(name)
            if clus_res is None:
                raise AttributeError(
                    f"'{name}' is not a valid clustering algorithm name. "
                    f"Valid names: {list(self.result.clustering_results.keys())}"
                )
            Fn_fl = clus_res.Fn_fl
            Fn_std_fl = clus_res.Fn_std_fl
            order_fl = clus_res.order_fl
            labels = clus_res.labels

            ordmax = self.run_params.ordmax
            # step = self.run_params.step

            fig, ax = plot.stab_clus_plot(
                Fn_fl,
                order_fl,
                labels,
                ordmax=ordmax,
                plot_noise=plot_noise,
                freqlim=freqlim,
                Fn_std=Fn_std_fl,
                name=name,
            )
            return fig, ax

    def plot_freqvsdamp_cluster(self, name: str, plot_noise: bool = True) -> tuple:
        """
        Plot frequency-damping scatter with cluster coloring.

        Shows each pole's natural frequency vs. damping ratio, colored by cluster label.
        Optionally include noise points (label == -1).

        Parameters
        ----------
        name : str
            Name of the clustering result to plot. Use 'all' to create figures for all clusterings.
        plot_noise : bool, optional
            If True, noise points are included. Default is True.

        Returns
        -------
        tuple or list of tuples
            If name is single clustering, returns (fig, ax).
            If name == 'all', returns lists: ([fig1, fig2, ...], [ax1, ax2, ...]).

        Raises
        ------
        ValueError
            If there are no clustering results stored.
        AttributeError
            If the specified clustering name does not exist.
        """
        if name == "all":
            if not self.result.clustering_results:
                raise ValueError("No clustering results available.")
            names = list(self.result.clustering_results.keys())
            figs, axs = [], []
            for nm in names:
                clus_res = self.result.clustering_results.get(nm)
                if clus_res is None:
                    raise AttributeError(
                        f"'{nm}' is not a valid clustering algorithm name. "
                        f"Valid names: {list(self.result.clustering_results.keys())}"
                    )
                Fn_fl = clus_res.Fn_fl
                Xi_fl = clus_res.Xi_fl
                labels = clus_res.labels

                fig, ax = plot.freq_vs_damp_plot(
                    Fn_fl, Xi_fl, labels, plot_noise=plot_noise, name=nm
                )
                figs.append(fig)
                axs.append(ax)
            return figs, axs
        else:
            clus_res = self.result.clustering_results.get(name)
            if clus_res is None:
                raise AttributeError(
                    f"'{name}' is not a valid clustering algorithm name. "
                    f"Valid names: {list(self.result.clustering_results.keys())}"
                )
            Fn_fl = clus_res.Fn_fl
            Xi_fl = clus_res.Xi_fl
            labels = clus_res.labels

            fig, ax = plot.freq_vs_damp_plot(
                Fn_fl, Xi_fl, labels, plot_noise=plot_noise, name=name
            )
            return fig, ax

    def plot_dtot_distrib(
        self, name: str, bins: typing.Union[int, str] = "auto"
    ) -> tuple:
        """
        Plot histogram of pairwise distances from clustering distance matrix (dtot).

        Provides visualization of distance distribution used in clustering.

        Parameters
        ----------
        name : str
            Name of the clustering result to plot. Use 'all' to plot for every stored clustering.
        bins : int or str, optional
            Number of bins or binning strategy for histogram. Default is 'auto'.

        Returns
        -------
        tuple or list of tuples
            If name is single clustering, returns (fig, ax).
            If name == 'all', returns lists: ([fig1, fig2, ...], [ax1, ax2, ...]).

        Raises
        ------
        ValueError
            If there are no clustering results stored.
        AttributeError
            If the specified clustering name does not exist.
        """
        if name == "all":
            if not self.result.clustering_results:
                raise ValueError("No clustering results available.")
            names = list(self.result.clustering_results.keys())
            figs, axs = [], []
            for nm in names:
                clus_res = self.result.clustering_results.get(nm)
                if clus_res is None:
                    raise AttributeError(
                        f"'{nm}' is not a valid clustering algorithm name. "
                        f"Valid names: {list(self.result.clustering_results.keys())}"
                    )
                dtot = clus_res.dtot
                fig, ax = plot.plot_dtot_hist(dtot, bins=bins)
                figs.append(fig)
                axs.append(ax)
            return figs, axs
        else:
            clus_res = self.result.clustering_results.get(name)
            if clus_res is None:
                raise AttributeError(
                    f"'{name}' is not a valid clustering algorithm name. "
                    f"Valid names: {list(self.result.clustering_results.keys())}"
                )
            dtot = clus_res.dtot
            fig, ax = plot.plot_dtot_hist(dtot, bins=bins)
            return fig, ax

    def mpe(
        self,
        sel_freq: typing.List[float],
        order_in: typing.Union[int, str] = "find_min",
        rtol: float = 5e-2,
    ) -> None:
        """
        Extract modal parameters at specified frequencies (stationary MPE).

        Uses previously identified poles (Fn_poles, Xi_poles, Phi_poles) to extract modal
        parameters corresponding to the user-selected frequencies (sel_freq). Optionally,
        the minimum stable order is found automatically if order_in == 'find_min'.

        Parameters
        ----------
        sel_freq : list of float
            Frequencies (in Hz) at which to extract modal parameters.
        order_in : int or 'find_min', optional
            Fixed model order to use for extraction, or 'find_min' to automatically select
            the lowest order where each mode is stable. Default is 'find_min'.
        rtol : float, optional
            Relative tolerance for matching selected frequencies to identified poles.
            Default is 5e-2.

        Returns
        -------
        None

        Notes
        -----
        Saves extracted modal frequencies (self.result.Fn), damping ratios (self.result.Xi),
        mode shapes (self.result.Phi), and their uncertainties (self.result.Fn_std,
        self.result.Xi_std, self.result.Phi_std). The output order (self.result.order_out)
        is also stored.
        """
        super().mpe(sel_freq=sel_freq, order_in=order_in, rtol=rtol)

        # Store MPE parameters
        self.mpe_params.sel_freq = sel_freq
        self.mpe_params.order_in = order_in
        self.mpe_params.rtol = rtol

        # Retrieve identified poles and uncertainties
        Fn_pol = self.result.Fn_poles
        Xi_pol = self.result.Xi_poles
        Phi_pol = self.result.Phi_poles
        Lab = self.result.Lab
        step = self.run_params.step

        Fn_pol_std = self.result.Fn_poles_std
        Xi_pol_std = self.result.Xi_poles_std
        Phi_pol_std = self.result.Phi_poles_std

        # Perform modal parameter extraction
        Fn, Xi, Phi, order_out, Fn_std, Xi_std, Phi_std = ssi.SSI_mpe(
            sel_freq,
            Fn_pol,
            Xi_pol,
            Phi_pol,
            order_in,
            step,
            Lab=Lab,
            rtol=rtol,
            Fn_std=Fn_pol_std,
            Xi_std=Xi_pol_std,
            Phi_std=Phi_pol_std,
        )

        # Store extraction results in self.result
        self.result.order_out = order_out
        self.result.Fn = Fn
        self.result.Xi = Xi
        self.result.Phi = Phi
        self.result.Fn_std = Fn_std
        self.result.Xi_std = Xi_std
        self.result.Phi_std = Phi_std

    def mpe_from_plot(
        self,
        freqlim: typing.Optional[tuple[float, float]] = None,
        rtol: float = 1e-2,
    ) -> None:
        """
        Interactive modal parameter extraction based on frequency selection from a plot.

        Displays an interactive Stabilization Diagram or frequency plot (via SelFromPlot)
        where the user can click on modes of interest. Selected frequencies and corresponding
        orders are then used to extract modal parameters similarly to mpe().

        Parameters
        ----------
        freqlim : tuple(float, float), optional
            Frequency limits for the interactive plot. If None, limits are determined automatically.
            Default is None.
        rtol : float, optional
            Relative tolerance for matching selected frequencies to identified poles.
            Default is 1e-2.

        Returns
        -------
        None

        Notes
        -----
        After user selection, self.result.Fn, Xi, Phi, Fn_std, Xi_std, Phi_std, and order_out
        are populated with extracted modal parameters.
        """
        super().mpe_from_plot(freqlim=freqlim, rtol=rtol)

        # Save relative tolerance
        self.mpe_params.rtol = rtol

        Fn_pol = self.result.Fn_poles
        Xi_pol = self.result.Xi_poles
        Phi_pol = self.result.Phi_poles
        step = self.run_params.step

        Fn_pol_std = self.result.Fn_poles_std
        Xi_pol_std = self.result.Xi_poles_std
        Phi_pol_std = self.result.Phi_poles_std

        # Launch interactive frequency selection
        SFP = SelFromPlot(algo=self, freqlim=freqlim, plot="SSI")
        sel_freq = SFP.result[0]
        order = SFP.result[1]

        # Extract modal parameters from user-selected frequencies
        Fn, Xi, Phi, order_out, Fn_std, Xi_std, Phi_std = ssi.SSI_mpe(
            sel_freq,
            Fn_pol,
            Xi_pol,
            Phi_pol,
            order,
            step,
            Lab=None,
            rtol=rtol,
            Fn_std=Fn_pol_std,
            Xi_std=Xi_pol_std,
            Phi_std=Phi_pol_std,
        )

        # Store results
        self.result.order_out = order_out
        self.result.Fn = Fn
        self.result.Xi = Xi
        self.result.Phi = Phi
        self.result.Fn_std = Fn_std
        self.result.Xi_std = Xi_std
        self.result.Phi_std = Phi_std

    def plot_stab(
        self,
        freqlim: typing.Optional[tuple[float, float]] = None,
        hide_poles: typing.Optional[bool] = True,
        spectrum: bool = False,
        nSv: typing.Union[int, "all"] = "all",
    ) -> tuple:
        """
        Plot the Stabilization Diagram for the SSI algorithm.

        The diagram shows identified pole frequencies across model orders, highlighting
        stable poles. Optionally overlay the measured spectral singular values (CMIF plot).

        Parameters
        ----------
        freqlim : tuple(float, float), optional
            Frequency limits for the plot. If None, determined automatically. Default is None.
        hide_poles : bool, optional
            If True, hide individual poles for clarity. Default is True.
        spectrum : bool, optional
            If True, overlay the measured spectral singular values (CMIF) on a secondary axis.
            Default is False.
        nSv : int or 'all', optional
            Number of singular values for CMIF plot. Default is 'all'.

        Returns
        -------
        tuple
            (fig, ax) where fig is the matplotlib Figure and ax is the primary Axes.

        Raises
        ------
        ValueError
            If the SSI algorithm has not been run (self.result is None).
        """
        if self.result is None:
            raise ValueError("Run SSI algorithm first (call run()).")

        fig, ax = plot.stab_plot(
            Fn=self.result.Fn_poles,
            Lab=self.result.Lab,
            step=self.run_params.step,
            ordmax=self.run_params.ordmax,
            ordmin=self.run_params.ordmin,
            freqlim=freqlim,
            hide_poles=hide_poles,
            fig=None,
            ax=None,
            Fn_std=self.result.Fn_poles_std,
        )

        if spectrum:
            if not hasattr(self, "Sy"):
                self.est_spectrum()
            Sval, _ = fdd.SD_svalsvec(self.Sy)
            ax2 = ax.twinx()
            fig, ax = plot.CMIF_plot(Sval, self.freq, ax=ax2, freqlim=freqlim, nSv=nSv)

        return fig, ax

    def plot_freqvsdamp(
        self,
        freqlim: typing.Optional[tuple[float, float]] = None,
        hide_poles: typing.Optional[bool] = True,
    ) -> tuple:
        """
        Plot frequency vs. damping scatter (cluster plot) for all identified poles.

        Visualizes the distribution of identified mode frequencies and damping ratios,
        helping to identify clusters of physical modes in the dataset.

        Parameters
        ----------
        freqlim : tuple(float, float), optional
            Frequency limits for the plot. If None, determined automatically. Default is None.
        hide_poles : bool, optional
            If True, hide individual pole markers in the scatter. Default is True.

        Returns
        -------
        tuple
            (fig, ax) where fig is the matplotlib Figure and ax is the Axes.

        Raises
        ------
        ValueError
            If the SSI algorithm has not been run (self.result is None).
        """
        if self.result is None:
            raise ValueError("Run SSI algorithm first (call run()).")

        fig, ax = plot.cluster_plot(
            Fn=self.result.Fn_poles,
            Xi=self.result.Xi_poles,
            Lab=self.result.Lab,
            freqlim=freqlim,
            hide_poles=hide_poles,
        )
        return fig, ax

    def plot_svalH(self, iter_n: typing.Optional[int] = None) -> tuple:
        """
        Plot singular values of the Hankel matrix model order.

        This plot helps inspect the singular value decay as a function of block-row index,
        giving insight into model order selection and system observability.

        Parameters
        ----------
        iter_n : int, optional
            Specific iteration (model order) at which to plot singular values. If None,
            uses the maximum order (self.run_params.ordmax). Default is None.

        Returns
        -------
        tuple
            (fig, ax) where fig is the matplotlib Figure and ax is the Axes.

        Raises
        ------
        ValueError
            If the SSI algorithm has not been run (self.result is None).
        """
        if self.result is None:
            raise ValueError("Run SSI algorithm first (call run()).")
        if iter_n is None:
            iter_n = self.run_params.ordmax

        fig, ax = plot.svalH_plot(H=self.result.H, br=self.run_params.br, iter_n=iter_n)
        return fig, ax


# =============================================================================
# MULTISETUP
# =============================================================================
# (REF) STOCHASTIC SUBSPACE IDENTIFICATION
class SSI_MS(SSI[SSIRunParams, SSIMPEParams, SSIResult, typing.Iterable[dict]]):
    """
    Perform Stochastic Subspace Identification (SSI) on multi-setup measurement data.

    Extends the single-setup SSI implementation to handle data from multiple measurement
    setups (e.g., moving and reference sensors). Builds combined observability, state,
    and output matrices across setups and identifies global poles.

    Inherits methods and attributes from SSI, overriding run() to accommodate multi-setup data.

    Attributes
    ----------
    Inherits all attributes from SSI.
    method : Literal["dat", "cov_R", "cov"]
        Default SSI method for multi-setup. Set to 'cov' by default.
    """

    method: typing.Literal["dat", "cov_R", "cov"] = "cov"

    def run(self) -> SSIResult:
        """
        Execute the SSI algorithm across multiple experimental setups.

        Builds a Hankel matrix from concatenated multi-setup data, computes the global
        observability matrix (Obs), state transition matrix (A), and output matrix (C),
        then extracts poles (frequencies, damping, mode shapes) and applies validation criteria.

        Returns
        -------
        SSIResult
            Contains combined observability matrix (Obs), state matrix (A), output matrix (C),
            eigenvalues (Lambds), identified global poles (Fn_poles, Xi_poles, Phi_poles),
            pole labels (Lab), and pole uncertainties (Fn_poles_std, Xi_poles_std, Phi_poles_std)
            (uncertainties are set to None since calc_unc is False for multi-setup).

        Raises
        ------
        ValueError
            If input data does not match expected multi-setup format or run parameters are invalid.
        """
        Y = self.data
        br = self.run_params.br
        method_hank = self.run_params.method or self.method
        ordmin = self.run_params.ordmin
        ordmax = self.run_params.ordmax
        step = self.run_params.step
        sc = self.run_params.sc
        hc = self.run_params.hc

        # Build combined Hankel/observability matrices and retrieve A, C
        Obs, A, C = ssi.SSI_multi_setup(
            Y, self.fs, br, ordmax, step=1, method_hank=method_hank
        )

        # Maximum damping ratio for hard criterion
        hc_xi_max = hc["xi_max"]

        # Compute poles across multi-setup data
        Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = ssi.SSI_poles(
            Obs,
            A,
            C,
            ordmax,
            self.dt,
            step=step,
            calc_unc=False,
            HC=True,
            xi_max=hc_xi_max,
        )

        # Hard criteria thresholds
        hc_mpc_lim = hc["mpc_lim"]
        hc_mpd_lim = hc["mpd_lim"]

        if hc_mpc_lim is not None:
            mask3 = gen.HC_MPC(Phis, hc_mpc_lim)
            to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                to_mask, mask3, Phis.shape[2]
            )

        if hc_mpd_lim is not None:
            mask4 = gen.HC_MPD(Phis, hc_mpd_lim)
            to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                to_mask, mask4, Phis.shape[2]
            )

        # Apply soft criteria to label poles
        Lab = gen.SC_apply(
            Fns,
            Xis,
            Phis,
            ordmin,
            ordmax,
            step,
            sc["err_fn"],
            sc["err_xi"],
            sc["err_phi"],
        )

        return SSIResult(
            Obs=Obs,
            A=A,
            C=C,
            H=None,
            Lambds=Lambds,
            Fn_poles=Fns,
            Xi_poles=Xis,
            Phi_poles=Phis,
            Lab=Lab,
            Fn_poles_std=Fn_std,
            Xi_poles_std=Xi_std,
            Phi_poles_std=Phi_std,
        )
