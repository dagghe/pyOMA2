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
    Data-Driven Stochastic Subspace Identification (SSI) algorithm for single setup
    analysis.

    This class processes measurement data from a single setup experiment to identify
    and extract modal parameters using the SSIdat-ref method.

    Attributes
    ----------
    RunParamCls : Type[SSIRunParams]
        The class of parameters specific to this algorithm's run.
    ResultCls : Type[SSIResult]
        The class of results produced by this algorithm.
    method : str
        The method used in this SSI algorithm, set to 'cov' by default.
    """

    RunParamCls = SSIRunParams
    MPEParamCls = SSIMPEParams
    ResultCls = SSIResult
    method: typing.Literal["dat", "cov", "cov_R", "IOcov"] = "cov"

    def run(self) -> SSIResult:
        """
        Executes the SSIdat algorithm and returns the results.

        Processes the input data using the Data-Driven Stochastic Subspace Identification method.
        Computes state space matrices, modal parameters, and other relevant results.

        Returns
        -------
        SSIResult
            An object containing the computed matrices and modal parameters.
        """
        Y = self.data.T

        # Check for Input matrix U
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

        # Check if the user specifies reference indices
        if self.run_params.ref_ind is not None:
            ref_ind = self.run_params.ref_ind
            Yref = Y[ref_ind, :]
        else:
            Yref = Y

        # Check whether to estimate the spectrum
        if self.run_params.spetrum is True:
            # Check if the user specifies the run_params
            if self.run_params.fdd_run_params is not None:
                fdd_run_params = self.run_params.fdd_run_params
            else:
                fdd_run_params = FDDRunParams()  # use default parameters
            self.freq, self.Sy = fdd.SD_est(
                Y,
                Yref,
                self.dt,
                fdd_run_params.nxseg,
                fdd_run_params.method_SD,
                fdd_run_params.pov,
            )

        # Build Hankel matrix
        H, T = ssi.build_hank(
            Y=Y, Yref=Yref, br=br, method=method_hank, calc_unc=calc_unc, nb=nb, U=U
        )

        # Get state matrix and output matrix
        Obs, A, C, self.G = ssi.SSI_fast(H, br, ordmax, step=step)

        hc_xi_max = hc["xi_max"]
        # Get frequency poles (and damping and mode shapes)
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

        hc_mpc_lim = hc["mpc_lim"]
        hc_mpd_lim = hc["mpd_lim"]
        hc_CoV_max = hc["CoV_max"]

        # HC - MPC and MPD
        if hc_mpc_lim is not None:
            mask3 = gen.HC_MPC(Phis, hc_mpc_lim)
            lista = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                lista, mask3, Phis.shape[2]
            )
        if hc_mpd_lim is not None:
            mask4 = gen.HC_MPD(Phis, hc_mpd_lim)
            lista = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                lista, mask4, Phis.shape[2]
            )

        # HC - maximum covariance
        if Fn_std is not None and hc_CoV_max is not None:
            Fn_std, mask5 = gen.HC_CoV(Fns, Fn_std, hc_CoV_max)
            lista = [Fns, Xis, Phis, Lambds, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Xi_std, Phi_std = gen.applymask(
                lista, mask5, Phis.shape[2]
            )

        # Apply SOFT CRITERIA
        # Get the labels of the poles
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
        Estimate the power spectral density (PSD) of the input data.

        Parameters
        ----------
        run_params : FDDRunParams
            Parameters for the frequency domain decomposition (FDD) method.

        Returns
        -------
        None
        """
        if run_params is None:
            run_params = FDDRunParams()
        self.freq, self.Sy = fdd.SD_est(
            self.data.T,
            self.data.T[self.run_params.ref_ind, :]
            if self.run_params.ref_ind is not None
            else self.data.T,
            self.dt,
            run_params.nxseg,
            run_params.method_SD,
            run_params.pov,
        )

    def plot_sSy_VS_mSy(self, order, freqlim=None, nSv="all") -> None:
        """
        Plot the comparison between synthetic and measured spectra.

        Returns
        -------
        fig, ax : tuple
        """
        if self.Sy is None:
            raise ValueError("Estimate the spectrum first")
        if self.result is None:
            raise ValueError("Run algorithm first")
        order = int(order / self.run_params.step)
        AA = self.result.A
        CC = self.result.C
        GG = self.G
        R0 = 1 / (self.data.shape[0]) * np.dot(self.data[:, :].T, self.data[:, :])

        # Compute the synthetic spectrum
        sSy = ssi.synt_spctr(
            AA[order], CC[order], GG[order], R0, omega=self.freq * 2 * np.pi, dt=self.dt
        )

        # Singular values
        Sval, _ = fdd.SD_svalsvec(self.Sy)
        sSval, _ = fdd.SD_svalsvec(sSy)

        # Plot the comparison of synthetic and measured spectra
        fig, ax = plot.spectra_comparison(Sval, sSval, self.freq, freqlim, nSv)
        return fig, ax

    def add_clustering(self, *clusterings: Clustering) -> None:
        """
        Add clustering configurations to the AutoSSI algorithm.

        Parameters
        ----------
        *clusterings : Clustering
            One or more `Clustering` objects containing clustering configurations to add.

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
        Run all configured clustering methods sequentially.

        Iterates through all clustering configurations added using `add_clustering`
        and executes the clustering process.

        Returns
        -------
        None
        """
        for i in trange(len(self.clusterings.keys())):
            self.run_clustering(name=list(self.clusterings.keys())[i])
        logger.info("all done")

    def run_clustering(self, name: str) -> None:
        """
        Run a specific clustering method by name.

        Parameters
        ----------
        name : str
            Name of the clustering configuration to execute.

        Raises
        ------
        ValueError
            If the clustering method cannot be run before the main algorithm (`run()`) has been executed.

        Returns
        -------
        None
        """
        # TODO run_clustering can only run after run(), raise error otherwise
        # Get needed data
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

        logger.info("Running clustering %s...", name)

        # Get Steps
        steps = self.clusterings[name]
        step1 = steps[0]
        step2 = steps[1]
        step3 = steps[2]

        freq_lim = step3.freq_lim

        # =============================================================================
        # STEP 1
        hc = step1.hc
        sc = step1.sc
        hc_dict = step1.hc_dict
        sc_dict = step1.sc_dict

        pre_cluster = step1.pre_cluster
        pre_clus_typ = step1.pre_clus_typ
        pre_clus_dist = step1.pre_clus_dist
        transform = step1.transform

        # If the user specifies the freq_lim arg we remove the poles to speed up the analysis
        if freq_lim is not None:
            Fns, mask_flim = gen.HC_freqlim(Fns, freq_lim)
            lista = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                lista, mask_flim, Phis.shape[2]
            )

        if hc:
            # HC - damping
            if hc_dict["xi_max"] is not None:
                Xis, mask2 = gen.HC_damp(Xis, hc_dict["xi_max"])
                lista = [Fns, Lambds, Phis, Fn_std, Xi_std, Phi_std]
                Fns, Lambds, Phis, Fn_std, Xi_std, Phi_std = gen.applymask(
                    lista, mask2, Phis.shape[2]
                )
            if hc_dict["mpc_lim"] is not None:
                # HC - MPC and MPD
                mask3 = gen.HC_MPC(Phis, hc_dict["mpc_lim"])
                lista = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                    lista, mask3, Phis.shape[2]
                )
            if hc_dict["mpd_lim"] is not None:
                # HC - MPC and MPD
                mask4 = gen.HC_MPD(Phis, hc_dict["mpd_lim"])
                lista = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                    lista, mask4, Phis.shape[2]
                )
            if calc_unc is not False and hc_dict["CoV_max"] is not None:
                # HC - maximum covariance
                Fn_std, mask5 = gen.HC_CoV(Fns, Fn_std, hc_dict["CoV_max"])
                lista = [Fns, Xis, Phis, Lambds, Xi_std, Phi_std]
                Fns, Xis, Phis, Lambds, Xi_std, Phi_std = gen.applymask(
                    lista, mask5, Phis.shape[2]
                )

        if sc:
            # Apply SOFT CRITERIA
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
            lista = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                lista, Lab, Phis.shape[2]
            )
        # order array (same shape as Fns)
        order = np.full(Lambds.shape, np.nan)
        for _kk in range(0, Lambds.shape[1]):
            order[: _kk * step, _kk] = _kk * step

        MPC = np.apply_along_axis(gen.MPC, axis=2, arr=Phis)
        MPD = np.apply_along_axis(gen.MPD, axis=2, arr=Phis)

        non_nan_index = np.argwhere(~np.isnan(Fns.flatten(order="f")))
        # Vettorializzazione features
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

            stab_lab = np.argwhere(labels_all == 0)

            lista = [
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
            ]
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
            ) = clus.filter_fl_list(lista, stab_lab)

        if hc == "after" or sc == "after":
            # apply hc and sc criteria after initial clustering if == "after"
            list_array1d = [Fn_fl, Xi_fl, Lambd_fl, MPC_fl, MPD_fl, order_fl]
            Fns, Xis, Lambds, MPC, MPD, order = clus.oned_to_2d(
                list_array1d, order_fl, Fns.shape, step
            )
            Phis = clus.oned_to_2d([Phi_fl], order_fl, Phis.shape, step)[0]

            if hc == "after":
                if hc_dict["xi_max"] is not None:
                    Xis, mask2 = gen.HC_damp(Xis, hc_dict["xi_max"])
                    lista = [Fns, Lambds, Phis, Fn_std, Xi_std, Phi_std]
                    Fns, Lambds, Phis, Fn_std, Xi_std, Phi_std = gen.applymask(
                        lista, mask2, Phis.shape[2]
                    )
                if hc_dict["mpc_lim"] is not None:
                    # HC - MPC and MPD
                    mask3 = gen.HC_MPC(Phis, hc_dict["mpc_lim"])
                    lista = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                    Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                        lista, mask3, Phis.shape[2]
                    )
                if hc_dict["mpd_lim"] is not None:
                    # HC - MPC and MPD
                    mask4 = gen.HC_MPD(Phis, hc_dict["mpd_lim"])
                    lista = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                    Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                        lista, mask4, Phis.shape[2]
                    )
                if calc_unc is not False and hc_dict["CoV_max"] is not None:
                    # HC - maximum covariance
                    Fn_std, mask5 = gen.HC_CoV(Fns, Fn_std, hc_dict["CoV_max"])
                    lista = [Fns, Xis, Phis, Lambds, Xi_std, Phi_std]
                    Fns, Xis, Phis, Lambds, Xi_std, Phi_std = gen.applymask(
                        lista, mask5, Phis.shape[2]
                    )
            if sc == "after":
                # Apply SOFT CRITERIA
                # Get the labels of the poles
                Lab = gen.SC_apply(
                    Fns,
                    Xis,
                    Phis,
                    ordmin,
                    ordmax,
                    step,
                    sc_dict.err_fn,
                    sc_dict.err_xi,
                    sc_dict.err_phi,
                )
                lista = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                    lista, Lab, Phis.shape[2]
                )
            # Riappiattisci tutto
            non_nan_index = np.argwhere(~np.isnan(Fns.flatten(order="f")))
            # Vettorializzazione features
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
        # =============================================================================
        # STEP 2
        dist_feat = step2.distance
        weights = step2.weights
        sqrtsqr = step2.sqrtsqr
        method = step2.algo
        min_size = step2.min_size
        if min_size == "auto":
            min_size = int(0.1 * ordmax / step)
        dc = step2.dc
        n_clusters = step2.n_clusters

        # calculate distance_matrix
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

        # =============================================================================
        # STEP 3
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

        # Create a dictionary mapping each label to its corresponding indices
        clusters = {label: np.where(labels == label)[0] for label in unique_labels}

        for post_i in post_proc:
            if post_i == "fn_med" and calc_unc is True:
                # Frequency-based filtering
                flattened_results = (Fn_fl, Fn_std_fl)
                clusters, labels = clus.post_fn_med(clusters, labels, flattened_results)

            if post_i == "fn_IQR":
                # Frequency-based filtering
                clusters, labels = clus.post_fn_IQR(clusters, labels, Fn_fl)

            if post_i == "damp_IQR":
                # Damping-based filtering
                clusters, labels = clus.post_xi_IQR(clusters, labels, Xi_fl)

            if post_i == "min_size":
                # Minimum size based filtering (min_size from step2)
                clusters, labels = clus.post_min_size(clusters, labels, min_size)

            if post_i == "min_size_pctg":
                min_pctg = step3.min_pctg
                # Minimum size based filtering (minimum size min_pctg % of biggest cluster)
                clusters, labels = clus.post_min_size_pctg(clusters, labels, min_pctg)

            if post_i == "min_size_kmeans":
                # Minimum size based filtering (minimum size from kmeans)
                clusters, labels = clus.post_min_size_kmeans(labels)

            if post_i == "min_size_gmm":
                # Minimum size based filtering (minimum size from gaussian mixture)
                clusters, labels = clus.post_min_size_gmm(labels)

            if post_i == "merge_similar":
                # Merge similar clusters
                clusters, labels = clus.post_merge_similar(
                    clusters, labels, dtot, merge_dist
                )

            if post_i == "1xorder":
                # keep only one pole per order¨
                clusters, labels = clus.post_1xorder(clusters, labels, dtot, order_fl)

            if post_i == "MTT":
                # Removing outliers with the modified Thompson Tau Techinique (Neu 2017)
                flattened_results = (Fn_fl, Xi_fl)
                clusters, labels = clus.post_MTT(clusters, labels, flattened_results)

            if post_i == "ABP":
                # Removing outliers with the Adjusted boxplot method
                flattened_results = (Fn_fl, Xi_fl)
                clusters, labels = clus.post_adjusted_boxplot(
                    clusters, labels, flattened_results
                )
        # # Sort clusters in ascending order of frequency
        clusters, labels = clus.reorder_clusters(clusters, labels, Fn_fl)

        # limit the cluster to be inside freq_lim
        if freq_lim is not None:
            clusters, labels = clus.post_freq_lim(clusters, labels, freq_lim, Fn_fl)

        # Recompute medoids for the final clusters
        medoids = {}
        for label, indices in clusters.items():
            if len(indices) == 0:
                continue  # Skip empty clusters
            submatrix = dtot[np.ix_(indices, indices)]
            total_distances = submatrix.sum(axis=1)
            medoid_index = indices[np.argmin(total_distances)]
            medoids[label] = medoid_index

        # Compute inter-medoid distances for the final clusters
        medoid_indices = list(medoids.values())
        medoid_distances = dtot[np.ix_(medoid_indices, medoid_indices)]

        # SELECTION OF RESULTS
        flattened_results = (Fn_fl, Xi_fl, Phi_fl.squeeze(), order_fl)
        Fn_out, Xi_out, Phi_out, order_out = clus.output_selection(
            select, clusters, flattened_results, medoid_indices
        )
        if calc_unc is True:
            flattened_results1 = (Fn_std_fl, Xi_std_fl, Phi_std_fl.squeeze(), order_fl)
            Fn_std_out, Xi_std_out, Phi_std_out, order_out = clus.output_selection(
                select, clusters, flattened_results1, medoid_indices
            )
            Phi_std_out = Phi_std_out.T
        else:
            Fn_std_out = None
            Xi_std_out = None
            Phi_std_out = None

        logger.debug("...saving clustering %s result", name)

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
        self.result.clustering_results[f"{name}"] = ClusteringResult(**risultati)

    def plot_silhouette(self, name):
        """
        Plot the Stabilisation Diagram with clustering results.

        Parameters
        ----------
        name : str
            Name of the clustering result to plot.
        plot_noise : bool, optional
            Whether to include noise points in the plot. Default is True.

        Returns
        -------
        fig, ax : tuple
            Matplotlib figure and axes containing the Stabilisation Diagram.
        """
        if name == "all":
            if len(self.result.clustering_results) == 0:
                raise ValueError("No clustering results available.")
            names = list(self.result.clustering_results.keys())

            figs = []
            axs = []
            for name in names:
                try:
                    clus_res = self.result.clustering_results[name]
                except KeyError as e:
                    raise AttributeError(
                        f"'{name}' is not a valid clustering algorithm name.\n"
                        f"Valid names are: {list(self.result.clustering_results.keys())}"
                    ) from e
                labels = clus_res.labels
                dtot = clus_res.dtot
                # Use np.where to ensure that small negative elements are set to their abs value
                dtot_c = np.where((dtot < 0) & (np.abs(dtot) < 1e-10), np.abs(dtot), dtot)

                fig, ax = plot.plot_silhouette(dtot_c, labels, name)
                figs.append(fig)
                axs.append(ax)
            return figs, axs

        else:
            try:
                clus_res = self.result.clustering_results[name]
            except KeyError as e:
                raise AttributeError(
                    f"'{name}' is not a valid clustering algorithm name.\n"
                    f"Valid names are: {list(self.result.clustering_results.keys())}"
                ) from e
            clus_res = self.result.clustering_results[name]
            labels = clus_res.labels
            dtot = clus_res.dtot
            # Use np.where to ensure that small negative elements are set to their abs value
            dtot_c = np.where((dtot < 0) & (np.abs(dtot) < 1e-10), np.abs(dtot), dtot)

            fig, ax = plot.plot_silhouette(dtot_c, labels, name)
            return fig, ax

    def plot_stab_cluster(self, name, plot_noise=True):
        """
        Plot the Stabilisation Diagram with clustering results.

        Parameters
        ----------
        name : str
            Name of the clustering result to plot.
        plot_noise : bool, optional
            Whether to include noise points in the plot. Default is True.

        Returns
        -------
        fig, ax : tuple
            Matplotlib figure and axes containing the Stabilisation Diagram.
        """
        if name == "all":
            if len(self.result.clustering_results) == 0:
                raise ValueError("No clustering results available.")
            names = list(self.result.clustering_results.keys())

            figs = []
            axs = []
            for name in names:
                try:
                    clus_res = self.result.clustering_results[name]
                except KeyError as e:
                    raise AttributeError(
                        f"'{name}' is not a valid clustering algorithm name.\n"
                        f"Valid names are: {list(self.result.clustering_results.keys())}"
                    ) from e
                Fn_fl = clus_res.Fn_fl
                Fn_std_fl = clus_res.Fn_std_fl
                order_fl = clus_res.order_fl
                labels = clus_res.labels

                ordmax = self.run_params.ordmax
                step = self.run_params.step

                fig, ax = plot.stab_clus_plot(
                    Fn_fl,
                    order_fl,
                    labels,
                    ordmax=ordmax,
                    step=step,
                    plot_noise=plot_noise,
                    Fn_std=Fn_std_fl,
                    name=name,
                )
                figs.append(fig)
                axs.append(ax)
            return figs, axs

        else:
            try:
                clus_res = self.result.clustering_results[name]
            except KeyError as e:
                raise AttributeError(
                    f"'{name}' is not a valid clustering algorithm name.\n"
                    f"Valid names are: {list(self.result.clustering_results.keys())}"
                ) from e

            Fn_fl = clus_res.Fn_fl
            Fn_std_fl = clus_res.Fn_std_fl
            order_fl = clus_res.order_fl
            labels = clus_res.labels

            ordmax = self.run_params.ordmax
            step = self.run_params.step

            fig, ax = plot.stab_clus_plot(
                Fn_fl,
                order_fl,
                labels,
                ordmax=ordmax,
                step=step,
                plot_noise=plot_noise,
                Fn_std=Fn_std_fl,
                name=name,
            )
            return fig, ax

    def plot_freqvsdamp_cluster(self, name, plot_noise=True):
        """
        Plot the frequency-damping cluster plot for a clustering result.

        Parameters
        ----------
        name : str
            Name of the clustering result to plot.
        plot_noise : bool, optional
            Whether to include noise points in the plot. Default is True.

        Returns
        -------
        fig, ax : tuple
            Matplotlib figure and axes containing the frequency-damping cluster plot.
        """
        if name == "all":
            if len(self.result.clustering_results) == 0:
                raise ValueError("No clustering results available.")
            names = list(self.result.clustering_results.keys())

            figs = []
            axs = []
            for name in names:
                try:
                    clus_res = self.result.clustering_results[name]
                except KeyError as e:
                    raise AttributeError(
                        f"'{name}' is not a valid clustering algorithm name.\n"
                        f"Valid names are: {list(self.result.clustering_results.keys())}"
                    ) from e
                Fn_fl = clus_res.Fn_fl
                Xi_fl = clus_res.Xi_fl
                labels = clus_res.labels

                fig, ax = plot.freq_vs_damp_plot(
                    Fn_fl, Xi_fl, labels, plot_noise=plot_noise, name=name
                )
                figs.append(fig)
                axs.append(ax)
            return figs, axs

        else:
            try:
                clus_res = self.result.clustering_results[name]
            except KeyError as e:
                raise AttributeError(
                    f"'{name}' is not a valid clustering algorithm name.\n"
                    f"Valid names are: {list(self.result.clustering_results.keys())}"
                ) from e
            clus_res = self.result.clustering_results[name]
            Fn_fl = clus_res.Fn_fl
            Xi_fl = clus_res.Xi_fl
            labels = clus_res.labels

            fig, ax = plot.freq_vs_damp_plot(
                Fn_fl, Xi_fl, labels, plot_noise=plot_noise, name=name
            )
            return fig, ax

    def plot_dtot_distrib(self, name, bins="auto"):
        """
        Plot the distribution of the distance matrix for clustering.

        Parameters
        ----------
        name : str
            Name of the clustering result to plot.
        bins : str, optional
            Bin size for the histogram. Default is 'auto'.
        Returns
        -------
        fig, ax : tuple
            Matplotlib figure and axes containing the distance matrix histogram.
        """
        if name == "all":
            if len(self.result.clustering_results) == 0:
                raise ValueError("No clustering results available.")
            names = list(self.result.clustering_results.keys())

            figs = []
            axs = []
            for name in names:
                try:
                    clus_res = self.result.clustering_results[name]
                except KeyError as e:
                    raise AttributeError(
                        f"'{name}' is not a valid clustering algorithm name.\n"
                        f"Valid names are: {list(self.result.clustering_results.keys())}"
                    ) from e
                dtot = clus_res.dtot
                fig, ax = plot.plot_dtot_hist(dtot, bins=bins)
                figs.append(fig)
                axs.append(ax)
            return figs, axs
        else:
            try:
                clus_res = self.result.clustering_results[name]
            except KeyError as e:
                raise AttributeError(
                    f"'{name}' is not a valid clustering algorithm name.\n"
                    f"Valid names are: {list(self.result.clustering_results.keys())}"
                ) from e
            clus_res = self.result.clustering_results[name]
            dtot = clus_res.dtot
            fig, ax = plot.plot_dtot_hist(dtot, bins=bins)
            return fig, ax

    def mpe(
        self,
        sel_freq: typing.List[float],
        order_in: typing.Union[int, str] = "find_min",
        rtol: float = 5e-2,
    ) -> typing.Any:
        """
        Extracts the modal parameters at the selected frequencies.

        Parameters
        ----------
        sel_freq : list of float
            Selected frequencies for modal parameter extraction.
        order : int or str, optional
            Model order for extraction, or 'find_min' to auto-determine the minimum stable order.
            Default is 'find_min'.
        rtol : float, optional
            Relative tolerance for comparing frequencies. Default is 5e-2.

        Returns
        -------
        typing.Any
            The extracted modal parameters. The format and content depend on the algorithm's implementation.
        """
        super().mpe(sel_freq=sel_freq, order_in=order_in, rtol=rtol)

        # Save run parameters
        self.mpe_params.sel_freq = sel_freq
        self.mpe_params.order_in = order_in
        self.mpe_params.rtol = rtol

        # Get poles
        Fn_pol = self.result.Fn_poles
        Xi_pol = self.result.Xi_poles
        Phi_pol = self.result.Phi_poles
        Lab = self.result.Lab
        step = self.run_params.step

        # Get cov
        Fn_pol_std = self.result.Fn_poles_std
        Xi_pol_std = self.result.Xi_poles_std
        Phi_pol_std = self.result.Phi_poles_std
        # Extract modal results
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

        # Save results
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
    ) -> typing.Any:
        """
        Interactive method for extracting modal parameters by selecting frequencies from a plot.

        Parameters
        ----------
        freqlim : tuple of float, optional
            Frequency limits for the plot. If None, limits are determined automatically. Default is None.
        rtol : float, optional
            Relative tolerance for comparing frequencies. Default is 1e-2.

        Returns
        -------
        typing.Any
            The extracted modal parameters after interactive selection. Format depends on algorithm's
            implementation.
        """
        super().mpe_from_plot(freqlim=freqlim, rtol=rtol)

        # Save run parameters
        self.mpe_params.rtol = rtol

        # Get poles
        Fn_pol = self.result.Fn_poles
        Xi_pol = self.result.Xi_poles
        Phi_pol = self.result.Phi_poles
        step = self.run_params.step

        # Get cov
        Fn_pol_std = self.result.Fn_poles_std
        Xi_pol_std = self.result.Xi_poles_std
        Phi_pol_std = self.result.Phi_poles_std

        # call interactive plot
        SFP = SelFromPlot(algo=self, freqlim=freqlim, plot="SSI")
        sel_freq = SFP.result[0]
        order = SFP.result[1]

        # and then extract results
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

        # Save results
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
    ) -> typing.Any:
        """
        Plot the Stability Diagram for the SSI algorithms.

        The Stability Diagram helps visualize the stability of identified poles across different
        model orders, making it easier to separate physical poles from spurious ones.

        Parameters
        ----------
        freqlim : tuple of float, optional
            Frequency limits for the plot. If None, limits are determined automatically. Default is None.
        hide_poles : bool, optional
            Option to hide poles in the plot for clarity. Default is True.

        Returns
        -------
        typing.Any
            A tuple containing the matplotlib figure and axes of the Stability Diagram plot.
        """
        if not self.result:
            raise ValueError("Run algorithm first")

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
            # Instantiate a second axes that shares the same x-axis
            ax2 = ax.twinx()
            # Plot the spectrum on the second axes
            fig, ax = plot.CMIF_plot(Sval, self.freq, ax=ax2, freqlim=freqlim, nSv=nSv)

        return fig, ax

    def plot_freqvsdamp(
        self,
        freqlim: typing.Optional[tuple[float, float]] = None,
        hide_poles: typing.Optional[bool] = True,
    ) -> typing.Any:
        """
        Plot the frequency-damping cluster diagram for the identified modal parameters.

        The cluster diagram visualizes the relationship between frequencies and damping
        ratios for the identified poles, helping to identify clusters of physical modes.

        Parameters
        ----------
        freqlim : tuple of float, optional
            Frequency limits for the plot. If None, limits are determined automatically. Default is None.
        hide_poles : bool, optional
            Option to hide poles in the plot for clarity. Default is True.

        Returns
        -------
        typing.Any
            A tuple containing the matplotlib figure and axes of the cluster diagram plot.
        """
        if not self.result:
            raise ValueError("Run algorithm first")

        fig, ax = plot.cluster_plot(
            Fn=self.result.Fn_poles,
            Xi=self.result.Xi_poles,
            Lab=self.result.Lab,
            ordmin=self.run_params.ordmin,
            freqlim=freqlim,
            hide_poles=hide_poles,
        )
        return fig, ax

    def plot_svalH(
        self,
        iter_n: typing.Optional[int] = None,
    ) -> typing.Any:
        """
        Plot the singular values of the Hankel matrix for the SSI algorithm.

        This plot is useful for checking the influence of the number of block-rows, br,
        on the Singular Values of the Hankel matrix.

        Parameters
        ----------
        iter_n : int, optional
            The iteration number for which to plot the singular values. If None, the last
            iteration is used. Default is None.

        Returns
        -------
        typing.Any
            A tuple containing the matplotlib figure and axes of the singular value plot.

        Raises
        ------
        ValueError
            If the algorithm has not been run before plotting.
        """
        if not self.result:
            raise ValueError("Run algorithm first")
        if iter_n is None:
            iter_n = self.run_params.ordmax
        fig, ax = plot.svalH_plot(H=self.result.H, br=self.run_params.br, iter_n=iter_n)
        return fig, ax


# =============================================================================
# MULTISETUP
# =============================================================================
# (REF)DATA-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSI_MS(SSI[SSIRunParams, SSIMPEParams, SSIResult, typing.Iterable[dict]]):
    """
    Implements the Data-Driven Stochastic Subspace Identification (SSI) algorithm for multi-setup
    experiments.

    This class extends the SSIdat class to handle data from multiple experimental setups, with
    moving and reference sensors.

    Inherits all attributes and methods from SSIdat, with focus on multi-setup data handling.

    Attributes
    ----------
    Inherits all attributes from SSI.

    Methods
    -------
    Inherits other methods from SSI, applicable to multi-setup scenarios.
    """

    method: typing.Literal["dat", "cov_R", "cov"] = "cov"

    def run(self) -> SSIResult:
        """
        Executes the SSI algorithm for multiple setups and returns the results.

        Processes the input data from multiple setups using the Data-Driven Stochastic Subspace
        Identification method. It builds Hankel matrices for each setup and computes the state and
        output matrices, along with frequency poles.

        Returns
        -------
        SSIResult
            An object containing the system matrices, poles, damping ratios, and mode shapes across
            multiple setups.
        """

        Y = self.data
        br = self.run_params.br
        method_hank = self.run_params.method or self.method
        ordmin = self.run_params.ordmin
        ordmax = self.run_params.ordmax
        step = self.run_params.step
        sc = self.run_params.sc
        hc = self.run_params.hc

        # Build Hankel matrix and Get observability matrix, state matrix and output matrix
        Obs, A, C = ssi.SSI_multi_setup(
            Y, self.fs, br, ordmax, step=1, method_hank=method_hank
        )

        hc_xi_max = hc["xi_max"]
        # Get frequency poles (and damping and mode shapes)
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

        # VALIDATION CRITERIA FOR POLES
        hc_mpc_lim = hc["mpc_lim"]
        hc_mpd_lim = hc["mpd_lim"]
        # hc_CoV_max = hc["CoV_max"]

        # HC - MPC and MPD
        if hc_mpc_lim is not None:
            mask3 = gen.HC_MPC(Phis, hc_mpc_lim)
            lista = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                lista, mask3, Phis.shape[2]
            )
        if hc_mpd_lim is not None:
            mask4 = gen.HC_MPD(Phis, hc_mpd_lim)
            lista = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                lista, mask4, Phis.shape[2]
            )

        # Apply SOFT CRITERIA
        # Get the labels of the poles
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

        # Return results
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
