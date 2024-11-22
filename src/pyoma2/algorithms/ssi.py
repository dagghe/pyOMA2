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

from pyoma2.algorithms.data.result import SSIResult
from pyoma2.algorithms.data.run_params import SSIRunParams
from pyoma2.functions import gen, plot, ssi
from pyoma2.support.sel_from_plot import SelFromPlot

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


# =============================================================================
# SINGLE SETUP
# =============================================================================
# (REF)DATA-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSIdat(BaseAlgorithm[SSIRunParams, SSIResult, typing.Iterable[float]]):
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
        The method used in this SSI algorithm, set to 'dat' by default.
    """

    RunParamCls = SSIRunParams
    ResultCls = SSIResult
    method: typing.Literal["dat"] = "dat"

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
        br = self.run_params.br
        method_hank = self.run_params.method or self.method
        ordmin = self.run_params.ordmin
        ordmax = self.run_params.ordmax
        step = self.run_params.step
        sc = self.run_params.sc
        hc = self.run_params.hc
        calc_unc = self.run_params.calc_unc
        nb = self.run_params.nb

        if self.run_params.ref_ind is not None:
            ref_ind = self.run_params.ref_ind
            Yref = Y[ref_ind, :]
        else:
            Yref = Y

        # Build Hankel matrix
        H, T = ssi.build_hank(
            Y=Y, Yref=Yref, br=br, method=method_hank, calc_unc=calc_unc, nb=nb
        )

        # Get state matrix and output matrix
        (
            Obs,
            A,
            C,
        ) = ssi.SSI_fast(H, br, ordmax, step=step)

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
        self.run_params.sel_freq = sel_freq
        self.run_params.order_in = order_in
        self.run_params.rtol = rtol

        # Get poles
        Fn_pol = self.result.Fn_poles
        Xi_pol = self.result.Xi_poles
        Phi_pol = self.result.Phi_poles
        Lab = self.result.Lab
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
        self.run_params.rtol = rtol

        # Get poles
        Fn_pol = self.result.Fn_poles
        Xi_pol = self.result.Xi_poles
        Phi_pol = self.result.Phi_poles
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
        return fig, ax

    def plot_cluster(
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

        fig, ax = plot.svalH_plot(H=self.result.H, br=self.run_params.br, iter_n=iter_n)
        return fig, ax


# ------------------------------------------------------------------------------
# (REF)COVARIANCE-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
# FIXME ADD REFERENCE
class SSIcov(SSIdat):
    """
    Implements the Covariance-driven Stochastic Subspace Identification (SSI) algorithm
    for single setup experiments.

    This class is an extension of the SSIdat class, adapted for covariance-driven analysis.
    It processes measurement data from a single setup to identify system dynamics and extract
    modal parameters using the SSIcov-ref method.

    Inherits all attributes and methods from SSIdat.

    Attributes
    ----------
    method : str
        The method used in this SSI algorithm, overridden to 'cov' or 'cov_R' for
        covariance-based analysis.

    Methods
    -------
    Inherits all methods from SSIdat with covariance-specific implementations.
    """

    method: typing.Literal["cov_R", "cov"] = "cov"


# =============================================================================
# MULTISETUP
# =============================================================================
# (REF)DATA-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSIdat_MS(SSIdat[SSIRunParams, SSIResult, typing.Iterable[dict]]):
    """
    Implements the Data-Driven Stochastic Subspace Identification (SSI) algorithm for multi-setup
    experiments.

    This class extends the SSIdat class to handle data from multiple experimental setups, with
    moving and reference sensors.

    Inherits all attributes and methods from SSIdat, with focus on multi-setup data handling.

    Attributes
    ----------
    Inherits all attributes from SSIdat.

    Methods
    -------
    Inherits other methods from SSIdat, applicable to multi-setup scenarios.
    """

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


# ------------------------------------------------------------------------------
# (REF)COVARIANCE-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSIcov_MS(SSIdat_MS):
    """
    Implements the Covariance-Driven Stochastic Subspace Identification (SSI) algorithm
    for multi-setup experiments.

    This class extends SSIdat_MS, focusing on the covariance-driven approach to SSI
    for multiple experimental setups.

    Inherits all attributes and methods from SSIdat_MS, adapted for covariance-driven
    analysis methods.

    Attributes
    ----------
    Inherits all attributes from SSIdat_MS.

    Methods
    -------
    Inherits all methods from SSIdat_MS, adapted for covariance-based analysis.
    """

    method: typing.Literal["cov_R", "cov"] = "cov"
