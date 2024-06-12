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

    Methods
    -------
    run() -> SSIResult
        Executes the SSIdat algorithm on provided data, returning a SSIResult object with analysis results.
    mpe(...)
        Extracts modal parameters at selected frequencies.
    mpe_fromPlot(...)
        Interactive modal parameter extraction from a plot.
    plot_STDiag(...)
        Plots the Stability Diagram.
    plot_cluster(...)
        Plots the cluster diagram of identified modal parameters.
    plot_mode_g1(...)
        Plots the mode shapes using Geometry1.
    plot_mode_g2(...)
        Plots the mode shapes using Geometry2.
    anim_mode_g2(...)
        Creates an animation of mode shapes using Geometry2.
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
        method = self.run_params.method or self.method
        ordmin = self.run_params.ordmin
        ordmax = self.run_params.ordmax
        step = self.run_params.step
        err_fn = self.run_params.err_fn
        err_xi = self.run_params.err_xi
        err_phi = self.run_params.err_phi
        xi_max = self.run_params.xi_max
        mpc_lim = self.run_params.mpc_lim
        mpd_lim = self.run_params.mpd_lim

        if self.run_params.ref_ind is not None:
            ref_ind = self.run_params.ref_ind
            Yref = Y[ref_ind, :]
        else:
            Yref = Y

        # Build Hankel matrix
        H = ssi.BuildHank(Y, Yref, br, self.fs, method=method)
        # Get state matrix and output matrix
        A, C = ssi.SSI_FAST(H, br, ordmax, step)
        # Get frequency poles (and damping and mode shapes)
        Fn_pol, Sm_pol, Ms_pol = ssi.SSI_Poles(A, C, ordmax, self.dt, step=step)
        # Get the labels of the poles
        Lab = gen.lab_stab(
            Fn_pol,
            Sm_pol,
            Ms_pol,
            ordmin,
            ordmax,
            step,
            err_fn,
            err_xi,
            err_phi,
            xi_max,
            mpc_lim,
            mpd_lim,
        )

        # Return results
        return SSIResult(
            A=A,
            C=C,
            H=H,
            Fn_poles=Fn_pol,
            xi_poles=Sm_pol,
            Phi_poles=Ms_pol,
            Lab=Lab,
        )

    def mpe(
        self,
        sel_freq: typing.List[float],
        order: typing.Union[int, str] = "find_min",
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
        super().mpe(sel_freq=sel_freq, order=order, rtol=rtol)

        # Save run parameters
        self.run_params.sel_freq = sel_freq
        self.run_params.order_in = order
        self.run_params.rtol = rtol

        # Get poles
        Fn_pol = self.result.Fn_poles
        Sm_pol = self.result.xi_poles
        Ms_pol = self.result.Phi_poles
        Lab = self.result.Lab

        # Extract modal results
        Fn_SSI, Xi_SSI, Phi_SSI, order_out = ssi.SSI_MPE(
            sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=Lab, rtol=rtol
        )

        # Save results
        self.result.order_out = order_out
        self.result.Fn = Fn_SSI
        self.result.Xi = Xi_SSI
        self.result.Phi = Phi_SSI

    def mpe_fromPlot(
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
        super().mpe_fromPlot(freqlim=freqlim, rtol=rtol)

        # Save run parameters
        self.run_params.rtol = rtol

        # Get poles
        Fn_pol = self.result.Fn_poles
        Sm_pol = self.result.xi_poles
        Ms_pol = self.result.Phi_poles

        # chiamare plot interattivo
        SFP = SelFromPlot(algo=self, freqlim=freqlim, plot="SSI")
        sel_freq = SFP.result[0]
        order = SFP.result[1]

        # e poi estrarre risultati
        Fn_SSI, Xi_SSI, Phi_SSI, order_out = ssi.SSI_MPE(
            sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=None, rtol=rtol
        )

        # Save results
        self.result.order_out = order_out
        self.result.Fn = Fn_SSI
        self.result.Xi = Xi_SSI
        self.result.Phi = Phi_SSI

    def plot_STDiag(
        self,
        freqlim: typing.Optional[tuple[float, float]] = None,
        hide_poles: typing.Optional[bool] = True,
    ) -> typing.Any:
        """
        Plots the Stability Diagram for the SSIdat algorithm.

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
        fig, ax = plot.Stab_plot(
            Fn=self.result.Fn_poles,
            Lab=self.result.Lab,
            step=self.run_params.step,
            ordmax=self.run_params.ordmax,
            ordmin=self.run_params.ordmin,
            freqlim=freqlim,
            hide_poles=hide_poles,
            fig=None,
            ax=None,
        )
        return fig, ax

    def plot_cluster(
        self,
        freqlim: typing.Optional[tuple[float, float]] = None,
        hide_poles: typing.Optional[bool] = True,
    ) -> typing.Any:
        """
        Plots the frequency-damping cluster diagram for the identified modal parameters.

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

        fig, ax = plot.Cluster_plot(
            Fn=self.result.Fn_poles,
            Sm=self.result.xi_poles,
            Lab=self.result.Lab,
            ordmin=self.run_params.ordmin,
            freqlim=freqlim,
            hide_poles=hide_poles,
        )
        return fig, ax

    def SvalH_plot(
        self,
        iter_n: typing.Optional[int] = None,
    ) -> typing.Any:
        """ """
        if not self.result:
            raise ValueError("Run algorithm first")

        fig, ax = plot.SvalH_plot(H=self.result.H, br=self.run_params.br, iter_n=iter_n)
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
        The method used in this SSI algorithm, overridden to 'cov_bias', 'cov_mm', or 'cov_unb' for
        covariance-based analysis.

    Methods
    -------
    Inherits all methods from SSIdat with covariance-specific implementations.
    """

    method: typing.Literal["cov_R", "cov_mm"] = "cov_mm"


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
    run() -> SSIResult
        Executes the algorithm for multiple setups and returns the identification results.
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
        method = self.run_params.method or self.method
        ordmin = self.run_params.ordmin
        ordmax = self.run_params.ordmax
        step = self.run_params.step
        err_fn = self.run_params.err_fn
        err_xi = self.run_params.err_xi
        err_phi = self.run_params.err_phi
        xi_max = self.run_params.xi_max
        mpc_lim = self.run_params.mpc_lim
        mpd_lim = self.run_params.mpd_lim

        # Build Hankel matrix and Get state matrix and output matrix
        A, C = ssi.SSI_MulSet(
            Y, self.fs, br, ordmax, step=1, methodHank=method, method="FAST"
        )

        # Get frequency poles (and damping and mode shapes)
        Fn_pol, Sm_pol, Ms_pol = ssi.SSI_Poles(A, C, ordmax, self.dt, step=step)
        # Get the labels of the poles
        Lab = gen.lab_stab(
            Fn_pol,
            Sm_pol,
            Ms_pol,
            ordmin,
            ordmax,
            step,
            err_fn,
            err_xi,
            err_phi,
            xi_max,
            mpc_lim,
            mpd_lim,
        )

        # Return results
        return SSIResult(
            A=A,
            C=C,
            Fn_poles=Fn_pol,
            xi_poles=Sm_pol,
            Phi_poles=Ms_pol,
            Lab=Lab,
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

    method: typing.Literal["cov_bias", "cov_mm", "cov_unb"] = "cov_bias"
