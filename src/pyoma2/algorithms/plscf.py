"""
Poly-reference Least Square Frequency Domain (pLSCF) Module.
Part of the pyOMA2 package.
Authors:
Dag Pasca
Diego Margoni
"""

from __future__ import annotations

import logging
import typing

from pyoma2.algorithms.data.result import pLSCFResult
from pyoma2.algorithms.data.run_params import pLSCFRunParams
from pyoma2.functions import fdd, gen, plot, plscf
from pyoma2.support.sel_from_plot import SelFromPlot

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


# =============================================================================
# SINGLE SETUP
# =============================================================================
class pLSCF(BaseAlgorithm[pLSCFRunParams, pLSCFResult, typing.Iterable[float]]):
    """
    Implementation of the poly-reference Least Square Complex Frequency (pLSCF) algorithm for modal analysis.

    This class inherits from `BaseAlgorithm` and specializes in handling modal analysis computations and
    visualizations based on the pLSCF method. It provides methods to run the analysis, extract modal parameter
    estimation (MPE), plot stability diagrams, cluster diagrams, mode shapes, and animations of mode shapes.

    Parameters
    ----------
    BaseAlgorithm : type
        Inherits from the BaseAlgorithm class with specified type parameters for pLSCFRunParams, pLSCFResult,
        and Iterable[float].

    Attributes
    ----------
    RunParamCls : pLSCFRunParams
        Class attribute for run parameters specific to pLSCF algorithm.
    ResultCls : pLSCFResult
        Class attribute for results specific to pLSCF algorithm.
    """

    RunParamCls = pLSCFRunParams
    ResultCls = pLSCFResult

    def run(self) -> pLSCFResult:
        """
        Execute the pLSCF algorithm to perform modal analysis on the provided data.

        This method conducts a frequency domain analysis using the Least Square Complex Frequency method.
        It computes system matrices, identifies poles, and labels them based on stability and other
        criteria.

        Returns
        -------
        pLSCFResult
            An instance of `pLSCFResult` containing the analysis results, including frequencies, system
            matrices, identified poles, and their labels.
        """
        Y = self.data.T
        nxseg = self.run_params.nxseg
        method = self.run_params.method_SD
        pov = self.run_params.pov
        sgn_basf = self.run_params.sgn_basf
        ordmax = self.run_params.ordmax
        ordmin = self.run_params.ordmin
        err_fn = self.run_params.err_fn
        err_xi = self.run_params.err_xi
        err_phi = self.run_params.err_phi
        xi_max = self.run_params.xi_max
        mpc_lim = self.run_params.mpc_lim
        mpd_lim = self.run_params.mpd_lim

        freq, Sy = fdd.SD_Est(Y, Y, self.dt, nxseg, method=method, pov=pov)

        Ad, Bn = plscf.pLSCF(Sy, self.dt, ordmax, sgn_basf=sgn_basf)
        Fn_pol, Xi_pol, Ms_pol = plscf.pLSCF_Poles(
            Ad, Bn, self.dt, nxseg=nxseg, methodSy=method
        )
        Lab = gen.lab_stab(
            Fn_pol,
            Xi_pol,
            Ms_pol,
            ordmin,
            ordmax,
            step=1,
            err_fn=err_fn,
            err_xi=err_xi,
            err_ms=err_phi,
            max_xi=xi_max,
            mpc_lim=mpc_lim,
            mpd_lim=mpd_lim,
        )

        # Return results
        return self.ResultCls(
            freq=freq,
            Sy=Sy,
            Ad=Ad,
            Bn=Bn,
            Fn_poles=Fn_pol,
            xi_poles=Xi_pol,
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
        Extract the modal parameters at the selected frequencies and order.

        Parameters
        ----------
        sel_freq : List[float]
            A list of frequencies for which the modal parameters are to be estimated.
        order : int or str, optional
            The order for modal parameter estimation or "find_min".
            Default is 'find_min'.
        deltaf : float, optional
            The frequency range around each selected frequency to consider for estimation. Default is 0.05.
        rtol : float, optional
            Relative tolerance for convergence in the iterative estimation process. Default is 1e-2.

        Returns
        -------
        Any
            The results of the modal parameter estimation, typically including estimated frequencies, damping
            ratios, and mode shapes.
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
        Fn_pLSCF, Xi_pLSCF, Phi_pLSCF, order_out = plscf.pLSCF_MPE(
            sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=Lab, rtol=rtol
        )

        # Save results
        self.result.order_out = order_out
        self.result.Fn = Fn_pLSCF
        self.result.Xi = Xi_pLSCF
        self.result.Phi = Phi_pLSCF

    def mpe_fromPlot(
        self,
        freqlim: typing.Optional[tuple[float, float]] = None,
        rtol: float = 5e-2,
    ) -> typing.Any:
        """
        Extract the modal parameters directly from the stabilisation chart.

        Parameters
        ----------
        freqlim : tuple of float, optional
            A tuple specifying the frequency limits (min, max) for the plot. If None, the limits are
            determined automatically. Default is None.
        deltaf : float, optional
            The frequency range around each selected frequency to consider for estimation. Default is 0.05.
        rtol : float, optional
            Relative tolerance for convergence in the iterative estimation process. Default is 1e-2.

        Returns
        -------
        Any
            The results of the modal parameter estimation based on user selection from the plot.
        """
        super().mpe_fromPlot(freqlim=freqlim, rtol=rtol)

        # Save run parameters
        self.run_params.rtol = rtol

        # Get poles
        Fn_pol = self.result.Fn_poles
        Sm_pol = self.result.xi_poles
        Ms_pol = self.result.Phi_poles

        # chiamare plot interattivo
        SFP = SelFromPlot(algo=self, freqlim=freqlim, plot="pLSCF")
        sel_freq = SFP.result[0]
        order = SFP.result[1]

        # e poi estrarre risultati
        Fn_pLSCF, Xi_pLSCF, Phi_pLSCF, order_out = plscf.pLSCF_MPE(
            sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=None, rtol=rtol
        )

        # Save results
        self.result.order_out = order_out
        self.result.Fn = Fn_pLSCF
        self.result.Xi = Xi_pLSCF
        self.result.Phi = Phi_pLSCF

    def plot_STDiag(
        self,
        freqlim: typing.Optional[tuple[float, float]] = None,
        hide_poles: typing.Optional[bool] = True,
    ) -> typing.Any:
        """
        Plot the Stability Diagram.

        Parameters
        ----------
        freqlim : tuple of float, optional
            Frequency limits (min, max) for the stability diagram. If None, limits are determined
            automatically. Default is None.
        hide_poles : bool, optional
            Option to hide the unstable poles in the diagram for clarity. Default is True.

        Returns
        -------
        Any
            A tuple containing the matplotlib figure and axes objects for the stability diagram.
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


# =============================================================================
# MULTI SETUP
# =============================================================================
class pLSCF_MS(pLSCF[pLSCFRunParams, pLSCFResult, typing.Iterable[dict]]):
    """
    A multi-setup extension of the pLSCF class for the poly-reference Least Square Complex Frequency
    (pLSCF) algorithm.


    Parameters
    ----------
    pLSCF : type
        Inherits from the pLSCF class with specified type parameters for pLSCFRunParams, pLSCFResult, and
        Iterable[dict].

    Attributes
    ----------
    RunParamCls : pLSCFRunParams
        Class attribute for run parameters specific to pLSCF algorithm.
    ResultCls : pLSCFResult
        Class attribute for results specific to pLSCF algorithm.
    """

    RunParamCls = pLSCFRunParams
    ResultCls = pLSCFResult

    def run(self) -> pLSCFResult:
        """
        Execute the pLSCF algorithm to perform modal analysis on the provided data.

        This method conducts a frequency domain analysis using the Least Square Complex Frequency method.
        It computes system matrices, identifies poles, and labels them based on stability and other criteria.

        Returns
        -------
        pLSCFResult
            An instance of `pLSCFResult` containing the analysis results, including frequencies,
            system matrices, identified poles, and their labels.
        """

        Y = self.data
        nxseg = self.run_params.nxseg
        method = self.run_params.method_SD
        pov = self.run_params.pov
        sgn_basf = self.run_params.sgn_basf
        step = self.run_params.step
        ordmax = self.run_params.ordmax
        ordmin = self.run_params.ordmin
        err_fn = self.run_params.err_fn
        err_xi = self.run_params.err_xi
        err_phi = self.run_params.err_phi
        xi_max = self.run_params.xi_max
        mpc_lim = self.run_params.mpc_lim
        mpd_lim = self.run_params.mpd_lim
        # self.run_params.df = 1 / dt / nxseg

        freq, Sy = fdd.SD_PreGER(Y, self.fs, nxseg=nxseg, method=method, pov=pov)
        Ad, Bn = plscf.pLSCF(Sy, self.dt, ordmax, sgn_basf=sgn_basf)
        Fn_pol, Xi_pol, Ms_pol = plscf.pLSCF_Poles(
            Ad, Bn, self.dt, nxseg=nxseg, methodSy=method
        )
        Lab = gen.lab_stab(
            Fn_pol,
            Xi_pol,
            Ms_pol,
            ordmin,
            ordmax,
            step,
            err_fn=err_fn,
            err_xi=err_xi,
            err_ms=err_phi,
            max_xi=xi_max,
            mpc_lim=mpc_lim,
            mpd_lim=mpd_lim,
        )

        # Return results
        return self.ResultCls(
            freq=freq,
            Sy=Sy,
            Ad=Ad,
            Bn=Bn,
            Fn_poles=Fn_pol,
            xi_poles=Xi_pol,
            Phi_poles=Ms_pol,
            Lab=Lab,
        )
