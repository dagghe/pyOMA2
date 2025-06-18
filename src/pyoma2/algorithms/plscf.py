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

from pyoma2.algorithms.data.mpe_params import pLSCFMPEParams
from pyoma2.algorithms.data.result import pLSCFResult
from pyoma2.algorithms.data.run_params import pLSCFRunParams
from pyoma2.functions import fdd, gen, plot, plscf
from pyoma2.support.sel_from_plot import SelFromPlot

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


# =============================================================================
# SINGLE SETUP
# =============================================================================
class pLSCF(
    BaseAlgorithm[pLSCFRunParams, pLSCFMPEParams, pLSCFResult, typing.Iterable[float]]
):
    """
    Implementation of the poly-reference Least Square Complex Frequency (pLSCF) algorithm for modal analysis.

    This class inherits from `BaseAlgorithm` and specializes in handling modal analysis computations and
    visualizations based on the pLSCF method. It provides methods to run the analysis, extract modal parameter
    estimation (mpe), plot stability diagrams, cluster diagrams, mode shapes, and animations of mode shapes.

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
    MPEParamCls = pLSCFMPEParams
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
        # sgn_basf = self.run_params.sgn_basf
        ordmax = self.run_params.ordmax
        ordmin = self.run_params.ordmin
        sc = self.run_params.sc
        hc = self.run_params.hc

        if method == "per":
            sgn_basf = -1
        elif method == "cor":
            sgn_basf = +1

        freq, Sy = fdd.SD_est(Y, Y, self.dt, nxseg, method=method, pov=pov)

        Ad, Bn = plscf.pLSCF(Sy, self.dt, ordmax, sgn_basf=sgn_basf)

        Fns, Xis, Phis, Lambds = plscf.pLSCF_poles(
            Ad, Bn, self.dt, nxseg=nxseg, methodSy=method
        )

        # Apply HARD CRITERIA
        # hc_conj = hc.get("conj")
        hc_xi_max = hc["xi_max"]
        hc_mpc_lim = hc["mpc_lim"]
        hc_mpd_lim = hc["mpd_lim"]

        # HC - presence of complex conjugate
        # if hc_conj:
        Lambds, mask1 = gen.HC_conj(Lambds)
        lista = [Fns, Xis, Phis]
        Fns, Xis, Phis = gen.applymask(lista, mask1, Phis.shape[2])

        # HC - damping
        Xis, mask2 = gen.HC_damp(Xis, hc_xi_max)
        lista = [Fns, Phis]
        Fns, Phis = gen.applymask(lista, mask2, Phis.shape[2])

        # HC - MPC and MPD
        if hc_mpc_lim is not None:
            mask3 = gen.HC_MPC(Phis, hc_mpc_lim)
            lista = [Fns, Xis, Phis, Lambds]
            Fns, Xis, Phis, Lambds = gen.applymask(lista, mask3, Phis.shape[2])
        if hc_mpd_lim is not None:
            mask4 = gen.HC_MPD(Phis, hc_mpd_lim)
            lista = [Fns, Xis, Phis, Lambds]
            Fns, Xis, Phis, Lambds = gen.applymask(lista, mask4, Phis.shape[2])

        # Apply SOFT CRITERIA
        # Get the labels of the poles
        Lab = gen.SC_apply(
            Fns,
            Xis,
            Phis,
            ordmin,
            ordmax - 1,
            1,
            sc["err_fn"],
            sc["err_xi"],
            sc["err_phi"],
        )

        # Return results
        return self.ResultCls(
            freq=freq,
            Sy=Sy,
            Ad=Ad,
            Bn=Bn,
            Fn_poles=Fns,
            Xi_poles=Xis,
            Phi_poles=Phis,
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
        self.mpe_params.sel_freq = sel_freq
        self.mpe_params.order_in = order
        self.mpe_params.rtol = rtol

        # Get poles
        Fn_pol = self.result.Fn_poles
        Sm_pol = self.result.Xi_poles
        Ms_pol = self.result.Phi_poles
        Lab = self.result.Lab

        # Extract modal results
        Fn_pLSCF, Xi_pLSCF, Phi_pLSCF, order_out = plscf.pLSCF_mpe(
            sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=Lab, rtol=rtol
        )

        # Save results
        self.result.order_out = order_out
        self.result.Fn = Fn_pLSCF
        self.result.Xi = Xi_pLSCF
        self.result.Phi = Phi_pLSCF

    def mpe_from_plot(
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
        super().mpe_from_plot(freqlim=freqlim, rtol=rtol)

        # Save run parameters
        self.mpe_params.rtol = rtol

        # Get poles
        Fn_pol = self.result.Fn_poles
        Sm_pol = self.result.Xi_poles
        Ms_pol = self.result.Phi_poles

        # chiamare plot interattivo
        SFP = SelFromPlot(algo=self, freqlim=freqlim, plot="pLSCF")
        sel_freq = SFP.result[0]
        order = SFP.result[1]

        # e poi estrarre risultati
        Fn_pLSCF, Xi_pLSCF, Phi_pLSCF, order_out = plscf.pLSCF_mpe(
            sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=None, rtol=rtol
        )

        # Save results
        self.result.order_out = order_out
        self.result.Fn = Fn_pLSCF
        self.result.Xi = Xi_pLSCF
        self.result.Phi = Phi_pLSCF

    def plot_stab(
        self,
        freqlim: typing.Optional[tuple[float, float]] = None,
        hide_poles: typing.Optional[bool] = True,
        color_scheme: typing.Literal[
            "default", "classic", "high_contrast", "viridis"
        ] = "default",
    ) -> typing.Any:
        """
        Plot the Stability Diagram for the pLSCF analysis.

        The stability diagram helps visualize the stability of poles across different model orders.
        It can be used to identify stable poles, which correspond to physical modes.

        Parameters
        ----------
        freqlim : tuple of float, optional
            Frequency limits (min, max) for the stability diagram. If None, limits are determined
            automatically. Default is None.
        hide_poles : bool, optional
            Option to hide the unstable poles in the diagram for clarity. Default is True.
        color_scheme : typing.Literal["default", "classic", "high_contrast", "viridis"], optional
            Color scheme for stable/unstable poles. Options: 'default', 'classic',
            'high_contrast', 'viridis'.

        Returns
        -------
        Any
            A tuple containing the matplotlib figure and axes objects for the stability diagram.
        """
        fig, ax = plot.stab_plot(
            Fn=self.result.Fn_poles,
            Lab=self.result.Lab,
            step=1,
            ordmax=self.run_params.ordmax,
            ordmin=self.run_params.ordmin,
            freqlim=freqlim,
            hide_poles=hide_poles,
            fig=None,
            ax=None,
            color_scheme=color_scheme,
        )
        return fig, ax

    def plot_freqvsdamp(
        self,
        freqlim: typing.Optional[tuple[float, float]] = None,
        hide_poles: typing.Optional[bool] = True,
        color_scheme: typing.Literal[
            "default", "classic", "high_contrast", "viridis"
        ] = "default",
    ) -> typing.Any:
        """
        Plot the frequency-damping cluster diagram for the identified modal parameters.

        The cluster diagram visualizes the distribution of identified modal frequencies and their
        corresponding damping ratios. It helps identify clusters of stable modes.

        Parameters
        ----------
        freqlim : tuple of float, optional
            Frequency limits for the plot. If None, limits are determined automatically. Default is None.
        hide_poles : bool, optional
            Option to hide poles in the plot for clarity. Default is True.
        color_scheme : typing.Literal["default", "classic", "high_contrast", "viridis"], optional
            Color scheme for stable/unstable poles. Options: 'default', 'classic',
            'high_contrast', 'viridis'.

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
            freqlim=freqlim,
            hide_poles=hide_poles,
            color_scheme=color_scheme,
        )
        return fig, ax


# =============================================================================
# MULTI SETUP
# =============================================================================
class pLSCF_MS(pLSCF[pLSCFRunParams, pLSCFMPEParams, pLSCFResult, typing.Iterable[dict]]):
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
    MPEParamCls = pLSCFMPEParams
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
        # sgn_basf = self.run_params.sgn_basf
        # step = self.run_params.step
        ordmax = self.run_params.ordmax
        ordmin = self.run_params.ordmin
        sc = self.run_params.sc
        hc = self.run_params.hc

        if method == "per":
            sgn_basf = -1
        elif method == "cor":
            sgn_basf = +1

        freq, Sy = fdd.SD_PreGER(Y, self.fs, nxseg=nxseg, method=method, pov=pov)

        Ad, Bn = plscf.pLSCF(Sy, self.dt, ordmax, sgn_basf=sgn_basf)

        Fns, Xis, Phis, Lambds = plscf.pLSCF_poles(
            Ad, Bn, self.dt, nxseg=nxseg, methodSy=method
        )

        # Apply HARD CRITERIA
        hc_xi_max = hc["xi_max"]
        hc_mpc_lim = hc["mpc_lim"]
        hc_mpd_lim = hc["mpd_lim"]

        # HC - presence of complex conjugate
        # if hc_conj:
        Lambds, mask1 = gen.HC_conj(Lambds)
        lista = [Fns, Xis, Phis]
        Fns, Xis, Phis = gen.applymask(lista, mask1, Phis.shape[2])

        # HC - damping
        Xis, mask2 = gen.HC_damp(Xis, hc_xi_max)
        lista = [Fns, Phis]
        Fns, Phis = gen.applymask(lista, mask2, Phis.shape[2])

        # HC - MPC and MPD
        if hc_mpc_lim is not None:
            mask3 = gen.HC_MPC(Phis, hc_mpc_lim)
            lista = [Fns, Xis, Phis, Lambds]
            Fns, Xis, Phis, Lambds = gen.applymask(lista, mask3, Phis.shape[2])
        if hc_mpd_lim is not None:
            mask4 = gen.HC_MPD(Phis, hc_mpd_lim)
            lista = [Fns, Xis, Phis, Lambds]
            Fns, Xis, Phis, Lambds = gen.applymask(lista, mask4, Phis.shape[2])

        # Apply SOFT CRITERIA
        # Get the labels of the poles
        Lab = gen.SC_apply(
            Fns,
            Xis,
            Phis,
            ordmin,
            ordmax - 1,
            1,
            sc["err_fn"],
            sc["err_xi"],
            sc["err_phi"],
        )

        # Return results
        return self.ResultCls(
            freq=freq,
            Sy=Sy,
            Ad=Ad,
            Bn=Bn,
            Fn_poles=Fns,
            Xi_poles=Xis,
            Phi_poles=Phis,
            Lab=Lab,
        )
