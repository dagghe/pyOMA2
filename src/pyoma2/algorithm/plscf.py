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

import matplotlib.pyplot as plt
import pandas as pd

from pyoma2.algorithm.data.geometry import Geometry1, Geometry2
from pyoma2.algorithm.data.result import pLSCFResult

# from .result import BaseResult
from pyoma2.algorithm.data.run_params import pLSCFRunParams
from pyoma2.functions import FDD_funct, Gen_funct, plot_funct, pLSCF_funct
from pyoma2.functions.plot_funct import (
    plt_lines,
    plt_nodes,
    plt_quiver,
    plt_surf,
    set_ax_options,
    set_view,
)
from pyoma2.plot.anim_mode import AniMode

# from .run_params import BaseRunParams
from pyoma2.plot.Sel_from_plot import SelFromPlot

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


# =============================================================================
# SINGLE SETUP
# =============================================================================
class pLSCF_algo(BaseAlgorithm[pLSCFRunParams, pLSCFResult, typing.Iterable[float]]):
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

        freq, Sy = FDD_funct.SD_Est(Y, Y, self.dt, nxseg, method=method, pov=pov)

        Ad, Bn = pLSCF_funct.pLSCF(Sy, self.dt, ordmax, sgn_basf=sgn_basf)
        Fn_pol, Xi_pol, Ms_pol = pLSCF_funct.pLSCF_Poles(
            Ad, Bn, self.dt, nxseg=nxseg, methodSy=method
        )
        Lab = Gen_funct.lab_stab(
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
        Fn_pLSCF, Xi_pLSCF, Phi_pLSCF, order_out = pLSCF_funct.pLSCF_MPE(
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
        Fn_pLSCF, Xi_pLSCF, Phi_pLSCF, order_out = pLSCF_funct.pLSCF_MPE(
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
        fig, ax = plot_funct.Stab_plot(
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

        fig, ax = plot_funct.Cluster_plot(
            Fn=self.result.Fn_poles,
            Sm=self.result.xi_poles,
            Lab=self.result.Lab,
            ordmin=self.run_params.ordmin,
            freqlim=freqlim,
            hide_poles=hide_poles,
        )
        return fig, ax

    def plot_mode_g1(
        self,
        Geo1: Geometry1,
        mode_numb: int,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
    ) -> typing.Any:
        """
        Plots a 3D mode shape for a specified mode number using the Geometry1 object.

        Parameters
        ----------
        Geo1 : Geometry1
            Geometry object containing sensor coordinates and other information.
        mode_numb : int
            Mode number to visualize.
        scaleF : int, optional
            Scale factor for mode shape visualization. Default is 1.
        view : {'3D', 'xy', 'xz', 'yz', 'x', 'y', 'z'}, optional
            View for the 3D plot. Default is '3D'.
        remove_fill : bool, optional
            Whether to remove fill from the plot. Default is True.
        remove_grid : bool, optional
            Whether to remove grid from the plot. Default is True.
        remove_axis : bool, optional
            Whether to remove axis from the plot. Default is True.

        Returns
        -------
        typing.Any
            A tuple containing the matplotlib figure and axes of the mode shape plot.
        """
        if self.result.Fn is None:
            raise ValueError("Run algorithm first")

        # Select the (real) mode shape
        phi = self.result.Phi[:, int(mode_numb - 1)].real
        fn = self.result.Fn[int(mode_numb - 1)]

        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        # set title
        ax.set_title(f"Mode nr. {mode_numb}, $f_n$={fn:.3f}Hz")

        # plot sensors' nodes
        sens_coord = Geo1.sens_coord[["x", "y", "z"]].to_numpy()
        plt_nodes(ax, sens_coord, color="red")

        # plot Mode shape
        plt_quiver(
            ax,
            sens_coord,
            Geo1.sens_dir * phi.reshape(-1, 1),
            scaleF=scaleF,
            #            names=Geo1.sens_names,
        )

        # Check that BG nodes are defined
        if Geo1.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, Geo1.bg_nodes, color="gray", alpha=0.5)
            # Check that BG lines are defined
            if Geo1.bg_lines is not None:
                # if True plot
                plt_lines(ax, Geo1.bg_nodes, Geo1.bg_lines, color="gray", alpha=0.5)
            if Geo1.bg_surf is not None:
                # if True plot
                plt_surf(ax, Geo1.bg_nodes, Geo1.bg_surf, alpha=0.1)

        # check for sens_lines
        if Geo1.sens_lines is not None:
            # if True plot
            plt_lines(ax, sens_coord, Geo1.sens_lines, color="red")

        # Set ax options
        set_ax_options(
            ax,
            bg_color="w",
            remove_fill=remove_fill,
            remove_grid=remove_grid,
            remove_axis=remove_axis,
        )

        # Set view
        set_view(ax, view=view)
        return fig, ax

    def plot_mode_g2(
        self,
        Geo2: Geometry2,
        mode_numb: typing.Optional[int],
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
        *args,
        **kwargs,
    ) -> typing.Any:
        """
        Plots a 3D mode shape for a specified mode number using the Geometry2 object.

        Parameters
        ----------
        Geo2 : Geometry2
            Geometry object containing nodes, sensor information, and additional geometrical data.
        mode_numb : int
            Mode number to visualize.
        scaleF : int, optional
            Scale factor for mode shape visualization. Default is 1.
        view : {'3D', 'xy', 'xz', 'yz', 'x', 'y', 'z'}, optional
            View for the 3D plot. Default is '3D'.
        remove_fill : bool, optional
            Whether to remove fill from the plot. Default is True.
        remove_grid : bool, optional
            Whether to remove grid from the plot. Default is True.
        remove_axis : bool, optional
            Whether to remove axis from the plot. Default is True.
        *args, **kwargs
            Additional arguments for customizations.

        Returns
        -------
        typing.Any
            A tuple containing the matplotlib figure and axes of the mode shape plot.
        """
        if self.result.Fn is None:
            raise ValueError("Run algorithm first")

        # Select the (real) mode shape
        fn = self.result.Fn[int(mode_numb - 1)]
        phi = self.result.Phi[:, int(mode_numb - 1)].real * scaleF
        # create mode shape dataframe
        df_phi = pd.DataFrame(
            {"sName": Geo2.sens_names, "Phi": phi},
        )
        mapping = dict(zip(df_phi["sName"], df_phi["Phi"]))
        # reshape the mode shape dataframe to fit the pts coord
        df_phi_map = Geo2.sens_map.replace(mapping).astype(float)
        # add together coordinates and mode shape displacement
        newpoints = Geo2.pts_coord.add(df_phi_map * Geo2.sens_sign, fill_value=0)
        # extract only the displacement array
        newpoints = newpoints.to_numpy()[:, 1:]

        # create fig and ax
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        ax.set_title(f"Mode nr. {mode_numb}, $f_n$={fn:.3f}Hz")

        # Check that BG nodes are defined
        if Geo2.bg_nodes is not None:
            # if True plot
            plot_funct.plt_nodes(ax, Geo2.bg_nodes, color="gray", alpha=0.5)
            # Check that BG lines are defined
            if Geo2.bg_lines is not None:
                # if True plot
                plot_funct.plt_lines(
                    ax, Geo2.bg_nodes, Geo2.bg_lines, color="gray", alpha=0.5
                )
            if Geo2.bg_surf is not None:
                # if True plot
                plot_funct.plt_surf(ax, Geo2.bg_nodes, Geo2.bg_surf, alpha=0.1)
        # PLOT MODE SHAPE
        plot_funct.plt_nodes(ax, newpoints, color="red")
        # check for sens_lines
        if Geo2.sens_lines is not None:
            # if True plot
            plot_funct.plt_lines(ax, newpoints, Geo2.sens_lines, color="red")

        # Set ax options
        plot_funct.set_ax_options(
            ax,
            bg_color="w",
            remove_fill=remove_fill,
            remove_grid=remove_grid,
            remove_axis=remove_axis,
        )

        # Set view
        plot_funct.set_view(ax, view=view)

        return fig, ax

    def anim_mode_g2(
        self,
        Geo2: Geometry2,
        mode_numb: typing.Optional[int],
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
        saveGIF: bool = False,
        *args,
        **kwargs,
    ) -> typing.Any:
        """
        Creates an animated visualization of a 3D mode shape for a specified mode number using Geometry2.

        Parameters
        ----------
        Geo2 : Geometry2
            Geometry object containing nodes, sensor information, and additional geometrical data.
        mode_numb : int, optional
            Mode number to visualize. If None, no mode is selected.
        scaleF : int, optional
            Scale factor for mode shape animation. Default is 1.
        view : {'3D', 'xy', 'xz', 'yz', 'x', 'y', 'z'}, optional
            View for the 3D animation. Default is '3D'.
        remove_fill : bool, optional
            Whether to remove fill from the animation. Default is True.
        remove_grid : bool, optional
            Whether to remove grid from the animation. Default is True.
        remove_axis : bool, optional
            Whether to remove axis from the animation. Default is True.
        saveGIF : bool, optional
            Whether to save the animation as a GIF file. Default is False.
        *args, **kwargs
            Additional arguments for customization.

        Returns
        -------
        typing.Any
            The animation object or any relevant output, depending on the implementation and provided
            parameters.
        """
        if self.result.Fn is None:
            raise ValueError("Run algorithm first")

        Res = self.result
        logger.debug("Running AniMode...")
        AniMode(
            Geo=Geo2,
            Res=Res,
            mode_numb=mode_numb,
            scaleF=scaleF,
            view=view,
            remove_axis=remove_axis,
            remove_fill=remove_fill,
            remove_grid=remove_grid,
            saveGIF=saveGIF,
        )
        logger.debug("...end AniMode...")


# =============================================================================
# MULTI SETUP
# =============================================================================
class pLSCF_algo_MS(pLSCF_algo[pLSCFRunParams, pLSCFResult, typing.Iterable[dict]]):
    """
    A multi-setup extension of the pLSCF_algo class for the poly-reference Least Square Complex Frequency
    (pLSCF) algorithm.


    Parameters
    ----------
    pLSCF_algo : type
        Inherits from the pLSCF_algo class with specified type parameters for pLSCFRunParams, pLSCFResult, and
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

        freq, Sy = FDD_funct.SD_PreGER(Y, self.fs, nxseg=nxseg, method=method, pov=pov)
        Ad, Bn = pLSCF_funct.pLSCF(Sy, self.dt, ordmax, sgn_basf=sgn_basf)
        Fn_pol, Xi_pol, Ms_pol = pLSCF_funct.pLSCF_Poles(
            Ad, Bn, self.dt, nxseg=nxseg, methodSy=method
        )
        Lab = Gen_funct.lab_stab(
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
