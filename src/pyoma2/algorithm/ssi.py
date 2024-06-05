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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyoma2.algorithm.data.geometry import Geometry1, Geometry2
from pyoma2.algorithm.data.result import SSIResult

# from .result import BaseResult
from pyoma2.algorithm.data.run_params import SSIRunParams
from pyoma2.functions import (
    Gen_funct,
    SSI_funct,
    plot_funct,
)
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
# (REF)DATA-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSIdat_algo(BaseAlgorithm[SSIRunParams, SSIResult, typing.Iterable[float]]):
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
        H = SSI_funct.BuildHank(Y, Yref, br, self.fs, method=method)
        # Get state matrix and output matrix
        A, C = SSI_funct.SSI_FAST(H, br, ordmax, step)
        # Get frequency poles (and damping and mode shapes)
        Fn_pol, Sm_pol, Ms_pol = SSI_funct.SSI_Poles(A, C, ordmax, self.dt, step=step)
        # Get the labels of the poles
        Lab = Gen_funct.lab_stab(
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
        Fn_SSI, Xi_SSI, Phi_SSI, order_out = SSI_funct.SSI_MPE(
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
        Fn_SSI, Xi_SSI, Phi_SSI, order_out = SSI_funct.SSI_MPE(
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

    def SvalH_plot(
        self,
        iter_n: typing.Optional[int] = None,
    ) -> typing.Any:
        """ """
        if not self.result:
            raise ValueError("Run algorithm first")

        fig, ax = plot_funct.SvalH_plot(
            H=self.result.H, br=self.run_params.br, iter_n=iter_n
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
            scaleF=scaleF,
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
        color: str = "cmap",
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

        if Geo2.cstrn is not None:
            aa = Geo2.cstrn.to_numpy(na_value=0)[:, :]
            aa = np.nan_to_num(aa, copy=True, nan=0.0)
            val = aa @ phi
            ctn_df = pd.DataFrame(
                {"cName": Geo2.cstrn.index, "val": val},
            )

            mapping = dict(zip(df_phi["sName"], df_phi["Phi"]))
            mapping1 = dict(zip(ctn_df["cName"], ctn_df["val"]))
            mapp = dict(mapping, **mapping1)
        else:
            mapp = dict(zip(df_phi["sName"], df_phi["Phi"]))

        # reshape the mode shape dataframe to fit the pts coord
        df_phi_map = Geo2.sens_map.replace(mapp).astype(float)
        # add together coordinates and mode shape displacement
        # newpoints = Geo2.pts_coord.add(df_phi_map * Geo2.sens_sign, fill_value=0)
        newpoints = (
            Geo2.pts_coord.to_numpy() + df_phi_map.to_numpy() * Geo2.sens_sign.to_numpy()
        )
        # extract only the displacement array
        # newpoints = newpoints.to_numpy()[:, :]

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
        if color == "cmap":
            oldpoints = Geo2.pts_coord.to_numpy()[:, :]
            plot_funct.plt_nodes(ax, newpoints, color="cmap", initial_coord=oldpoints)

        else:
            plot_funct.plt_nodes(ax, newpoints, color=color)
        # check for sens_lines
        if Geo2.sens_lines is not None:
            if color == "cmap":
                plot_funct.plt_lines(
                    ax, newpoints, Geo2.sens_lines, color="cmap", initial_coord=oldpoints
                )
            else:
                plot_funct.plt_lines(ax, newpoints, Geo2.sens_lines, color=color)

        if Geo2.sens_surf is not None:
            if color == "cmap":
                plot_funct.plt_surf(
                    ax,
                    newpoints,
                    Geo2.sens_surf,
                    color="cmap",
                    initial_coord=oldpoints,
                    alpha=0.4,
                )
            else:
                plot_funct.plt_surf(ax, newpoints, Geo2.sens_surf, color=color, alpha=0.4)

        # Set ax options
        plot_funct.set_ax_options(
            ax,
            bg_color="w",
            remove_fill=True,
            remove_grid=True,
            remove_axis=True,
            scaleF=scaleF,
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
        logger.debug("Running AniMode SSI...")
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
        logger.debug("...end AniMode SSI...")


# ------------------------------------------------------------------------------
# (REF)COVARIANCE-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
# FIXME ADD REFERENCE
class SSIcov_algo(SSIdat_algo):
    """
    Implements the Covariance-driven Stochastic Subspace Identification (SSI) algorithm
    for single setup experiments.

    This class is an extension of the SSIdat_algo class, adapted for covariance-driven analysis.
    It processes measurement data from a single setup to identify system dynamics and extract
    modal parameters using the SSIcov-ref method.

    Inherits all attributes and methods from SSIdat_algo.

    Attributes
    ----------
    method : str
        The method used in this SSI algorithm, overridden to 'cov_bias', 'cov_mm', or 'cov_unb' for
        covariance-based analysis.

    Methods
    -------
    Inherits all methods from SSIdat_algo with covariance-specific implementations.
    """

    method: typing.Literal["cov_bias", "cov_mm", "cov_unb"] = "cov_bias"


# =============================================================================
# MULTISETUP
# =============================================================================
# (REF)DATA-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSIdat_algo_MS(SSIdat_algo[SSIRunParams, SSIResult, typing.Iterable[dict]]):
    """
    Implements the Data-Driven Stochastic Subspace Identification (SSI) algorithm for multi-setup
    experiments.

    This class extends the SSIdat_algo class to handle data from multiple experimental setups, with
    moving and reference sensors.

    Inherits all attributes and methods from SSIdat_algo, with focus on multi-setup data handling.

    Attributes
    ----------
    Inherits all attributes from SSIdat_algo.

    Methods
    -------
    run() -> SSIResult
        Executes the algorithm for multiple setups and returns the identification results.
    Inherits other methods from SSIdat_algo, applicable to multi-setup scenarios.
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
        A, C = SSI_funct.SSI_MulSet(
            Y, self.fs, br, ordmax, step=1, methodHank=method, method="FAST"
        )

        # Get frequency poles (and damping and mode shapes)
        Fn_pol, Sm_pol, Ms_pol = SSI_funct.SSI_Poles(A, C, ordmax, self.dt, step=step)
        # Get the labels of the poles
        Lab = Gen_funct.lab_stab(
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
class SSIcov_algo_MS(SSIdat_algo_MS):
    """
    Implements the Covariance-Driven Stochastic Subspace Identification (SSI) algorithm
    for multi-setup experiments.

    This class extends SSIdat_algo_MS, focusing on the covariance-driven approach to SSI
    for multiple experimental setups.

    Inherits all attributes and methods from SSIdat_algo_MS, adapted for covariance-driven
    analysis methods.

    Attributes
    ----------
    Inherits all attributes from SSIdat_algo_MS.

    Methods
    -------
    Inherits all methods from SSIdat_algo_MS, adapted for covariance-based analysis.
    """

    method: typing.Literal["cov_bias", "cov_mm", "cov_unb"] = "cov_bias"
