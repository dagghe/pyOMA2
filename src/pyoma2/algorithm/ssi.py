"""STOCHASTIC SUBSPACE IDENTIFICATION (SSI) ALGORITHM"""

import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import (  # controlla che i parametri passati siano quelli giusti
    validate_call,
)

from pyoma2.algorithm.data.geometry import Geometry1, Geometry2
from pyoma2.algorithm.data.result import SSIResult

# from .result import BaseResult
from pyoma2.algorithm.data.run_params import SSIRunParams
from pyoma2.functions import (  # noqa: F401
    FDD_funct,
    Gen_funct,
    SSI_funct,
    plot_funct,
    pLSCF_funct,
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


# =============================================================================
# (REF)DATA-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSIdat_algo(BaseAlgorithm[SSIRunParams, SSIResult]):
    RunParamCls = SSIRunParams
    ResultCls = SSIResult
    method: typing.Literal["dat"] = "dat"

    def run(self) -> SSIResult:
        super()._pre_run()
        print(self.run_params)
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

        if self.run_params.ref_ind is not None:
            ref_ind = self.run_params.ref_ind
            Yref = Y[ref_ind, :]
        else:
            Yref = Y

        # Build Hankel matrix
        H = SSI_funct.BuildHank(Y, Yref, br, self.fs, method=method)
        # Get state matrix and output matrix
        A, C = SSI_funct.SSI_FAST(H, br, ordmax)
        # Get frequency poles (and damping and mode shapes)
        Fn_pol, Sm_pol, Ms_pol = SSI_funct.SSI_Poles(A, C, ordmax, self.dt, step=step)
        # Get the labels of the poles
        Lab = SSI_funct.Lab_stab_SSI(
            Fn_pol, Sm_pol, Ms_pol, ordmin, ordmax, step, err_fn, err_xi, err_phi, xi_max
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

    @validate_call
    def mpe(
        self,
        sel_freq: float,
        order: str = "find_min",
        deltaf: float = 0.05,
        rtol: float = 1e-2,
    ) -> typing.Any:
        super().mpe(sel_freq=sel_freq, order=order, deltaf=deltaf, rtol=rtol)

        Fn_pol = self.result.Fn_pol
        Sm_pol = self.result.Sm_pol
        Ms_pol = self.result.Ms_pol
        Lab = self.result.Lab

        Fn_SSI, Xi_SSI, Phi_SSI = SSI_funct.SSI_MPE(
            sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=Lab, deltaf=deltaf, rtol=rtol
        )

        # Save results
        self.result.Fn = Fn_SSI
        self.result.Xi = Xi_SSI
        self.result.Phi = Phi_SSI

    @validate_call
    def mpe_fromPlot(
        self,
        freqlim: typing.Optional[float] = None,
        deltaf: float = 0.05,
        rtol: float = 1e-2,
    ) -> typing.Any:
        super().mpe_fromPlot(freqlim=freqlim, deltaf=deltaf, rtol=rtol)

        Fn_pol = self.result.Fn_poles
        Sm_pol = self.result.xi_poles
        Ms_pol = self.result.Phi_poles

        # chiamare plot interattivo
        SFP = SelFromPlot(algo=self, freqlim=freqlim, plot="SSI")
        sel_freq = SFP.result[0]
        order = SFP.result[1][0]

        # e poi estrarre risultati
        Fn_SSI, Xi_SSI, Phi_SSI = SSI_funct.SSI_MPE(
            sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=None, deltaf=deltaf, rtol=rtol
        )

        # Save results
        # Qui è corretto perchè result esiste dopo che si è fatto il run()
        self.result.Fn = Fn_SSI
        self.result.Xi = Xi_SSI
        self.result.Phi = Phi_SSI

    def plot_STDiag(
        self,
        freqlim: typing.Optional[float] = None,
        hide_poles: typing.Optional[bool] = True,
        # # da testare (per aggiungere CMIF_plot su ax2=ax1.twin())
        # plotSval: typing.Optional[False,True] = False,
        # nSv: typing.Optional[int] = "all",
        # nxseg: typing.Optional[int] = 1024,
        # method_SD : typing.Literal["cor", "per"] = "cor",
        # pov: typing.Optional[float] = 0.5,
    ) -> typing.Any:
        """Tobe implemented, plot for SSIdat, SSIcov
        Stability Diagram
        """

        fig, ax = plot_funct.Stab_SSI_plot(
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
        freqlim: typing.Optional[float] = None,
        hide_poles: typing.Optional[bool] = True,
    ) -> typing.Any:
        """Tobe implemented, plot for FDD, EFDD, FSDD
        Mode Identification Function (MIF)
        """
        if not self.result:
            raise ValueError("Run algorithm first")

        # step= self.run_params.step,
        # ordmax= self.run_params.ordmax,
        fig, ax = plot_funct.Cluster_SSI_plot(
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
        remove_fill: True | False = True,
        remove_grid: True | False = True,
        remove_axis: True | False = True,
    ) -> typing.Any:
        """Tobe implemented, plot for FDD, EFDD, FSDD
        Mode Identification Function (MIF)
        """

        if self.result.Fn is None:
            raise ValueError("Run algorithm first")

        # Select the (real) mode shape
        phi = self.result.Phi[:, int(mode_numb - 1)].real
        fn = self.result.Fn[int(mode_numb - 1)]

        fig = plt.figure(figsize=(10, 10), tight_layout=True)
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
            names=Geo1.sens_names,
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
        remove_fill: True | False = True,
        remove_grid: True | False = True,
        remove_axis: True | False = True,
        *args,
        **kwargs,
    ) -> typing.Any:
        """Tobe implemented, plot for FDD, EFDD, FSDD
        Mode Identification Function (MIF)
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

        ax.set_title(f"Mode nr. {self.mode_numb}, $f_n$={fn:.3f}Hz")

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
        remove_fill: True | False = True,
        remove_grid: True | False = True,
        remove_axis: True | False = True,
        *args,
        **kwargs,
    ) -> typing.Any:
        """Tobe implemented, plot for FDD, EFDD, FSDD
        Mode Identification Function (MIF)
        """
        if self.result.Fn is None:
            raise ValueError("Run algorithm first")

        Res = self.result
        print("Running AniMode SSI...")
        AniMode(
            Geo=Geo2,
            Res=Res,
            mode_numb=mode_numb,
            scaleF=scaleF,
            view=view,
            remove_axis=remove_axis,
            remove_fill=remove_fill,
            remove_grid=remove_grid,
        )
        print("...end AniMode SSI...")


# =============================================================================
# (REF)COVARIANCE-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSIcov_algo(SSIdat_algo):
    method: typing.Literal["cov_bias", "cov_mm", "cov_unb"] = "cov_bias"
