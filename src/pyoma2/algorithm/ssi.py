"""STOCHASTIC SUBSPACE IDENTIFICATION (SSI) ALGORITHM"""

import typing

import numpy as np
from pydantic import (  # controlla che i parametri passati siano quelli giusti
    validate_call,
)

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
        # method = self.run_params.method_hank
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
        # qui method deve essere uno nell elseif del file SSI_func (vedi 10 01:00)
        H = SSI_funct.BuildHank(Y, Yref, 1 / self.dt, self.fs, method=self.method)
        # Get state matrix and output matrix
        A, C = SSI_funct.SSI_FAST(H, br, ordmax)
        # Get frequency poles (and damping and mode shapes)
        Fn_pol, Sm_pol, Ms_pol = SSI_funct.SSI_Poles(
            A, C, ordmax, self.dt, step=step
        )
        # Get the labels of the poles
        Lab = SSI_funct.Lab_stab_SSI(
            Fn_pol, Sm_pol, Ms_pol, ordmin, ordmax, step, err_fn, err_xi, err_phi, xi_max
        )

        # FIXME Non serve fare così, basta ritornare la classe result, poi saraà SingleSetup a salvarla
        # Fake result: FIXME return real SSIResult
        return SSIResult(
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
        # Qui è corretto perchè result esiste dopo che si è fatto il run()
        self.result.Fn = Fn_SSI
        self.result.Sm = Xi_SSI
        self.result.Ms = Phi_SSI

    @validate_call
    def mpe_fromPlot(
        self,
        freqlim: typing.Optional[float] = None,
        deltaf: float = 0.05,
        rtol: float = 1e-2,
    ) -> typing.Any:
        super().mpe_fromPlot(freqlim=freqlim, deltaf=deltaf, rtol=rtol)

        Fn_pol = self.result.Fn_pol
        Sm_pol = self.result.Sm_pol
        Ms_pol = self.result.Ms_pol

        # chiamare plot interattivo
        sel_freq, order = SelFromPlot(algo=self, freqlim=freqlim, plot="SSI")

        # e poi estrarre risultati
        Fn_SSI, Xi_SSI, Phi_SSI = SSI_funct.SSI_MPE(
            sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=None, deltaf=deltaf, rtol=rtol
        )

        # Save results
        # Qui è corretto perchè result esiste dopo che si è fatto il run()
        self.result.Fn = Fn_SSI
        self.result.Sm = Xi_SSI
        self.result.Ms = Phi_SSI

    def plot_STDiag(self,
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
            Fn= self.result.Fn_poles,
            Lab= self.result.Lab,
            step= self.run_params.step,
            ordmax= self.run_params.ordmax,
            ordmin= self.run_params.ordmin,
            freqlim=freqlim,
            hide_poles=hide_poles,
            fig=None,
            ax=None,)
        return fig, ax

    def plot_cluster(self,
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
            Fn= self.result.Fn_poles,
            Sm= self.result.xi_poles,
            Lab= self.result.Lab,
            ordmin= self.run_params.ordmin,
            freqlim=freqlim,
            hide_poles=hide_poles,)
        return fig, ax

    def plot_mode_g1(self, *args, **kwargs) -> typing.Any:
        """Tobe implemented, plot for FDD, EFDD, FSDD
        Mode Identification Function (MIF)
        """
        if not self.geometry1:
            raise ValueError("Definde the geometry first")

        if not self.result.Fn:
            raise ValueError("Run algorithm first")
        # argomenti plot mode:
        # modenumb: int # (da 1 a result.Phi.shape[1]+1)

        # fig, ax = 
        # return fig, ax

    def plot_mode_g2(self, *args, **kwargs) -> typing.Any:
        """Tobe implemented, plot for FDD, EFDD, FSDD
        Mode Identification Function (MIF)
        """
        if not self.geometry2:
            raise ValueError("Definde the geometry first")

        if not self.result.Fn:
            raise ValueError("Run algorithm first")
        # argomenti plot mode:
        # modenumb: int # (da 1 a result.Phi.shape[1]+1)

        # fig, ax = 
        # return fig, ax

    def anim_mode(self, *args, **kwargs) -> typing.Any:
        """Tobe implemented, plot for FDD, EFDD, FSDD
        Mode Identification Function (MIF)
        """
        if not self.geometry2:
            raise ValueError("Definde the geometry (method 2) first")

        if not self.result:
            raise ValueError("Run algorithm first")

        # fig, ax = 
        # return fig, ax
    


# =============================================================================
# (REF)COVARIANCE-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSIcov_algo(SSIdat_algo):
    method: typing.Literal["cov_bias", "cov_matmul", "cov_unb"] = "cov_bias"



# =============================================================================
# ------------------------------------------------------------------------------


"""...same for other alghorithms"""
