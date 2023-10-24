import abc
import typing

from pydantic import (
    validate_call,
)  # controlla che i parametri passati siano quelli giusti

import Gen_funct, FDD_funct, SSI_funct, pLSCF_funct, plot_funct

from .result import BaseResult
from .run_params import BaseRunParams


class BaseAlgorithm(abc.ABC):
    """Abstract class for Modal Analysis algorithms"""

    def __init__(
        self,
        run_params: BaseRunParams,
        name: typing.Optional[str] = None,
    ):
        self.run_params = run_params
        self.name = name or self.__class__.__name__
        self.result: BaseResult

    @abc.abstractmethod
    def run(self) -> typing.Any:
        """Run main algorithm using self.run_params"""

    @abc.abstractmethod
    def mpe(self, *args, **kwargs) -> typing.Any:
        """Return modes"""
        if not self.result:
            raise ValueError("Run algorithm first")

    # Si puo avere un metodo per i due plot diversi? (MIF e stab diag)
    @abc.abstractmethod
    def mpe_fromPlot(self, *args, **kwargs) -> typing.Any:
        """Select peaks"""
        if not self.result:
            raise ValueError("Run algorithm first")

    # Questo mi sa che non serve
    @abc.abstractmethod
    def mod_ex(self, *args, **kwargs) -> typing.Any:
        if not self.result:
            raise ValueError("Run algorithm first")

#------------------------------------------------------------------------------
# BASIC FREQUENCY DOMAIN DECOMPOSITION
class FDD_algo(BaseAlgorithm):
    def run(self) -> typing.Any:
        print(self.run_params)
        Y = self.data.T
        dt = self.dt
        nxseg = self.run_params.nxseg
        method = self.run_params.method_SD
        self.run_params.df = 1/dt/nxseg # frequency resolution for the spectrum

        freq, Sy = SD_Est(Y, Y, dt, nxseg, method=method)
        Sval, Svec = SD_svalsvec(Sy)
        # come si salvano i risultati ora?
        # self.result.freq = freq
        # self.result.Sy = Sy
        # self.result.Sval = Sval
        # self.result.Svec = Svec
       

    @validate_call
    def mpe(self, sel_freq: float, DF: float = 0.1) -> typing.Any:
        super().mpe(sel_freq=sel_freq, DF=DF)
        self.run_params.sel_freq = sel_freq
        self.run_params.DF = DF

        # Get Modal Parameters
        Fn_FDD, Phi_FDD = FDD_MPE(Sy, freq, sel_freq, DF=DF)

        # e poi salvare?
        # self.result.Fn_FDD = Fn_FDD
        # self.result.Phi_FDD = Phi_FDD


    @validate_call
    def mpe_fromPlot(
        self, freqlim: typing.Optional[float] = None, DF: float = 0.1
    ) -> typing.Any:
        super().mpe_fromPlot(freqlim=freqlim)
        self.run_params.DF = DF
        # chiamare plot interattivo
        # _ = Sel_from_plot.SelFromPlot(self, freqlim=freqlim, plot="FDD")

        # e poi estrarre risultati
        Fn_FDD, Phi_FDD = FDD_MPE(Sy, freq, sel_freq, DF=DF)

        # e poi salvare?
        # self.result.Fn_FDD = Fn_FDD
        # self.result.Phi_FDD = Phi_FDD

#    @validate_call
#    def mod_ex(self, ndf: int = 5) -> typing.Any:
#        super().mod_ex(ndf=ndf)

#------------------------------------------------------------------------------
# ENHANCED FREQUENCY DOMAIN DECOMPOSITION
class EFDD_algo(BaseAlgorithm):
    def run(self) -> typing.Any:
        print(self.run_params)
        Y = self.data.T
        dt = self.dt
        nxseg = self.run_params.nxseg
        method = self.run_params.method_SD
        self.run_params.df = 1/dt/nxseg # frequency resolution for the spectrum

        freq, Sy = SD_Est(Y, Y, dt, nxseg, method=method)
        Sval, Svec = SD_svalsvec(Sy)
        # come si salvano i risultati ora?
        # self.result.freq = freq
        # self.result.Sval = Sval

    @validate_call
    def mpe(self, sel_freq: float, method: str, methodSy: str = "cor", 
        DF1: float = 0.1, DF2: float = 1., cm: int = 1, MAClim: float = 0.85,
        sppk: int = 3, npmax: int = 20,
    ) -> typing.Any:

        super().mpe(sel_freq=sel_freq,method=method, methodSy=methodSy, DF1=DF1,
            DF2=DF2, cm=cm, MAClim=MAClim, sppk=sppk, npmax=npmax)

        Fn_FDD, Xi_FDD, Phi_FDD, forPlot = EFDD_MPE(Sy, freq, dt, sel_freq, methodSy,
    method="EFDD", DF1=0.1, DF2=1., cm=1, MAClim=0.85, sppk=3, npmax=20, )

        # e poi salvare?
        # self.result.Fn_FDD = Fn_FDD
        # self.result.Xi_FDD = Xi_FDD
        # self.result.Phi_FDD = Phi_FDD
        # self.result.forPlot = forPlot    

    @validate_call
    def mpe_fromPlot(
        self, freqlim: typing.Optional[float] = None, DF: float = 0.1
    ) -> typing.Any:
        super().mpe_fromPlot(freqlim=freqlim)
        # chiamare plot interattivo
        # _ = Sel_from_plot.SelFromPlot(self, freqlim=freqlim, plot="FDD")

        # e poi estrarre risultati
        Fn_FDD, Xi_FDD, Phi_FDD, forPlot = EFDD_MPE(Sy, freq, dt, sel_freq, methodSy,
    method="EFDD", DF1=0.1, DF2=1., cm=1, MAClim=0.85, sppk=3, npmax=20, )

        # e poi salvare?
        # self.result.Fn_FDD = Fn_FDD
        # self.result.Xi_FDD = Xi_FDD
        # self.result.Phi_FDD = Phi_FDD
        # self.result.forPlot = forPlot 

#    @validate_call
#    def mod_ex(self, ndf: int = 5) -> typing.Any:
#        super().mod_ex(ndf=ndf)

#------------------------------------------------------------------------------
# FREQUENCY SPATIAL DOMAIN DECOMPOSITION
class FSDD_algo(BaseAlgorithm):
    def run(self) -> typing.Any:
        print(self.run_params)
        Y = self.data.T
        dt = self.dt
        nxseg = self.run_params.nxseg
        method = self.run_params.method_SD
        df = 1/dt/nxseg

        freq, Sy = SD_Est(Y, Y, dt, nxseg, method=method)
        Sval, Svec = SD_svalsvec(Sy)
        # come si salvano i risultati ora?
        # self.result.

    @validate_call
    def mpe(self, sel_freq: float, DF: float = 0.1) -> typing.Any:
        super().mpe(sel_freq=sel_freq, DF=DF)
        Fn_FDD, Xi_FDD, Phi_FDD, forPlot = EFDD_MPE(Sy, freq, dt, sel_freq, methodSy,
    method="FSDD", DF1=0.1, DF2=1., cm=1, MAClim=0.85, sppk=3, npmax=20, )

        # e poi salvare?

    @validate_call
    def mpe_fromPlot(
        self, freqlim: typing.Optional[float] = None, DF: float = 0.1
    ) -> typing.Any:
        super().mpe_fromPlot(freqlim=freqlim)
        # chiamare plot interattivo
        # _ = Sel_from_plot.SelFromPlot(self, freqlim=freqlim, plot="FDD")

        # e poi estrarre risultati
        Fn_FDD, Xi_FDD, Phi_FDD, forPlot = EFDD_MPE(Sy, freq, dt, sel_freq, methodSy,
    method="EFDD", DF1=0.1, DF2=1., cm=1, MAClim=0.85, sppk=3, npmax=20, )

        # e poi salvare?

#    @validate_call
#    def mod_ex(self, ndf: int = 5) -> typing.Any:
#        super().mod_ex(ndf=ndf)

#------------------------------------------------------------------------------
# (REF)DATA-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSIdat_algo(BaseAlgorithm):
    def run(self) -> typing.Any:
        print(self.run_params)
        Y = self.data.T
        dt = self.dt
        p = self.run_params.br
        #method = self.run_params.method_hank
        ordmin = self.run_params.ordmin
        ordmax = self.run_params.ordmax
        step = self.run_params.step
        err_fn = self.run_params.err_fn
        err_xi = self.run_params.err_xi
        err_phi = self.run_params.err_phi
        xi_max = self.run_params.xi_max


        # if self.ref_ind is not None:

        # Build Hankel matrix
        H = BuildHank(Y, Y, 1/dt, fs, method="dat")
        # Get state matrix and output matrix
        A, C = SSI_FAST(H, br, ordmax)
        # Get frequency poles (and damping and mode shapes)
        Fn_pol, Sm_pol, Ms_pol = SSI_Poles(A, C, ordmax, dt, step=step)
        # Get the labels of the poles
        Lab = Lab_stab_SSI(Fn_pol, Sm_pol, Ms_pol, ordmin, ordmax, step, 
                            err_fn, err_xi, err_phi, xi_max)
        # come si salvano i risultati ora?

    @validate_call
    def mpe(self, sel_freq: float, order: str = "find_min") -> typing.Any:
        super().mpe(sel_freq=sel_freq, order=order)
        Fn_SSI, Xi_SSI, Phi_SSI = SSI_MPE(
    sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=None, deltaf=0.05, rtol=1e-2
    )


    @validate_call
    def mpe_fromPlot(
        self,
        freqlim: typing.Optional[float] = None,
        # ma non li ho gia definiti questi?
        ordmin: int = 0,
        ordmax: typing.Optional[int] = None,
        #method: str = "1",
    ) -> typing.Any:
        super().mpe_fromPlot(
            freqlim=freqlim,
            # stessa cosa qui
            ordmin=ordmin,
            ordmax=ordmax,
            #method=method,
        )
        # chiamare plot interattivo
        # _ = Sel_from_plot.SelFromPlot(self, freqlim=freqlim, plot="SSI")

        # e poi estrarre risultati
        Fn_SSI, Xi_SSI, Phi_SSI = SSI_MPE(
    sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=None, deltaf=0.05, rtol=1e-2
    )
        # e poi salvare?

#    @validate_call
#    def mod_ex(self, *args, **kwargs) -> typing.Any:
#        super().mod_ex(*args, **kwargs)

#------------------------------------------------------------------------------
# (REF)COVARIANCE-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSIcov_algo(BaseAlgorithm):
    def run(self) -> typing.Any:
        print(self.run_params)
        Y = self.data.T
        dt = self.dt
        p = self.run_params.br
        method = self.run_params.method_hank
        ordmin = self.run_params.ordmin
        ordmax = self.run_params.ordmax
        step = self.run_params.step
        err_fn = self.run_params.err_fn
        err_xi = self.run_params.err_xi
        err_phi = self.run_params.err_phi
        xi_max = self.run_params.xi_max


        # if self.ref_ind is not None:

        # Build Hankel matrix
        H = BuildHank(Y, Y, 1/dt, fs, method=method)
        # Get state matrix and output matrix
        A, C = SSI_FAST(H, br, ordmax)
        # Get frequency poles (and damping and mode shapes)
        Fn_pol, Sm_pol, Ms_pol = SSI_Poles(A, C, ordmax, dt, step=step)
        # Get the labels of the poles
        Lab = Lab_stab_SSI(Fn_pol, Sm_pol, Ms_pol, ordmin, ordmax, step, 
                            err_fn, err_xi, err_phi, xi_max)
        # come si salvano i risultati ora?

    @validate_call
    def mpe(self, sel_freq: float, order: str = "find_min") -> typing.Any:
        super().mpe(sel_freq=sel_freq, order=order)
        Fn_SSI, Xi_SSI, Phi_SSI = SSI_MPE(
    sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=None, deltaf=0.05, rtol=1e-2
    )


    @validate_call
    def mpe_fromPlot(
        self,
        freqlim: typing.Optional[float] = None,
        # ma non li ho gia definiti questi?
        ordmin: int = 0,
        ordmax: typing.Optional[int] = None,
        #method: str = "1",
    ) -> typing.Any:
        super().mpe_fromPlot(
            freqlim=freqlim,
            # stessa cosa qui
            ordmin=ordmin,
            ordmax=ordmax,
            #method=method,
        )
        # chiamare plot interattivo
        # _ = Sel_from_plot.SelFromPlot(self, freqlim=freqlim, plot="SSI")

        # e poi estrarre risultati
        Fn_SSI, Xi_SSI, Phi_SSI = SSI_MPE(
    sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=None, deltaf=0.05, rtol=1e-2
    )
        # e poi salvare?

#    @validate_call
#    def mod_ex(self, *args, **kwargs) -> typing.Any:
#        super().mod_ex(*args, **kwargs)

#------------------------------------------------------------------------------



"""...same for other alghorithms"""
