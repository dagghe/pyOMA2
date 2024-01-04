import abc
import typing

from pydantic import (  # controlla che i parametri passati siano quelli giusti
    validate_call,
)

from pyoma2.functions import (FDD_funct, Gen_funct, SSI_funct, plot_funct,
                              pLSCF_funct)


from pyoma2.algorithm.result import BaseResult
# from .result import BaseResult

from pyoma2.algorithm.run_params import BaseRunParams
# from .run_params import BaseRunParams

from pyoma2.algorithm.Sel_from_plot import SelFromPlot

from pyoma2.OMA import SingleSetup

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
    def run(self, data, fs) -> typing.Any:
        """Run main algorithm using self.run_params"""

    @abc.abstractmethod
    def mpe(self, *args, **kwargs) -> typing.Any:
        """Return modes"""
        if not self.result:
            raise ValueError("Run algorithm first")

    @abc.abstractmethod
    def mpe_fromPlot(self, *args, **kwargs) -> typing.Any:
        """Select peaks"""
        if not self.result:
            raise ValueError("Run algorithm first")

# METODI PER PLOT "STATICI" DOVE SI AGGIUNGONO? 
# ALLA CLASSE BASE o A QUELLA SPECIFICA?
# =============================================================================
# BASIC FREQUENCY DOMAIN DECOMPOSITION
class FDD_algo(BaseAlgorithm):
    def run(self, data, fs) -> typing.Any:
        print(self.run_params)
        Y = data.T
        dt = 1 / fs
        nxseg = self.run_params.nxseg
        method = self.run_params.method_SD
        # self.run_params.df = 1 / dt / nxseg

        freq, Sy = FDD_funct.SD_Est(Y, Y, dt, nxseg, method=method)
        Sval, Svec = FDD_funct.SD_svalsvec(Sy)

        # Save results
        self.result.freq = freq
        self.result.Sy = Sy
        self.result.Sval = Sval
        self.result.Svec = Svec

    @validate_call
    def mpe(self, sel_freq: float, DF: float = 0.1) -> typing.Any:
        super().mpe(sel_freq=sel_freq, DF=DF)

        self.run_params.sel_freq = sel_freq
        self.run_params.DF = DF
        Sy = self.result.Sy
        freq = self.result.freq

        # Get Modal Parameters
        Fn_FDD, Phi_FDD = FDD_funct.FDD_MPE(Sy, freq, sel_freq, DF=DF)

        # Save results
        self.result.Fn = Fn_FDD
        self.result.Phi = Phi_FDD

    @validate_call
    def mpe_fromPlot(
        self, freqlim: typing.Optional[float] = None, DF: float = 0.1
    ) -> typing.Any:
        super().mpe_fromPlot(freqlim=freqlim)

        Sy = self.result.Sy
        freq = self.result.freq

        self.run_params.DF = DF

        # chiamare plot interattivo
        sel_freq = SelFromPlot(self, freqlim=freqlim, plot="FDD")

        # e poi estrarre risultati
        Fn_FDD, Phi_FDD = FDD_funct.FDD_MPE(Sy, freq, sel_freq, DF=DF)

        # Save results
        self.result.Fn = Fn_FDD
        self.result.Phi = Phi_FDD


# =============================================================================
# ENHANCED FREQUENCY DOMAIN DECOMPOSITION EFDD
# FREQUENCY SPATIAL DOMAIN DECOMPOSITION FSDD
class EFDD_algo(BaseAlgorithm):
    def run(self, data, fs) -> typing.Any:
        print(self.run_params)
        Y = data.T
        dt = 1 / fs
        nxseg = self.run_params.nxseg
        method = self.run_params.method_SD

        self.run_params.df = 1 / dt / nxseg  # frequency resolution for the spectrum

        freq, Sy = FDD_funct.SD_Est(Y, Y, dt, nxseg, method=method)
        Sval, Svec = FDD_funct.SD_svalsvec(Sy)

        # Save results
        self.result.freq = freq
        self.result.Sy = Sy
        self.result.Sval = Sval
        self.result.Svec = Svec

    @validate_call
    def mpe(
        self,
        sel_freq: float,
        method: str,
        methodSy: str = "cor",
        DF1: float = 0.1,
        DF2: float = 1.0,
        cm: int = 1,
        MAClim: float = 0.85,
        sppk: int = 3,
        npmax: int = 20,
    ) -> typing.Any:

        super().mpe(
            sel_freq=sel_freq,
            method=method,
            methodSy=methodSy,
            DF1=DF1,
            DF2=DF2,
            cm=cm,
            MAClim=MAClim,
            sppk=sppk,
            npmax=npmax,
        )

        dt = self.run_params.dt # ATTENZIONE
        Sy = self.result.Sy
        freq = self.result.freq

        Fn_FDD, Xi_FDD, Phi_FDD, forPlot = FDD_funct.EFDD_MPE(
            Sy,
            freq,
            dt,
            sel_freq,
            methodSy,
            method="EFDD",
            DF1=0.1,
            DF2=1.0,
            cm=1,
            MAClim=0.85,
            sppk=3,
            npmax=20,
        )

        # Save results
        self.result.Fn = Fn_FDD
        self.result.Xi = Xi_FDD
        self.result.Phi = Phi_FDD
        self.result.forPlot = forPlot

    @validate_call
    def mpe_fromPlot(
        self, 
        method: str,
        methodSy: str = "cor",
        DF1: float = 0.1,
        DF2: float = 1.0,
        cm: int = 1,
        MAClim: float = 0.85,
        sppk: int = 3,
        npmax: int = 20,
        freqlim: typing.Optional[float] = None,
    ) -> typing.Any:
        super().mpe_fromPlot(
            method=method,
            methodSy=methodSy,
            DF1=DF1,
            DF2=DF2,
            cm=cm,
            MAClim=MAClim,
            sppk=sppk,
            npmax=npmax,
            freqlim=freqlim
            )

        dt = self.run_params.dt # ATTENZIONE
        Sy = self.result.Sy
        freq = self.result.freq

        # chiamare plot interattivo
        sel_freq = SelFromPlot(self, freqlim=freqlim, plot="FDD")

        # e poi estrarre risultati
        Fn_FDD, Xi_FDD, Phi_FDD, forPlot = FDD_funct.EFDD_MPE(
            Sy,
            freq,
            dt,
            sel_freq,
            methodSy,
            method="EFDD",
            DF1=0.1,
            DF2=1.0,
            cm=1,
            MAClim=0.85,
            sppk=3,
            npmax=20,
        )

        # Save results
        self.result.Fn = Fn_FDD
        self.result.Xi = Xi_FDD
        self.result.Phi = Phi_FDD
        self.result.forPlot = forPlot


# =============================================================================
# (REF)DATA-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSIdat_algo(BaseAlgorithm):
    def run(self, data, fs) -> typing.Any:
        print(self.run_params)
        Y = data.T
        dt = 1 / fs
        br = self.run_params.br
        # method = self.run_params.method_hank
        ordmin = self.run_params.ordmin
        ordmax = self.run_params.ordmax
        step = self.run_params.step
        err_fn = self.run_params.err_fn
        err_xi = self.run_params.err_xi
        err_phi = self.run_params.err_phi
        xi_max = self.run_params.xi_max

        if self.ref_ind is not None:
            ref_ind = self.run_params.ref_ind
            Yref = Y[ref_ind,:]
        else:
            Yref = Y

        # Build Hankel matrix
        H = SSI_funct.BuildHank(Y, Yref, 1 / dt, fs, method="dat")
        # Get state matrix and output matrix
        A, C = SSI_funct.SSI_FAST(H, br, ordmax)
        # Get frequency poles (and damping and mode shapes)
        Fn_pol, Sm_pol, Ms_pol = SSI_funct.SSI_funct.SSI_Poles(
            A, C, ordmax, dt, step=step
        )
        # Get the labels of the poles
        Lab = SSI_funct.Lab_stab_SSI(
            Fn_pol, Sm_pol, Ms_pol, ordmin, ordmax, step, err_fn, err_xi, err_phi, xi_max
        )

        # Save results
        self.result.H = H
        self.result.A = A
        self.result.C = C
        self.result.Fn_pol = Fn_pol
        self.result.Sm_pol = Sm_pol
        self.result.Ms_pol = Ms_pol

    @validate_call
    def mpe(self, 
            sel_freq: float, 
            order: str = "find_min"
        ) -> typing.Any:
        super().mpe(sel_freq=sel_freq, 
                    order=order)

        Fn_pol = self.result.Fn_pol
        Sm_pol = self.result.Sm_pol
        Ms_pol = self.result.Ms_pol

        Fn_SSI, Xi_SSI, Phi_SSI = SSI_funct.SSI_MPE(
            sel_freq, 
            Fn_pol, 
            Sm_pol, 
            Ms_pol, 
            order, 
            Lab=None, 
            deltaf=0.05, 
            rtol=1e-2
        )

        # Save results
        self.result.Fn = Fn_SSI
        self.result.Sm = Xi_SSI
        self.result.Ms = Phi_SSI

    @validate_call
    def mpe_fromPlot(
        self,
        freqlim: typing.Optional[float] = None,
    ) -> typing.Any:
        super().mpe_fromPlot(
            freqlim=freqlim,
        )
        
        Fn_pol = self.result.Fn_pol
        Sm_pol = self.result.Sm_pol
        Ms_pol = self.result.Ms_pol

        # chiamare plot interattivo
        sel_freq, order = SelFromPlot(self, freqlim=freqlim, plot="SSI")

        # e poi estrarre risultati
        Fn_SSI, Xi_SSI, Phi_SSI = SSI_funct.SSI_MPE(
            sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=None, deltaf=0.05, rtol=1e-2
        )
        
        # Save results
        self.result.Fn = Fn_SSI
        self.result.Sm = Xi_SSI
        self.result.Ms = Phi_SSI

# =============================================================================
# (REF)COVARIANCE-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSIcov_algo(BaseAlgorithm):
    def run(self, data, fs) -> typing.Any:
        print(self.run_params)
        Y = data.T
        dt = 1 / fs
        br = self.run_params.br
        method = self.run_params.method_hank
        ordmin = self.run_params.ordmin
        ordmax = self.run_params.ordmax
        step = self.run_params.step
        err_fn = self.run_params.err_fn
        err_xi = self.run_params.err_xi
        err_phi = self.run_params.err_phi
        xi_max = self.run_params.xi_max

        if self.ref_ind is not None:
            ref_ind = self.run_params.ref_ind
            Yref = Y[ref_ind,:]
        else:
            Yref = Y

        # Build Hankel matrix
        H = SSI_funct.BuildHank(Y, Yref, 1 / dt, fs, method=method)
        # Get state matrix and output matrix
        A, C = SSI_funct.SSI_FAST(H, br, ordmax)
        # Get frequency poles (and damping and mode shapes)
        Fn_pol, Sm_pol, Ms_pol = SSI_funct.SSI_Poles(A, C, ordmax, dt, step=step)
        # Get the labels of the poles
        Lab = SSI_funct.Lab_stab_SSI(
            Fn_pol, Sm_pol, Ms_pol, ordmin, ordmax, step, err_fn, err_xi, err_phi, xi_max
        )

        # Save results
        self.result.H = H
        self.result.A = A
        self.result.C = C
        self.result.Fn_pol = Fn_pol
        self.result.Sm_pol = Sm_pol
        self.result.Ms_pol = Ms_pol

    @validate_call
    def mpe(self, 
            sel_freq: float, 
            order: str = "find_min"
        ) -> typing.Any:
        super().mpe(sel_freq=sel_freq, 
                    order=order)

        Fn_pol = self.result.Fn_pol
        Sm_pol = self.result.Sm_pol
        Ms_pol = self.result.Ms_pol

        Fn_SSI, Xi_SSI, Phi_SSI = SSI_funct.SSI_MPE(
            sel_freq, 
            Fn_pol, 
            Sm_pol, 
            Ms_pol, 
            order, 
            Lab=None, 
            deltaf=0.05, 
            rtol=1e-2
        )

        # Save results
        self.result.Fn = Fn_SSI
        self.result.Sm = Xi_SSI
        self.result.Ms = Phi_SSI

    @validate_call
    def mpe_fromPlot(
        self,
        freqlim: typing.Optional[float] = None,
    ) -> typing.Any:
        super().mpe_fromPlot(
            freqlim=freqlim,
        )
        
        Fn_pol = self.result.Fn_pol
        Sm_pol = self.result.Sm_pol
        Ms_pol = self.result.Ms_pol

        # chiamare plot interattivo
        sel_freq, order = SelFromPlot(self, freqlim=freqlim, plot="SSI")

        # e poi estrarre risultati
        Fn_SSI, Xi_SSI, Phi_SSI = SSI_funct.SSI_MPE(
            sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=None, deltaf=0.05, rtol=1e-2
        )
        
        # Save results
        self.result.Fn = Fn_SSI
        self.result.Sm = Xi_SSI
        self.result.Ms = Phi_SSI

# =============================================================================
# ------------------------------------------------------------------------------


"""...same for other alghorithms"""
