"""Positive static typing fixtures for algorithm/setup generics.

This module must type-check cleanly under mypy. It is not collected by pytest.
"""

from __future__ import annotations

import numpy as np

from pyoma2.algorithms import (
    EFDD,
    EFDD_MS,
    FDD,
    FDD_MS,
    SSI,
    SSI_MS,
    FDDAlgorithm,
    pLSCF,
    pLSCF_MS,
)
from pyoma2.algorithms.base import BaseAlgorithm, MultiSetupData, SingleSetupData
from pyoma2.algorithms.data.mpe_params import EFDDMPEParams, FDDMPEParams, SSIMPEParams
from pyoma2.algorithms.data.result import EFDDResult, FDDResult, SSIResult
from pyoma2.algorithms.data.run_params import EFDDRunParams, FDDRunParams, SSIRunParams
from pyoma2.algorithms.fdd import FSDD
from pyoma2.setup import MultiSetup_PreGER, SingleSetup


def _single_data() -> SingleSetupData:
    return np.zeros((100, 4), dtype=float)


def _multi_datasets() -> list[np.ndarray]:
    return [
        np.zeros((100, 4), dtype=float),
        np.zeros((100, 5), dtype=float),
    ]


def construct_algorithms() -> None:
    """Public constructors remain usable without explicit type arguments."""
    fdd: FDD = FDD()
    efdd: EFDD = EFDD()
    fsdd: FSDD = FSDD()
    ssi: SSI = SSI()
    plscf: pLSCF = pLSCF()
    fdd_ms: FDD_MS = FDD_MS()
    efdd_ms: EFDD_MS = EFDD_MS()
    ssi_ms: SSI_MS = SSI_MS()
    plscf_ms: pLSCF_MS = pLSCF_MS()
    _ = (
        fdd,
        efdd,
        fsdd,
        ssi,
        plscf,
        fdd_ms,
        efdd_ms,
        ssi_ms,
        plscf_ms,
    )


def specialized_param_and_result_types() -> None:
    """Concrete classes expose the specialized run/mpe/result types."""
    fdd = FDD(run_params=FDDRunParams())
    efdd = EFDD(run_params=EFDDRunParams())
    ssi = SSI(run_params=SSIRunParams(br=5, ordmax=20))

    fdd_rp: FDDRunParams | None = fdd.run_params
    fdd_mpe: FDDMPEParams | None = fdd.mpe_params
    fdd_res: FDDResult | None = fdd.result

    efdd_rp: EFDDRunParams | None = efdd.run_params
    efdd_mpe: EFDDMPEParams | None = efdd.mpe_params
    efdd_res: EFDDResult | None = efdd.result

    ssi_rp: SSIRunParams | None = ssi.run_params
    ssi_mpe: SSIMPEParams | None = ssi.mpe_params
    ssi_res: SSIResult | None = ssi.result

    _ = (fdd_rp, fdd_mpe, fdd_res, efdd_rp, efdd_mpe, efdd_res, ssi_rp, ssi_mpe, ssi_res)


def single_setup_accepts_single_algorithms() -> None:
    setup = SingleSetup(data=_single_data(), fs=100.0)
    setup.add_algorithms(
        FDD(name="fdd"),
        EFDD(name="efdd"),
        SSI(name="ssi"),
        pLSCF(name="plscf"),
    )
    algo = setup["fdd"]
    data: SingleSetupData | None = algo.data
    _ = data


def multi_setup_accepts_multi_algorithms() -> None:
    setup = MultiSetup_PreGER(
        fs=100.0,
        ref_ind=[[0, 1], [0, 1]],
        datasets=_multi_datasets(),
    )
    setup.add_algorithms(
        FDD_MS(name="fdd_ms"),
        EFDD_MS(name="efdd_ms"),
        SSI_MS(name="ssi_ms"),
        pLSCF_MS(name="plscf_ms"),
    )
    algo = setup["fdd_ms"]
    data: MultiSetupData | None = algo.data
    _ = data


def run_param_cls_factory() -> None:
    """Class-level RunParamCls factories return the specialized run-param type."""
    fdd_params: FDDRunParams = FDD.RunParamCls(nxseg=512)
    efdd_params: EFDDRunParams = EFDD.RunParamCls(nxseg=1024)
    ssi_params: SSIRunParams = SSI.RunParamCls(br=5, ordmax=20)
    fdd = FDD(run_params=fdd_params)
    _ = (fdd_params, efdd_params, ssi_params, fdd)


def run_returns_specialized_result() -> None:
    """``run()`` is generic in T_Result, so EFDD().run() is EFDDResult."""
    fdd_res: FDDResult = FDD().run()
    efdd_res: EFDDResult = EFDD().run()
    ssi_res: SSIResult = SSI().run()
    _ = (fdd_res, efdd_res, ssi_res)


class _FactoryContractAlgo(
    BaseAlgorithm[FDDRunParams, FDDMPEParams, FDDResult, SingleSetupData]
):
    """Downstream subclass using the documented plain factory assignment."""

    RunParamCls = FDDRunParams
    MPEParamCls = FDDMPEParams
    ResultCls = FDDResult

    def run(self) -> FDDResult:
        return FDDResult()

    def mpe(self, *args: object, **kwargs: object) -> None:
        return None

    def mpe_from_plot(self, *args: object, **kwargs: object) -> None:
        return None


def fdd_family_shares_spectrum_protocol() -> None:
    """FDD-family classes share ``run`` / ``plot_CMIF``, not ``mpe(DF=...)``."""

    def use_spectrum(algorithm: FDDAlgorithm) -> None:
        result: FDDResult = algorithm.run()
        fig_ax = algorithm.plot_CMIF(nSv="all")
        _ = (result, fig_ax)

    use_spectrum(FDD())
    use_spectrum(EFDD())
    use_spectrum(FSDD())
    use_spectrum(FDD_MS())
    use_spectrum(EFDD_MS())


def factory_assignment_without_classvar() -> None:
    """Subclasses assign factories as plain attributes, not ClassVar."""
    algo = _FactoryContractAlgo()
    params: FDDRunParams = _FactoryContractAlgo.RunParamCls(nxseg=128)
    result_cls: type[FDDResult] = _FactoryContractAlgo.ResultCls
    _ = (algo, params, result_cls)
