"""Negative static typing fixtures for algorithm/setup generics.

This module is expected to produce mypy errors. It is not collected by pytest.
"""

from __future__ import annotations

import numpy as np

from pyoma2.algorithms import EFDD, FDD, FDD_MS, SSI, SSI_MS
from pyoma2.algorithms.data.mpe_params import EFDDMPEParams
from pyoma2.algorithms.data.result import EFDDResult
from pyoma2.algorithms.data.run_params import (
    EFDDRunParams,
    FDDRunParams,
    SSIRunParams,
    pLSCFRunParams,
)
from pyoma2.setup import MultiSetup_PreGER, SingleSetup


def invalid_type_argument_on_fdd() -> None:
    # Public FDD is fully specialized and is not a generic class
    _ = FDD[str]()


def forged_fdd_specialization() -> None:
    # FDD[...] must not invent EFDD param/result types; factories stay FDD's
    alias = FDD[EFDDRunParams, EFDDMPEParams, EFDDResult](run_params=EFDDRunParams())
    _ = alias


def invalid_type_argument_on_ssi() -> None:
    # Public SSI locks data to SingleSetupData and is not indexable as a generic
    _ = SSI[str]()


def wrong_run_params_type() -> None:
    # Bare FDD means defaults (FDDRunParams, ...); SSI run params are rejected
    fdd: FDD = FDD(run_params=SSIRunParams(br=5, ordmax=20))
    _ = fdd


def unannotated_wrong_run_params() -> None:
    # Incompatible specialization must fail even without a target annotation.
    # Assign to real names: mypy skips constructor checks on dummy ``_``.
    fdd = FDD(run_params=SSIRunParams(br=5, ordmax=20))
    fdd2 = FDD(run_params=pLSCFRunParams(ordmax=20))
    ssi = SSI(run_params=FDDRunParams())
    _ = (fdd, fdd2, ssi)


def single_setup_rejects_multi_algorithm() -> None:
    setup = SingleSetup(data=np.zeros((50, 2)), fs=100.0)
    setup.add_algorithms(FDD_MS())
    setup.add_algorithms(SSI_MS())


def multi_setup_rejects_single_algorithm() -> None:
    setup = MultiSetup_PreGER(
        fs=100.0,
        ref_ind=[[0], [0]],
        datasets=[np.zeros((50, 2)), np.zeros((50, 3))],
    )
    setup.add_algorithms(FDD())
    setup.add_algorithms(EFDD())
    setup.add_algorithms(SSI())


def assign_wrong_result_type() -> None:
    fdd = FDD(run_params=FDDRunParams())
    # result is FDDResult | None, not str
    fdd.result = "not a result"


def efdd_is_not_a_typed_fdd() -> None:
    # EFDD.mpe does not accept FDD's DF keyword; not a typed FDD
    def use_fdd(x: FDD) -> None:
        x.mpe([1.0], DF=0.2)

    use_fdd(EFDD())
