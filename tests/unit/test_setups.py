import typing

import numpy as np
import pytest

from src.pyoma2.algorithms import BaseAlgorithm
from src.pyoma2.setup import MultiSetup_PoSER, MultiSetup_PreGER, SingleSetup

from ..factory import FakeAlgorithm, FakeAlgorithm2


def test_init_single_setup(ss: SingleSetup) -> None:
    """Test the initialization of SingleSetup."""
    assert ss.dt == 1 / ss.fs
    assert ss.Nch == ss.data.shape[1]
    assert ss.Ndat == ss.data.shape[0]
    assert ss.dt * ss.Ndat == ss.T
    assert ss.algorithms == {}


@pytest.mark.parametrize(
    "init_kwargs, algorithms, run_first, expected_exception, expected_message",
    [
        (
            {"ref_ind": [], "single_setups": []},
            {},
            False,
            ValueError,
            "You must pass at least one setup",
        ),
        (
            {"ref_ind": [], "single_setups": [SingleSetup(np.zeros((10, 10)), fs=100)]},
            {},
            False,
            ValueError,
            "You must pass setups with at least one algorithm",
        ),
        (
            {"ref_ind": [], "single_setups": [SingleSetup(np.zeros((10, 10)), fs=100)]},
            {0: [FakeAlgorithm(name="one"), FakeAlgorithm(name="two")]},
            False,
            ValueError,
            "You must pass distinct algorithms for setup 1. Duplicates: ",
        ),
        (
            {"ref_ind": [], "single_setups": [SingleSetup(np.zeros((10, 10)), fs=100)]},
            {0: [FakeAlgorithm(name="one")]},
            False,
            ValueError,
            "You must pass Single setups that have already been run",
        ),
        (
            {
                "ref_ind": [],
                "single_setups": [
                    SingleSetup(np.zeros((10, 10)), fs=100),
                    SingleSetup(np.zeros((10, 10)), fs=100),
                ],
            },
            {
                0: [
                    FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
                    FakeAlgorithm2(run_params=FakeAlgorithm2.RunParamCls(param1=1)),
                ],
                1: [
                    FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
                ],
            },
            True,
            ValueError,
            "You must pass all algorithms for setup ",
        ),
    ],
    ids=[
        "no setup",
        "setups with no algorithm",
        "same setup with duplicate algorithms types",
        "algorithm in setup not ran",
        "setup with less algorithms than expected",
    ],
)
def test_init_multisetup_poser_exc(
    init_kwargs: typing.Dict,
    algorithms: typing.Dict[int, typing.List[BaseAlgorithm]],
    run_first: bool,
    expected_exception: Exception,
    expected_message: str,
) -> None:
    """Test the failure of initialization of MultiSetup_PoSER."""

    # Add algorithms to the setups
    for i, ss in enumerate(init_kwargs["single_setups"]):
        if i in algorithms:
            ss.add_algorithms(*algorithms[i])
        if run_first:
            ss.run_all()

    # Test the exception
    with pytest.raises(expected_exception) as excinfo:
        MultiSetup_PoSER(**init_kwargs)
    assert expected_message in str(excinfo.value)


def test_init_multisetup_poser(multi_setup_data_fixture) -> None:
    """Test the initialization of MultiSetup_PoSER."""

    # setup multisetup poser initialization
    set1, set2, set3, *_ = multi_setup_data_fixture
    ss1 = SingleSetup(set1, fs=100)
    ss2 = SingleSetup(set2, fs=100)
    ss3 = SingleSetup(set3, fs=100)
    ss1.add_algorithms(
        FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1), name="fa_1"),
        FakeAlgorithm2(run_params=FakeAlgorithm2.RunParamCls(param1=1), name="fa2_1"),
    )
    ss2.add_algorithms(
        FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
        FakeAlgorithm2(run_params=FakeAlgorithm2.RunParamCls(param1=1)),
    )
    ss3.add_algorithms(
        FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
        FakeAlgorithm2(run_params=FakeAlgorithm2.RunParamCls(param1=1)),
    )
    ss1.run_all()
    ss2.run_all()
    ss3.run_all()

    # initialize MultiSetup_PoSER
    msp = MultiSetup_PoSER(ref_ind=[0, 1, 2], single_setups=[ss1, ss2, ss3])
    assert msp.ref_ind == [0, 1, 2]
    assert msp.setups == [ss1, ss2, ss3]

    # access hidden attribute __alg_ref
    assert msp._MultiSetup_PoSER__alg_ref == {
        alg.__class__: alg.name for alg in msp.setups[0].algorithms.values()
    }

    # set setups after initialization
    with pytest.raises(AttributeError) as excinfo:
        msp.setups = []
    assert "Cannot set setups after initialization" in str(excinfo.value)

    # access result before merging
    with pytest.raises(ValueError) as excinfo:
        _ = msp.result
    assert "You must run merge_results() first" in str(excinfo.value)


def test_init_multisetup_preger(multi_setup_data_fixture) -> None:
    """Test the initialization of MultiSetup_PreGER."""

    # setup multisetup poser initialization
    set1, set2, set3, *_ = multi_setup_data_fixture
    # list of datasets and reference indices
    data = [set1, set2, set3]
    ref_ind = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

    # initialize MultiSetup_PreGER
    msp = MultiSetup_PreGER(fs=100, ref_ind=ref_ind, datasets=data)

    assert msp.dt == 1 / msp.fs
    assert msp.Nsetup == len(ref_ind)

    assert len(msp.data) == len(data)

    assert all(([k for k in d] == ["ref", "mov"] for d in msp.data))
    assert all(([isinstance(v, np.ndarray) for v in d.values()] for d in msp.data))
    assert msp.Ts == [600.0, 600.0, 600.0]
