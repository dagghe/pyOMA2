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
    "init_kwargs, algorithms, names, run_first, expected_exception, expected_message",
    [
        # 1. No setups
        (
            {"ref_ind": [], "single_setups": []},  # init_kwargs
            {},  # algorithms
            [],  # names
            False,  # run_first
            ValueError,  # expected_exception
            "You must pass at least two setup",  # expected_message
        ),
        # 2. Only one setup
        (
            {
                "ref_ind": [],
                "single_setups": [SingleSetup(np.zeros((10, 10)), fs=100)],
            },  # init_kwargs
            {0: [FakeAlgorithm(name="one")]},  # algorithms
            ["one"],  # names
            False,  # run_first
            ValueError,  # expected_exception
            "You must pass at least two setup",  # expected_message
        ),
        # 3. Setups with no algorithms
        (
            {
                "ref_ind": [],
                "single_setups": [
                    SingleSetup(np.zeros((10, 10)), fs=100),
                    SingleSetup(np.zeros((10, 10)), fs=100),
                ],
            },  # init_kwargs
            {},  # algorithms
            ["one"],  # names
            False,  # run_first
            ValueError,  # expected_exception
            "You must pass setups with at least one algorithm",  # expected_message
        ),
        # 4. Names len different from nr of algorithms
        (
            {
                "ref_ind": [],
                "single_setups": [
                    SingleSetup(np.zeros((10, 10)), fs=100),
                    SingleSetup(np.zeros((10, 10)), fs=100),
                ],
            },  # init_kwargs
            {
                0: [
                    FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
                ],
                1: [
                    FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
                ],
            },  # algorithms
            ["one", "two", "three"],  # names
            False,  # run_first
            ValueError,  # expected_exception
            "The number of names must match the number of algorithms",  # expected_message
        ),
        # 5. Algorithm in setup not ran
        (
            {
                "ref_ind": [],
                "single_setups": [
                    SingleSetup(np.zeros((10, 10)), fs=100),
                    SingleSetup(np.zeros((10, 10)), fs=100),
                ],
            },  # init_kwargs
            {
                0: [
                    FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
                ],
                1: [
                    FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
                ],
            },  # algorithms
            [
                "one",
            ],  # names
            False,  # run_first
            ValueError,  # expected_exception
            "You must pass Single setups that have already been run",  # expected_message
        ),
        # 6. Setup with less algorithms than expected
        (
            {
                "ref_ind": [],
                "single_setups": [
                    SingleSetup(np.zeros((10, 10)), fs=100),
                    SingleSetup(np.zeros((10, 10)), fs=100),
                ],
            },  # init_kwargs
            {
                0: [
                    FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
                    FakeAlgorithm2(run_params=FakeAlgorithm2.RunParamCls(param1=1)),
                ],
                1: [
                    FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
                ],
            },  # algorithms
            ["one", "two"],
            True,
            ValueError,
            "The algorithms must be consistent between setups",
        ),
        # 7. Setup with different order of algorithms
        (
            {
                "ref_ind": [],
                "single_setups": [
                    SingleSetup(np.zeros((10, 10)), fs=100),
                    SingleSetup(np.zeros((10, 10)), fs=100),
                ],
            },  # init_kwargs
            {
                0: [
                    FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
                    FakeAlgorithm2(run_params=FakeAlgorithm2.RunParamCls(param1=1)),
                ],
                1: [
                    # order is inverted
                    FakeAlgorithm2(run_params=FakeAlgorithm2.RunParamCls(param1=1)),
                    FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
                ],
            },  # algorithms
            ["one", "two"],
            True,
            ValueError,
            "The algorithms must be consistent between setups",
        ),
        # 8. Setup with different order of algorithms
        (
            {
                "ref_ind": [],
                "single_setups": [
                    SingleSetup(np.zeros((10, 10)), fs=100),
                    SingleSetup(np.zeros((10, 10)), fs=100),
                ],
            },  # init_kwargs
            {
                0: [
                    FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
                    FakeAlgorithm2(run_params=FakeAlgorithm2.RunParamCls(param1=1)),
                ],
                1: [
                    # different set of algorithms
                    FakeAlgorithm(run_params=FakeAlgorithm2.RunParamCls(param1=1)),
                    FakeAlgorithm(run_params=FakeAlgorithm.RunParamCls(param1=1)),
                ],
            },  # algorithms
            ["one", "two"],
            True,
            ValueError,
            "The algorithms must be consistent between setups",
        ),
    ],
    ids=[
        "1. No setups",
        "2. Only one setup",
        "3. Setups with no algorithms",
        "4. Names len different from nr of algorithms",
        "5. Algorithm in setup not ran",
        "6. Setup with less algorithms than expected",
        "7. Setup with different order of algorithms",
        "8. Setup with different set of algorithms",
    ],
)
def test_init_multisetup_poser_exc(
    init_kwargs: typing.Dict,
    algorithms: typing.Dict[int, typing.List[BaseAlgorithm]],
    names: typing.List[str],
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
        MultiSetup_PoSER(**init_kwargs, names=names)
    assert expected_message in str(excinfo.value)


def test_init_multisetup_poser(multi_setup_data_fixture) -> None:
    """Test the initialization of MultiSetup_PoSER."""

    # setup multisetup poser initialization
    set1, set2, set3 = multi_setup_data_fixture
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
    msp = MultiSetup_PoSER(
        ref_ind=[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        single_setups=[ss1, ss2, ss3],
        names=["one", "two"],
    )
    assert msp.ref_ind == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    assert msp.setups == [ss1, ss2, ss3]

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
    set1, set2, set3 = multi_setup_data_fixture
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


def test_plot_data_single_column(ss):
    ss.data = ss.data[:, 0]
    ss.plot_data()
    assert False
