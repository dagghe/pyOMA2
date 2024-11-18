from __future__ import annotations

import shutil
import sys
import typing
import unittest.mock
from unittest.mock import MagicMock

if typing.TYPE_CHECKING:
    from pyoma2.setup import BaseSetup, MultiSetup_PoSER, MultiSetup_PreGER, SingleSetup

import numpy as np
import pytest
from pyoma2.algorithms import SSIcov
from pyoma2.support.utils.sample_data import (
    SAMPLE_DATA_DEFAULT_LOCAL_DIR,
    get_sample_data,
)

from .factory import FakeAlgorithm, FakeAlgorithm2, FakeResult, FakeRunParams


@pytest.fixture(scope="session")
def fake_algorithm_no_param_fixture() -> typing.Generator[FakeAlgorithm, None, None]:
    """Fixture for FakeAlgorithm without parameters."""
    yield FakeAlgorithm()


@pytest.fixture(scope="session")
def fake_ran_algorithm() -> typing.Generator[FakeAlgorithm, None, None]:
    """Fixture for FakeAlgorithm that has been run."""
    fa = FakeAlgorithm()
    # set result to mock that the algorithm has been run
    fa.result = FakeResult()
    yield fa


@pytest.fixture(scope="session")
def fake_ran_algorithm2() -> typing.Generator[FakeAlgorithm2, None, None]:
    """Fixture for FakeAlgorithm2 that has been run."""
    fa = FakeAlgorithm2()
    # set result to mock that the algorithm has been run
    fa.result = FakeResult()
    yield fa


@pytest.fixture(scope="session")
def fake_algorithm2_no_param_fixture() -> typing.Generator[FakeAlgorithm2, None, None]:
    """Fixture for FakeAlgorithm without parameters."""
    yield FakeAlgorithm2()


@pytest.fixture(scope="session")
def fake_algorithm_with_param_fixture() -> typing.Generator[FakeAlgorithm, None, None]:
    """Fixture for FakeAlgorithm with parameters."""
    yield FakeAlgorithm(run_params=FakeRunParams())


@pytest.fixture(scope="session")
def fake_single_setup_fixture_no_param() -> typing.Generator[SingleSetup, None, None]:
    """Fixture for SingleSetup without parameters."""
    from pyoma2.setup import SingleSetup

    ss = SingleSetup(data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), fs=1000)
    ss.add_algorithms(FakeAlgorithm())
    yield ss


@pytest.fixture(scope="session")
def fake_single_setup_fixture_with_param() -> typing.Generator[SingleSetup, None, None]:
    """Fixture for SingleSetup with parameters."""
    from pyoma2.setup import SingleSetup

    ss = SingleSetup(data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), fs=1000)
    ss.add_algorithms(FakeAlgorithm(run_params=FakeRunParams()))
    yield ss


@pytest.fixture(scope="session")
def single_setup_data_fixture():
    """Fixture for SingleSetup data"""
    # mock with repeatable random the palisaden/Palisaden_dataset.npy data
    np.random.seed(42)
    data = np.random.rand(540_000, 6)

    # import geometry files
    # Names of the channels
    Names = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"]

    yield (
        data,
        Names,
    )


@pytest.fixture(scope="function", name="bs")
def base_setup_fixture(
    single_setup_data_fixture,
) -> typing.Generator[BaseSetup, None, None]:
    """
    Fixture for BaseSetup with parameters.

    it has 2 algorithms:
        FakeAlgorithm with name "fake_1"
        FakeAlgorithm2 with name "fake_2"
    """
    from pyoma2.setup import BaseSetup

    data, *_ = single_setup_data_fixture
    ss = BaseSetup()
    ss.data = data
    ss.fs = 100
    yield ss


@pytest.fixture(scope="function", name="ss")
def single_setup_fixture(
    single_setup_data_fixture,
) -> typing.Generator[SingleSetup, None, None]:
    """Fixture for SingleSetup with parameters."""
    from pyoma2.setup import SingleSetup

    data, *_ = single_setup_data_fixture
    ss = SingleSetup(data=data, fs=100)
    yield ss


@pytest.fixture(scope="session")
def multi_setup_data_fixture():
    """Fixture for MultiSetup data"""
    set1 = np.load(get_sample_data(filename="set1.npy", folder="3SL"), allow_pickle=True)
    set2 = np.load(get_sample_data(filename="set2.npy", folder="3SL"), allow_pickle=True)
    set3 = np.load(get_sample_data(filename="set3.npy", folder="3SL"), allow_pickle=True)
    yield (
        set1,
        set2,
        set3,
    )


@pytest.fixture(scope="function", name="ms_poser")
def multi_setup_poser_fixture(
    multi_setup_data_fixture,
) -> typing.Generator[MultiSetup_PoSER, None, None]:
    from pyoma2.setup import MultiSetup_PoSER, SingleSetup

    """Fixture for MultiSetup Poser with parameters."""
    set1, set2, set3 = multi_setup_data_fixture
    ss1 = SingleSetup(set1, fs=100)
    ss2 = SingleSetup(set2, fs=100)
    ss3 = SingleSetup(set3, fs=100)
    # Detrend and decimate
    ss1.decimate_data(q=2)
    ss2.decimate_data(q=2)
    ss3.decimate_data(q=2)

    # Initialise the algorithms for setup 1
    ssicov1 = SSIcov(name="SSIcov1", method="cov", br=50, ordmax=80)
    # Add algorithms to the class
    ss1.add_algorithms(ssicov1)
    ss1.run_all()

    # Initialise the algorithms for setup 2
    ssicov2 = SSIcov(name="SSIcov2", method="cov", br=50, ordmax=80)
    ss2.add_algorithms(ssicov2)
    ss2.run_all()

    # Initialise the algorithms for setup 2
    ssicov3 = SSIcov(name="SSIcov3", method="cov", br=50, ordmax=80)
    ss3.add_algorithms(ssicov3)
    ss3.run_all()

    # run mpe
    ss1.mpe(
        "SSIcov1",
        sel_freq=[2.63, 2.69, 3.43, 8.29, 8.42, 10.62, 14.00, 14.09, 17.57],
        order=50,
    )
    ss2.mpe(
        "SSIcov2",
        sel_freq=[2.63, 2.69, 3.43, 8.29, 8.42, 10.62, 14.00, 14.09, 17.57],
        order=40,
    )
    ss3.mpe(
        "SSIcov3",
        sel_freq=[2.63, 2.69, 3.43, 8.29, 8.42, 10.62, 14.00, 14.09, 17.57],
        order=40,
    )
    # reference indices
    ref_ind = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    # Creating Multi setup
    msp = MultiSetup_PoSER(
        ref_ind=ref_ind, single_setups=[ss1, ss2, ss3], names=["the_coolest_algo"]
    )
    yield msp


@pytest.fixture(scope="function", name="ms_preger")
def multi_setup_preger_fixture(
    multi_setup_data_fixture,
) -> typing.Generator[MultiSetup_PreGER, None, None]:
    """Fixture for MultiSetup Poser with parameters."""
    from pyoma2.setup import MultiSetup_PreGER

    set1, set2, set3 = multi_setup_data_fixture
    data = [set1, set2, set3]
    ref_ind = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    # Creating Multi setup
    msp = MultiSetup_PreGER(fs=100, ref_ind=ref_ind, datasets=data)
    yield msp


# Mock imports for GUI and plotting
sys.modules["vtkmodules"] = unittest.mock.Mock()
sys.modules["pyvista"] = unittest.mock.Mock()
sys.modules["pyvistaqt"] = unittest.mock.Mock()


@pytest.fixture(autouse=True)
def mock_imports():
    with unittest.mock.patch(
        "matplotlib.pyplot.figure"
    ) as mock_figure, unittest.mock.patch(
        "matplotlib.pyplot.show"
    ) as mock_show, unittest.mock.patch(
        "matplotlib.pyplot.subplots"
    ) as subplots, unittest.mock.patch(
        "pyoma2.support.sel_from_plot.NavigationToolbar2Tk"
    ), unittest.mock.patch(
        "pyoma2.support.sel_from_plot.FigureCanvasTkAgg"
    ), unittest.mock.patch(
        "src.pyoma2.functions.plot.plt.tight_layout"
    ), unittest.mock.patch("tkinter.Tk"), unittest.mock.patch("tkinter.Menu"):
        """
        Mocks the imports for the tests.
        All mocked imports area bout GUI and plotting.
        """

        def subplots_side_effect(nrows=1, ncols=1, *args, **kwargs):
            """
            Mock for matplotlib.pyplot.subplots.
            Returns a tuple with a MagicMock for the figure and a 2-dimensional
            array of MagicMocks for the axes.
            """
            if nrows == 1 and ncols == 1:
                return (MagicMock(), MagicMock())
            else:
                if nrows == 1 or ncols == 1:
                    size = max(
                        nrows, ncols
                    )  # Determine the size of the 1-dimensional array
                    mock_array = np.empty(
                        size, dtype=object
                    )  # Create a 1-dimensional array
                    for i in range(size):
                        mock_array[i] = MagicMock()
                else:
                    mock_array = np.empty((nrows, ncols), dtype=object)
                    for i in range(nrows):
                        for j in range(ncols):
                            mock_array[i, j] = MagicMock()

                return (MagicMock(), mock_array)

        subplots.side_effect = subplots_side_effect
        yield mock_figure, mock_show, subplots


@pytest.fixture(scope="session", autouse=True)
def cleanup_sample_data_dir():
    """
    Fixture to delete the SAMPLE_DATA_DEFAULT_LOCAL_DIR after all tests have run.
    """
    yield
    # Cleanup code to delete the directory
    if SAMPLE_DATA_DEFAULT_LOCAL_DIR.exists() and SAMPLE_DATA_DEFAULT_LOCAL_DIR.is_dir():
        shutil.rmtree(SAMPLE_DATA_DEFAULT_LOCAL_DIR)
        print(f"Deleted directory: {SAMPLE_DATA_DEFAULT_LOCAL_DIR}")
