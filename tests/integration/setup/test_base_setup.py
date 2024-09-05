import numpy as np
import pytest
from pyoma2.setup.base import BaseSetup

from tests.factory import FakeAlgorithm, FakeResult, FakeRunParams


@pytest.fixture
def base_setup():
    """Fixture for BaseSetup class."""
    setup = BaseSetup()
    setup.data = np.random.rand(1000, 3)
    setup.fs = 100
    return setup


def test_add_algorithms(base_setup):
    """Test the add_algorithms method."""
    alg1 = FakeAlgorithm(name="alg1")
    alg2 = FakeAlgorithm(name="alg2")
    base_setup.add_algorithms(alg1, alg2)

    assert "alg1" in base_setup.algorithms
    assert "alg2" in base_setup.algorithms
    assert base_setup.algorithms["alg1"].data is not None
    assert base_setup.algorithms["alg2"].fs == 100


def test_run_all(base_setup):
    """Test the run_all method."""
    alg1 = FakeAlgorithm(name="alg1", run_params=FakeRunParams(param1=2, param2="test"))
    alg2 = FakeAlgorithm(name="alg2", run_params=FakeRunParams(param1=3, param2="test"))
    base_setup.add_algorithms(alg1, alg2)

    base_setup.run_all()

    assert isinstance(base_setup.algorithms["alg1"].result, FakeResult)
    assert isinstance(base_setup.algorithms["alg2"].result, FakeResult)


def test_run_by_name(base_setup):
    """Test the run_by_name method."""
    alg = FakeAlgorithm(
        name="test_alg", run_params=FakeRunParams(param1=2, param2="test")
    )
    base_setup.add_algorithms(alg)

    base_setup.run_by_name("test_alg")

    assert isinstance(base_setup.algorithms["test_alg"].result, FakeResult)


def test_mpe(base_setup):
    """Test the MPE method."""
    alg = FakeAlgorithm(name="test_alg")
    base_setup.add_algorithms(alg)

    base_setup.MPE("test_alg")

    # Since FakeAlgorithm's mpe method doesn't do anything, we just check if it runs without error


def test_mpe_from_plot(base_setup):
    """Test the mpe_from_plot method."""
    alg = FakeAlgorithm(name="test_alg")
    base_setup.add_algorithms(alg)

    base_setup.mpe_from_plot("test_alg")

    # Since FakeAlgorithm's mpe_from_plot method doesn't do anything, we just check if it runs without error


def test_getitem(base_setup):
    """Test the __getitem__ method."""
    alg = FakeAlgorithm(name="test_alg")
    base_setup.add_algorithms(alg)

    retrieved_alg = base_setup["test_alg"]

    assert retrieved_alg.name == "test_alg"


def test_get(base_setup):
    """Test the get method."""
    alg = FakeAlgorithm(name="test_alg")
    base_setup.add_algorithms(alg)

    retrieved_alg = base_setup.get("test_alg")
    nonexistent_alg = base_setup.get("nonexistent", default="Not found")

    assert retrieved_alg.name == "test_alg"
    assert nonexistent_alg == "Not found"


def test_decimate_data():
    """Test the _decimate_data method."""
    data = np.random.rand(10000, 100)
    fs = 1000
    q = 10

    new_data, new_fs, dt, Ndat, T = BaseSetup._decimate_data(data, fs, q)

    assert new_data.shape == (10000, 10)
    assert new_fs == 100
    assert dt == 1 / 100
    assert Ndat == 10000
    assert np.isclose(T, 10, atol=1e-6)


def test_detrend_data():
    """Test the _detrend_data method."""
    data = np.arange(100).reshape(100, 1) + np.random.rand(100, 1)

    detrended_data = BaseSetup._detrend_data(data)

    assert detrended_data.shape == data.shape
    assert np.allclose(np.mean(detrended_data), 0, atol=1e-10)


def test_filter_data():
    """Test the _filter_data method."""
    t = np.linspace(0, 1, 1000, endpoint=False)
    data = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    fs = 1000

    filtered_data = BaseSetup._filter_data(data, fs, Wn=15, btype="lowpass")

    assert filtered_data.shape == data.shape
    # Check that high frequency component is attenuated
    assert np.max(np.abs(np.fft.fft(filtered_data)[200:])) < np.max(
        np.abs(np.fft.fft(data)[200:])
    )
