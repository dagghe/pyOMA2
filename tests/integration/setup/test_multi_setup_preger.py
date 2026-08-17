import numpy as np
from scipy.signal import decimate, detrend

from pyoma2.algorithms import SSI_MS, pLSCF_MS
from pyoma2.functions.gen import filter_data, pre_multisetup
from pyoma2.setup import MultiSetup_PreGER


def test_geo1(ms_preger: MultiSetup_PreGER) -> None:
    """
    Test the first geometric definition.
    """

    # Test that the geometric is not defined
    assert ms_preger.geo1 is None

    ms_preger.def_geo1_by_file(path="./tests/test_data/3SL/Geo1.xlsx")

    assert ms_preger.geo1 is not None


def test_geo2(ms_preger: MultiSetup_PreGER) -> None:
    """
    Test the second geometric definition.
    """

    # Test that the geometric is not defined
    assert ms_preger.geo2 is None

    ms_preger.def_geo2_by_file(path="./tests/test_data/3SL/Geo2.xlsx")

    assert ms_preger.geo2 is not None


def _assert_preger_data_matches_datasets(ms_preger: MultiSetup_PreGER) -> None:
    expected = pre_multisetup(ms_preger.datasets, ms_preger.ref_ind)
    assert len(ms_preger.data) == len(expected)
    for actual, exp in zip(ms_preger.data, expected, strict=True):
        np.testing.assert_allclose(actual["ref"], exp["ref"])
        np.testing.assert_allclose(actual["mov"], exp["mov"])


def test_plot_data(ms_preger: MultiSetup_PreGER) -> None:
    """
    Test the plotting and data manipulation methods of the MultiSetup_PreGER class.
    """
    originals = [d.copy() for d in ms_preger.datasets]
    initial_fs = ms_preger.fs
    initial_dt = ms_preger.dt

    decimation_factor = 4
    ms_preger.decimate_data(q=decimation_factor)
    for orig, new, ndat in zip(
        originals, ms_preger.datasets, ms_preger.Ndats, strict=True
    ):
        expected = decimate(orig, decimation_factor, axis=0)
        np.testing.assert_allclose(new, expected)
        assert new.shape[0] == ndat == expected.shape[0]
    assert ms_preger.fs == initial_fs / decimation_factor
    assert ms_preger.dt == 1 / ms_preger.fs
    _assert_preger_data_matches_datasets(ms_preger)

    ms_preger.rollback()
    for orig, restored in zip(originals, ms_preger.datasets, strict=True):
        np.testing.assert_array_equal(restored, orig)
    assert ms_preger.fs == initial_fs
    assert ms_preger.dt == initial_dt
    _assert_preger_data_matches_datasets(ms_preger)

    ms_preger.detrend_data()
    for orig, new in zip(originals, ms_preger.datasets, strict=True):
        np.testing.assert_allclose(new, detrend(orig, axis=0))
    assert ms_preger.fs == initial_fs
    assert ms_preger.dt == initial_dt
    _assert_preger_data_matches_datasets(ms_preger)

    ms_preger.rollback()
    ms_preger.filter_data(Wn=1, order=1, btype="lowpass")
    for orig, new in zip(originals, ms_preger.datasets, strict=True):
        np.testing.assert_allclose(
            new, filter_data(orig, fs=initial_fs, Wn=1, order=1, btype="lowpass")
        )
    assert ms_preger.fs == initial_fs
    assert ms_preger.dt == initial_dt
    _assert_preger_data_matches_datasets(ms_preger)

    ms_preger.rollback()
    for orig, restored in zip(originals, ms_preger.datasets, strict=True):
        np.testing.assert_array_equal(restored, orig)
    assert ms_preger.fs == initial_fs
    assert ms_preger.dt == initial_dt

    # test PLOT_DATA method
    try:
        figs, axs = ms_preger.plot_data(data_idx=[0, 1, 2])
        assert isinstance(figs, list)
        assert isinstance(axs, list)
    except Exception as e:
        assert False, f"plot_data raised an exception {e}"

    # test PLOT_CH_INFO method
    try:
        figs, axs = ms_preger.plot_ch_info(data_idx=[0, 1, 2], ch_idx=[-1])
        assert isinstance(figs, list)
        assert isinstance(axs, list)
    except Exception as e:
        assert False, f"plot_ch_info raised an exception {e}"

    # test plot_STFT method
    try:
        figs, axs = ms_preger.plot_STFT(data_idx=[0, 1, 2])
        assert isinstance(figs, list)
        assert isinstance(axs, list)
    except Exception as e:
        assert False, f"plot_STFT raised an exception {e}"


def test_run(ms_preger: MultiSetup_PreGER) -> None:
    """
    Test the running of the algorithms in the MultiSetup_PreGER class.
    """
    # Define geometry1
    ms_preger.def_geo1_by_file(path="./tests/test_data/3SL/Geo1.xlsx")  # BG lines

    # Define geometry 2
    ms_preger.def_geo2_by_file(path="./tests/test_data/3SL/Geo2.xlsx")

    # Initialise the algorithms
    ssidat = SSI_MS(name="SSIdat", br=5, ordmax=5)
    plscf = pLSCF_MS(name="pLSCF", ordmax=5, nxseg=64)

    ms_preger.decimate_data(q=50)

    # Add algorithms to the class
    ms_preger.add_algorithms(ssidat, plscf)

    # Results are None
    assert ms_preger["SSIdat"].result is None
    assert ms_preger["pLSCF"].result is None

    # Run all algorithms
    ms_preger.run_all()

    # Check the results
    assert ms_preger["SSIdat"].result is not None
    assert ms_preger["pLSCF"].result is not None
