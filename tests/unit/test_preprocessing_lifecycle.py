"""Offline regression tests for preprocessing ↔ algorithm lifecycle.

These tests use synthetic arrays so they do not depend on downloaded sample
data. They assert the data the algorithm actually consumed (shape / fs /
Nyquist), not just pointer identity on a dummy ``run()``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import decimate, detrend

from pyoma2.algorithms import FDD, FDD_MS, SSI, SSI_MS
from pyoma2.algorithms.data.run_params import FDDRunParams
from pyoma2.functions.gen import pre_multisetup
from pyoma2.setup import MultiSetup_PreGER, SingleSetup
from pyoma2.setup.base import BaseSetup
from tests.factory import FakeRunParams, RecordingAlgorithm


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _single_setup(
    n_samples: int = 256, n_channels: int = 3, fs: float = 100.0
) -> SingleSetup:
    data = _rng().normal(size=(n_samples, n_channels))
    return SingleSetup(data, fs=fs)


def _preger_setup(
    n_samples: int = 256,
    n_channels: int = 4,
    n_setups: int = 2,
    fs: float = 100.0,
) -> MultiSetup_PreGER:
    datasets = [_rng().normal(size=(n_samples, n_channels)) for _ in range(n_setups)]
    ref_ind = [[0, 1] for _ in range(n_setups)]
    return MultiSetup_PreGER(fs=fs, ref_ind=ref_ind, datasets=datasets)


def _apply_preprocess(setup, op: str) -> None:
    if op == "decimate":
        setup.decimate_data(q=2)
    elif op == "filter":
        setup.filter_data(Wn=10, order=2, btype="lowpass")
    elif op == "detrend":
        setup.detrend_data()
    else:
        raise ValueError(op)


def _assert_algorithm_bound(alg, setup) -> None:
    assert alg.fs == setup.fs
    assert alg.dt == setup.dt == 1 / setup.fs
    assert alg.data is setup.data


def _assert_preger_data_matches_datasets(setup: MultiSetup_PreGER) -> None:
    expected = pre_multisetup(setup.datasets, setup.ref_ind)
    assert len(setup.data) == len(expected)
    for actual, exp in zip(setup.data, expected, strict=True):
        np.testing.assert_allclose(actual["ref"], exp["ref"])
        np.testing.assert_allclose(actual["mov"], exp["mov"])


# ---------------------------------------------------------------------------
# Decimation axis contract
# ---------------------------------------------------------------------------


def test_decimate_data_rejects_nonzero_axis() -> None:
    data = _rng().normal(size=(20, 4))
    with pytest.raises(ValueError, match="axis=0"):
        BaseSetup._decimate_data(data, fs=100.0, q=2, axis=1)


@pytest.mark.parametrize("setup_factory", [_single_setup, _preger_setup])
def test_public_decimate_rejects_nonzero_axis_and_leaves_state(setup_factory) -> None:
    setup = setup_factory()
    fs_before = setup.fs
    dt_before = setup.dt
    data_before = setup.data

    with pytest.raises(ValueError, match="axis=0"):
        setup.decimate_data(q=2, axis=1)

    assert setup.fs == fs_before
    assert setup.dt == dt_before
    assert setup.data is data_before


def test_decimate_axis_0_and_scipy_kwargs_match_scipy() -> None:
    data = _rng().normal(size=(80, 3))
    fs = 100.0
    q = 2
    kwargs = {"n": 8, "ftype": "fir", "zero_phase": True, "axis": 0}

    newdata, new_fs, dt, ndat, duration = BaseSetup._decimate_data(
        data, fs=fs, q=q, **kwargs
    )
    expected = decimate(data, q, **kwargs)

    np.testing.assert_allclose(newdata, expected)
    assert new_fs == fs / q
    assert dt == 1 / new_fs
    assert ndat == expected.shape[0]
    assert np.isclose(duration, ndat / new_fs)


def test_preger_decimate_with_scipy_kwargs_matches_scipy() -> None:
    setup = _preger_setup(n_samples=80)
    originals = [d.copy() for d in setup.datasets]
    q = 2
    kwargs = {"n": 8, "ftype": "fir", "zero_phase": True}

    setup.decimate_data(q=q, **kwargs)

    assert setup.fs == 50.0
    assert setup.dt == 1 / setup.fs
    for orig, new, ndat in zip(originals, setup.datasets, setup.Ndats, strict=True):
        expected = decimate(orig, q, axis=0, **kwargs)
        np.testing.assert_allclose(new, expected)
        assert new.shape[0] == ndat == expected.shape[0]
    _assert_preger_data_matches_datasets(setup)


# ---------------------------------------------------------------------------
# PreGER dataset / metadata invariants
# ---------------------------------------------------------------------------


def test_preger_preprocess_keeps_datasets_and_data_in_sync() -> None:
    setup = _preger_setup()
    originals = [d.copy() for d in setup.datasets]
    initial_fs = setup.fs
    initial_ndats = [d.shape[0] for d in originals]
    initial_ts = [n / initial_fs for n in initial_ndats]

    setup.decimate_data(q=2)
    for orig, new in zip(originals, setup.datasets, strict=True):
        np.testing.assert_allclose(new, decimate(orig, 2, axis=0))
        assert new.shape[0] == orig.shape[0] // 2
    assert setup.fs == initial_fs / 2
    assert setup.dt == 1 / setup.fs
    assert setup.Ndats == [n // 2 for n in initial_ndats]
    assert all(
        np.isclose(t, n / setup.fs) for t, n in zip(setup.Ts, setup.Ndats, strict=True)
    )
    _assert_preger_data_matches_datasets(setup)

    setup.rollback()
    for orig, restored in zip(originals, setup.datasets, strict=True):
        np.testing.assert_array_equal(restored, orig)
    assert setup.fs == initial_fs
    assert setup.dt == 1 / initial_fs
    assert setup.Ndats == initial_ndats
    assert all(np.isclose(a, b) for a, b in zip(setup.Ts, initial_ts, strict=True))
    _assert_preger_data_matches_datasets(setup)

    setup.detrend_data()
    for orig, new in zip(originals, setup.datasets, strict=True):
        np.testing.assert_allclose(new, detrend(orig, axis=0))
    _assert_preger_data_matches_datasets(setup)

    setup.rollback()
    setup.filter_data(Wn=10, order=2, btype="lowpass")
    _assert_preger_data_matches_datasets(setup)
    assert setup.fs == initial_fs


# ---------------------------------------------------------------------------
# Algorithm lifecycle: spy records what run() actually consumed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["decimate", "filter", "detrend"])
def test_single_setup_preprocess_invalidates_and_rerun_records_new_data(op: str) -> None:
    setup = _single_setup()
    alg = RecordingAlgorithm(name="rec", run_params=FakeRunParams())
    setup.add_algorithms(alg)
    setup.run_all()

    first = alg.result
    assert first is not None
    assert first.fs == 100.0
    assert first.n_samples == 256
    first_id = first.data_id

    _apply_preprocess(setup, op)
    assert alg.result is None
    _assert_algorithm_bound(alg, setup)

    setup.run_all()
    second = alg.result
    assert second is not None
    assert second is not first
    assert second.fs == setup.fs
    assert second.dt == setup.dt
    assert second.data_id == id(setup.data)
    assert second.data_id != first_id
    if op == "decimate":
        assert second.n_samples == 128
        assert second.fs == 50.0
    else:
        assert second.n_samples == 256
        assert second.fs == 100.0


@pytest.mark.parametrize("op", ["decimate", "filter", "detrend"])
def test_preger_preprocess_invalidates_and_rerun_records_new_data(op: str) -> None:
    setup = _preger_setup()
    alg = RecordingAlgorithm(name="rec", run_params=FakeRunParams())
    setup.add_algorithms(alg)
    setup.run_all()

    first = alg.result
    assert first is not None
    assert first.fs == 100.0
    assert first.n_samples == 256

    _apply_preprocess(setup, op)
    assert alg.result is None
    _assert_algorithm_bound(alg, setup)

    setup.run_all()
    second = alg.result
    assert second is not None
    assert second is not first
    assert second.fs == setup.fs
    assert second.data_id == id(setup.data)
    if op == "decimate":
        assert second.n_samples == 128
        assert second.fs == 50.0
    else:
        assert second.n_samples == 256


@pytest.mark.parametrize("setup_factory", [_single_setup, _preger_setup])
def test_rollback_preserves_algorithms_and_restores_recorded_data(setup_factory) -> None:
    setup = setup_factory()
    alg = RecordingAlgorithm(name="rec", run_params=FakeRunParams())
    setup.add_algorithms(alg)
    setup.run_all()
    setup.decimate_data(q=2)
    setup.run_all()
    after_decimate = alg.result
    assert after_decimate.fs == 50.0

    setup.rollback()
    assert "rec" in setup.algorithms
    assert alg.result is None
    _assert_algorithm_bound(alg, setup)
    assert setup.fs == 100.0

    setup.run_all()
    restored = alg.result
    assert restored is not after_decimate
    assert restored.fs == 100.0
    assert restored.n_samples == 256
    assert restored.data_id == id(setup.data)


def test_single_rollback_snapshot_survives_inplace_mutation() -> None:
    setup = _single_setup()
    original = setup.data.copy()
    setup.decimate_data(q=2)
    setup.rollback()

    setup.data[0, 0] = 1e6
    setup.rollback()

    np.testing.assert_array_equal(setup.data, original)


def test_preger_rollback_snapshot_survives_inplace_mutation() -> None:
    setup = _preger_setup()
    original_datasets = [d.copy() for d in setup.datasets]
    original_ref_ind = [list(refs) for refs in setup.ref_ind]
    setup.decimate_data(q=2)
    setup.rollback()

    setup.datasets[0][0, 0] = 1e6
    setup.ref_ind[0][0] = 99
    setup.rollback()

    for orig, restored in zip(original_datasets, setup.datasets, strict=True):
        np.testing.assert_array_equal(restored, orig)
    assert setup.ref_ind == original_ref_ind


# ---------------------------------------------------------------------------
# Real algorithms: FDD result + SSI instance caches
# ---------------------------------------------------------------------------


def test_fdd_rerun_after_decimate_uses_new_nyquist() -> None:
    setup = _single_setup(n_samples=512)
    alg = FDD(name="fdd", nxseg=64)
    setup.add_algorithms(alg)
    setup.run_all()

    freq_before = alg.result.freq
    assert np.isclose(freq_before[-1], 50.0)

    setup.decimate_data(q=2)
    assert alg.result is None
    _assert_algorithm_bound(alg, setup)

    setup.run_all()
    assert alg.result is not None
    assert alg.result.freq is not freq_before
    assert np.isclose(alg.result.freq[-1], 25.0)


def test_fdd_ms_rerun_after_decimate_uses_new_nyquist() -> None:
    setup = _preger_setup(n_samples=512)
    alg = FDD_MS(name="fdd_ms", nxseg=64)
    setup.add_algorithms(alg)
    setup.run_all()

    freq_before = alg.result.freq
    assert np.isclose(freq_before[-1], 50.0)

    setup.decimate_data(q=2)
    assert alg.result is None

    setup.run_all()
    assert alg.result.freq is not freq_before
    assert np.isclose(alg.result.freq[-1], 25.0)


def test_ssi_spectrum_cache_invalidated_after_decimate() -> None:
    setup = _single_setup(n_samples=512)
    alg = SSI(name="ssi", br=5, ordmax=10)
    setup.add_algorithms(alg)
    setup.run_all()
    alg.est_spectrum(FDDRunParams(nxseg=64))

    freq_before = alg.freq
    sy_before = alg.Sy
    g_before = alg.G
    assert freq_before is not None
    assert sy_before is not None
    assert g_before is not None
    assert np.isclose(freq_before[-1], 50.0)

    setup.decimate_data(q=2)
    assert alg.result is None
    assert alg.freq is None
    assert alg.Sy is None
    assert alg.G is None
    _assert_algorithm_bound(alg, setup)

    alg.est_spectrum(FDDRunParams(nxseg=64))
    assert alg.freq is not freq_before
    assert alg.Sy is not sy_before
    assert np.isclose(alg.freq[-1], 25.0)


def test_ssi_plot_stab_reestimates_spectrum_after_decimate() -> None:
    setup = _single_setup(n_samples=512)
    alg = SSI(name="ssi", br=5, ordmax=10)
    setup.add_algorithms(alg)
    setup.run_all()
    alg.est_spectrum(FDDRunParams(nxseg=64))
    setup.decimate_data(q=2)
    setup.run_all()

    fig, ax = alg.plot_stab(spectrum=True)

    assert fig is not None
    assert ax is not None
    assert alg.Sy is not None
    assert np.isclose(alg.freq[-1], 25.0)


def test_ssi_ms_spectrum_cache_invalidated_after_decimate() -> None:
    setup = _preger_setup(n_samples=512)
    alg = SSI_MS(name="ssi_ms", br=5, ordmax=10)
    setup.add_algorithms(alg)
    alg.est_spectrum(FDDRunParams(nxseg=64))

    freq_before = alg.freq
    sy_before = alg.Sy
    assert np.isclose(freq_before[-1], 50.0)

    setup.decimate_data(q=2)
    assert alg.freq is None
    assert alg.Sy is None

    alg.est_spectrum(FDDRunParams(nxseg=64))
    assert alg.freq is not freq_before
    assert alg.Sy is not sy_before
    assert np.isclose(alg.freq[-1], 25.0)
