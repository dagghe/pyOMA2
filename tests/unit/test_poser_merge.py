"""Tests for the PoSER merge of single-setup SSI results."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyoma2.algorithms import FDD, SSI
from pyoma2.algorithms.data.result import FDDResult, SSIResult
from pyoma2.setup import MultiSetup_PoSER, SingleSetup

# Mode shapes on the three shared reference sensors; pairwise MAC < 0.1
REF_A = [1.0, 0.2, 0.1]
REF_B = [0.1, 1.0, -0.2]
REF_C = [-0.2, 0.1, 1.0]


def _phi(*columns: list[float]) -> np.ndarray:
    """Stack mode shape columns laid out as [reference DOFs..., roving DOFs...]."""
    return np.column_stack([np.asarray(col, dtype=float) for col in columns])


def _poser(*results: SSIResult, ref_ind: list[list[int]] | None = None):
    """Build a MultiSetup_PoSER around already identified SSI results."""
    setups = []
    for i, res in enumerate(results):
        ss = SingleSetup(np.zeros((64, res.Phi.shape[0])), fs=100.0)
        alg = SSI(name=f"ssi_{i}", br=10)
        ss.add_algorithms(alg)
        alg._set_result(res)
        setups.append(ss)
    if ref_ind is None:
        ref_ind = [[0, 1, 2] for _ in results]
    return MultiSetup_PoSER(ref_ind=ref_ind, single_setups=setups, names=["ssi"])


def _merge(*results: SSIResult, **kwargs):
    return _poser(*results).merge_results(**kwargs)["ssi"]


def test_aligned_modes_keep_index_merge() -> None:
    """Same modes in the same order: identical to the index merge."""
    fn = [np.array([2.0, 5.0, 9.0]), np.array([2.1, 5.1, 9.1])]
    xi = [np.array([0.010, 0.020, 0.030]), np.array([0.012, 0.018, 0.033])]
    phi_a = _phi(REF_A + [0.5, 0.6], REF_B + [0.7, 0.8], REF_C + [0.9, 1.1])
    phi_b = _phi(REF_A + [0.4, 0.3], REF_B + [0.2, 0.1], REF_C + [1.2, 1.3])

    merged = _merge(
        SSIResult(Fn=fn[0], Xi=xi[0], Phi=phi_a),
        SSIResult(Fn=fn[1], Xi=xi[1], Phi=phi_b),
    )

    assert_allclose(merged.Fn, np.mean(fn, axis=0))
    assert_allclose(merged.Fn_std, np.std(fn, axis=0))
    assert_allclose(merged.Xi, np.mean(xi, axis=0))
    assert_allclose(merged.Xi_std, np.std(xi, axis=0))
    assert_allclose(merged.Phi, np.vstack([phi_a, phi_b[3:]]))
    assert merged.setups_used == [[0, 1], [0, 1], [0, 1]]


def test_reordered_modes_are_paired_by_reference_shape() -> None:
    """Setup B lists the 5 Hz and 9 Hz modes swapped: they must not be cross-averaged."""
    merged = _merge(
        SSIResult(
            Fn=np.array([2.0, 5.0, 9.0]),
            Xi=np.array([0.01, 0.02, 0.03]),
            Phi=_phi(REF_A + [0.5], REF_B + [0.7], REF_C + [0.9]),
        ),
        SSIResult(
            Fn=np.array([2.1, 9.1, 5.1]),
            Xi=np.array([0.01, 0.03, 0.02]),
            Phi=_phi(REF_A + [0.4], REF_C + [1.2], REF_B + [0.2]),
        ),
    )

    assert_allclose(merged.Fn, [2.05, 5.05, 9.05])
    assert_allclose(merged.Xi, [0.01, 0.02, 0.03])
    # roving DOF of setup B lands on the matching physical mode
    assert_allclose(merged.Phi[4].real, [0.4, 0.2, 1.2])


def test_close_modes_are_paired_by_shape_not_nearest_frequency() -> None:
    """Close modes whose frequency scatter crosses are paired by reference MAC."""
    merged = _merge(
        SSIResult(
            Fn=np.array([2.63, 2.69]),
            Xi=np.array([0.01, 0.02]),
            Phi=_phi(REF_A + [0.5], REF_B + [0.7]),
        ),
        SSIResult(
            Fn=np.array([2.64, 2.68]),
            Xi=np.array([0.02, 0.01]),
            Phi=_phi(REF_B + [0.2], REF_A + [0.4]),
        ),
    )

    assert_allclose(merged.Fn, [(2.63 + 2.68) / 2, (2.69 + 2.64) / 2])
    assert_allclose(merged.Xi, [0.01, 0.02])


def test_different_modes_at_same_index_are_not_averaged() -> None:
    """Equal mode counts do not force a pairing between different physical modes."""
    merged = _merge(
        SSIResult(
            Fn=np.array([2.0, 5.0]),
            Xi=np.array([0.01, 0.02]),
            Phi=_phi(REF_A + [0.5], REF_B + [0.7]),
        ),
        SSIResult(
            Fn=np.array([2.1, 5.2]),
            Xi=np.array([0.01, 0.04]),
            Phi=_phi(REF_A + [0.4], REF_C + [1.2]),
        ),
    )

    assert_allclose(merged.Fn, [2.05, 5.0, 5.2])
    assert merged.setups_used == [[0, 1], [0], [1]]


def test_roving_sensors_are_scaled_onto_the_reference_setup() -> None:
    """Setup B is 10x setup A on the references, so its roving DOFs are divided by 10."""
    merged = _poser(
        SSIResult(
            Fn=np.array([3.0]), Xi=np.array([0.01]), Phi=_phi([1.0, 2.0, 5.0, 6.0])
        ),
        SSIResult(
            Fn=np.array([3.0]), Xi=np.array([0.01]), Phi=_phi([10.0, 20.0, 30.0, 40.0])
        ),
        ref_ind=[[0, 1], [0, 1]],
    ).merge_results()["ssi"]

    assert_allclose(merged.Phi[:, 0].real, [1.0, 2.0, 5.0, 6.0, 3.0, 4.0])


def test_missing_mode_is_merged_from_the_setups_that_identified_it() -> None:
    """The 9 Hz mode missing in setup B is kept, merged from setup A alone."""
    merged = _merge(
        SSIResult(
            Fn=np.array([2.0, 5.0, 9.0]),
            Xi=np.array([0.01, 0.02, 0.03]),
            Phi=_phi(REF_A + [0.5], REF_B + [0.7], REF_C + [0.9]),
        ),
        SSIResult(
            Fn=np.array([2.1, 5.1]),
            Xi=np.array([0.01, 0.02]),
            Phi=_phi(REF_A + [0.4], REF_B + [0.2]),
        ),
    )

    assert_allclose(merged.Fn, [2.05, 5.05, 9.0])
    assert_allclose(merged.Fn_std, [0.05, 0.05, 0.0])
    assert_allclose(merged.Xi, [0.01, 0.02, 0.03])
    assert merged.setups_used == [[0, 1], [0, 1], [0]]
    # setup A provides references and its roving DOF, setup B's roving DOF is unknown
    assert_allclose(merged.Phi[:4, 2].real, REF_C + [0.9])
    assert np.isnan(merged.Phi[4, 2])


def test_mode_found_only_in_a_later_setup_is_appended() -> None:
    """A mode that the first setup missed becomes a new merged mode."""
    merged = _merge(
        SSIResult(
            Fn=np.array([2.0, 9.0]),
            Xi=np.array([0.01, 0.03]),
            Phi=_phi(REF_A + [0.5], REF_C + [0.9]),
        ),
        SSIResult(
            Fn=np.array([2.1, 5.1, 9.1]),
            Xi=np.array([0.01, 0.02, 0.03]),
            Phi=_phi(REF_A + [0.4], REF_B + [0.2], REF_C + [1.2]),
        ),
    )

    assert_allclose(merged.Fn, [2.05, 9.05, 5.1])
    assert merged.setups_used == [[0, 1], [0, 1], [1]]
    assert_allclose(merged.Phi[:3, 2].real, REF_B)
    assert np.isnan(merged.Phi[3, 2])
    assert_allclose(merged.Phi[4, 2].real, 0.2)


def test_mode_missing_in_a_middle_setup() -> None:
    """With three setups, a mode missing in the second one is merged from the others."""
    merged = _merge(
        SSIResult(
            Fn=np.array([2.0, 5.0]),
            Xi=np.array([0.01, 0.02]),
            Phi=_phi(REF_A + [0.5], REF_B + [0.7]),
        ),
        SSIResult(Fn=np.array([2.15]), Xi=np.array([0.01]), Phi=_phi(REF_A + [0.4])),
        SSIResult(
            Fn=np.array([2.1, 5.2]),
            Xi=np.array([0.01, 0.04]),
            Phi=_phi(REF_A + [0.3], REF_B + [0.1]),
        ),
    )

    assert merged.setups_used == [[0, 1, 2], [0, 2]]
    assert_allclose(merged.Fn, [np.mean([2.0, 2.15, 2.1]), 5.1])
    assert_allclose(merged.Fn_std, [np.std([2.0, 2.15, 2.1]), 0.1])
    assert_allclose(merged.Xi, [0.01, 0.03])
    # roving DOFs of the three setups are rows 3, 4 and 5: the second one is unknown
    assert_allclose(merged.Phi[[3, 5], 1].real, [0.7, 0.1])
    assert np.isnan(merged.Phi[4, 1])


@pytest.mark.parametrize(
    "kwargs, expected_fn",
    [({}, [2.0, 2.3]), ({"freq_tol": 0.2}, [2.15])],
    ids=["default tolerance splits", "wider tolerance pairs"],
)
def test_frequency_tolerance_is_configurable(kwargs, expected_fn) -> None:
    """Modes further apart than ``freq_tol`` are not treated as the same mode."""
    merged = _merge(
        SSIResult(Fn=np.array([2.0]), Xi=np.array([0.01]), Phi=_phi(REF_A + [0.5])),
        SSIResult(Fn=np.array([2.3]), Xi=np.array([0.01]), Phi=_phi(REF_A + [0.4])),
        **kwargs,
    )

    assert_allclose(merged.Fn, expected_fn)


def test_zero_frequency_tolerance_is_rejected() -> None:
    """``freq_tol=0`` cannot define a pairing, even for identical frequencies."""
    msp = _poser(
        SSIResult(Fn=np.array([2.0]), Xi=np.array([0.01]), Phi=_phi(REF_A + [0.5])),
        SSIResult(Fn=np.array([2.0]), Xi=np.array([0.01]), Phi=_phi(REF_A + [0.4])),
    )
    with pytest.raises(ValueError, match="freq_tol must be positive"):
        msp.merge_results(freq_tol=0.0)


def test_results_without_damping_are_merged() -> None:
    """FDD does not identify damping: the merged damping ratios are None."""
    setups = []
    for i, phi in enumerate([_phi(REF_A + [0.5]), _phi(REF_A + [0.4])]):
        ss = SingleSetup(np.zeros((64, 4)), fs=100.0)
        alg = FDD(name=f"fdd_{i}")
        ss.add_algorithms(alg)
        alg._set_result(FDDResult(Fn=np.array([2.0 + i / 10]), Phi=phi))
        setups.append(ss)
    msp = MultiSetup_PoSER(ref_ind=[[0, 1, 2]] * 2, single_setups=setups, names=["fdd"])

    merged = msp.merge_results()["fdd"]

    assert_allclose(merged.Fn, [2.05])
    assert merged.Xi is None
    assert merged.Xi_std is None
