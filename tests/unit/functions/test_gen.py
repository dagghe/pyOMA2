import numpy as np
import pytest

from pyoma2.functions import gen


def test_merge_mode_shapes() -> None:
    """Test the merge_mode_shapes function."""
    MSarr_list = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    reflist = [[0], [1]]
    merged = gen.merge_mode_shapes(MSarr_list, reflist)
    assert merged.shape == (3, 2)


def test_merge_mode_shapes_scales_onto_first_setup() -> None:
    """Roving DOFs are scaled by the factor mapping each setup onto the first one."""
    MSarr_list = [np.array([[1.0], [2.0], [5.0]]), np.array([[10.0], [20.0], [30.0]])]
    merged = gen.merge_mode_shapes(MSarr_list, [[0, 1], [0, 1]])
    np.testing.assert_allclose(merged[:, 0].real, [1.0, 2.0, 5.0, 3.0])


def test_merge_mode_shapes_missing_mode() -> None:
    """An all-NaN column marks a mode the setup did not identify."""
    MSarr_list = [
        np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 5.0]]),
        np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]),
    ]
    merged = gen.merge_mode_shapes(MSarr_list, [[0, 1], [0, 1]])
    # the first setup that identified the mode provides references and scale
    np.testing.assert_allclose(merged[[0, 1, 3], 0].real, [10.0, 20.0, 30.0])
    assert np.isnan(merged[2, 0])
    np.testing.assert_allclose(merged[:, 1].real, [1.0, 2.0, 5.0, 3.0])


def test_match_modes() -> None:
    """Local modes are paired with merged modes, unmatched ones are appended."""
    ref = np.array([[1.0, 0.1, -0.2], [0.2, 1.0, 0.1], [0.1, -0.2, 1.0]])
    Fn_list = [np.array([2.0, 5.0]), np.array([9.1, 2.1, 5.2])]
    MSarr_list = [ref[:, :2], ref[:, [2, 0, 1]]]
    mode_map = gen.match_modes(Fn_list, MSarr_list, [[0, 1, 2], [0, 1, 2]])
    np.testing.assert_array_equal(mode_map, [[0, 1, -1], [1, 2, 0]])


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, [[0, -1], [-1, 0]]),
        ({"freq_tol": 0.2}, [[0], [0]]),
        ({"freq_tol": 0.2, "mac_min": 0.95}, [[0, -1], [-1, 0]]),
    ],
)
def test_match_modes_gates(kwargs, expected) -> None:
    """Pairs outside the frequency tolerance or below the MAC threshold are rejected."""
    Fn_list = [np.array([2.0]), np.array([2.3])]
    MSarr_list = [np.array([[1.0], [0.2], [0.1]]), np.array([[1.0], [0.5], [0.4]])]
    mode_map = gen.match_modes(Fn_list, MSarr_list, [[0, 1, 2], [0, 1, 2]], **kwargs)
    np.testing.assert_array_equal(mode_map, expected)


def test_merge_modal_params() -> None:
    """NaN marks a missing mode; weights apply only when every contributor has a std."""
    values = np.array([[2.0, 5.0, 9.0], [2.2, np.nan, 9.2]])
    stds = np.array([[0.1, 0.1, 0.1], [0.3, np.nan, np.nan]])
    mean, std = gen.merge_modal_params(values, stds)
    w = 1 / np.array([0.1, 0.3]) ** 2
    np.testing.assert_allclose(mean, [np.sum(w * [2.0, 2.2]) / np.sum(w), 5.0, 9.1])
    np.testing.assert_allclose(std, [1 / np.sqrt(np.sum(w)), 0.1, 0.1])


def test_merge_mode_shape_exc() -> None:
    """Test the merge_mode_shapes function with an exception."""
    MSarr_list = [np.array([[1, 2], [3, 4]]), np.array([[5], [7]])]
    reflist = [[0], [1], [2]]
    with pytest.raises(ValueError) as excinfo:
        gen.merge_mode_shapes(MSarr_list, reflist)
    assert "All mode shape arrays must have the same number of modes." in str(
        excinfo.value
    )


def test_MPC() -> None:
    """Test the MPC function."""
    phi = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    assert gen.MPC(phi) == pytest.approx(1.0)


def test_MPD() -> None:
    """Test the MPD function."""
    phi = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    assert gen.MPD(phi) == pytest.approx(0.052641260122719684)


def test_MSF() -> None:
    """Test the MSF function."""
    phi_1 = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    phi_2 = np.array([2 + 3j, 3 + 4j, 4 + 5j])
    assert gen.MSF(phi_1, phi_2) == pytest.approx([1.35342466])


def test_MSF_exc() -> None:
    """Test the MSF function with an exception."""
    phi_1 = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    phi_2 = np.array([2 + 3j, 3 + 4j, 4 + 5j, 5 + 6j])
    with pytest.raises(Exception) as excinfo:
        gen.MSF(phi_1, phi_2)
    assert "`phi_1` and `phi_2` must have the same shape" in str(excinfo.value)


def test_MCF() -> None:
    """Test the MCF function."""
    phi = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    assert gen.MCF(phi) == pytest.approx([0.01297999])


def test_MAC() -> None:
    """Test the MAC function."""
    phi_X = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    phi_A = np.array([2 + 3j, 3 + 4j, 4 + 5j])
    assert gen.MAC(phi_X, phi_A) == pytest.approx(0.9929349425964087 + 0j)


@pytest.mark.parametrize(
    "input_phi_X, expected_exc_msg",
    [
        (
            np.array([[1 + 2j, 2 + 3j, 3 + 4j]]),
            "Mode shapes must have the same first dimension",
        ),
        (
            np.array([[[1 + 2j, 2 + 3j, 3 + 4j]]]),
            " shape matrices must have 1 or 2 dimensions ",
        ),
    ],
)
def test_MAC_exc(input_phi_X: np.ndarray, expected_exc_msg: str) -> None:
    """Test the MAC function with an exception."""
    phi_A = np.array([2 + 3j, 3 + 4j, 4 + 5j, 5 + 6j])
    with pytest.raises(Exception) as excinfo:
        gen.MAC(input_phi_X, phi_A)
    assert expected_exc_msg in str(excinfo.value)


def test_PRE_MultiSetup() -> None:
    """Test the pre_multisetup function."""
    DataList = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    reflist = [[0], [1]]
    result = gen.pre_multisetup(DataList, reflist)
    assert len(result) == len(DataList)
    assert "ref" in result[0] and "mov" in result[0]


def test_invperm() -> None:
    """Test the invperm function."""
    p = np.array([3, 0, 2, 1])
    assert np.array_equal(gen.invperm(p), np.array([1, 3, 2, 0]))


def test_find_map() -> None:
    """Test the find_map function."""
    arr1 = np.array([10, 30, 20])
    arr2 = np.array([3, 2, 1])
    assert np.array_equal(gen.find_map(arr1, arr2), np.array([2, 0, 1]))


@pytest.mark.parametrize(
    "fs, Wn, order, btype, expected_shape",
    [
        (1000, 100, 4, "lowpass", (100, 2)),
    ],
)
def test_filter_data(fs, Wn, order, btype, expected_shape) -> None:
    """Test the filter_data function."""
    data = np.random.rand(100, 2)
    filt_data = gen.filter_data(data, fs, Wn, order, btype)
    assert filt_data.shape == expected_shape
