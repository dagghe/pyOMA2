import numpy as np
import pytest

from src.pyoma2.functions import Gen_funct


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        (
            {
                "Fn": np.array([[1.0, 2.0], [2.0, 3.0]]),
                "Sm": np.array([[0.1, 0.2], [0.3, 0.4]]),
                "Ms": np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
                "ordmin": 1,
                "ordmax": 2,
                "step": 1,
                "err_fn": 0.1,
                "err_xi": 0.1,
                "err_ms": 0.1,
                "max_xi": 0.5,
            },
            np.array([[0, 0], [0, 0]]),
        ),
    ],
)
def test_lab_stab(test_input, expected_output) -> None:
    """Test the lab_stab function."""
    result = Gen_funct.lab_stab(**test_input)
    np.testing.assert_array_equal(result, expected_output)


def test_merge_mode_shapes() -> None:
    """Test the merge_mode_shapes function."""
    MSarr_list = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    reflist = [[0], [1]]
    merged = Gen_funct.merge_mode_shapes(MSarr_list, reflist)
    assert merged.shape == (3, 2)


def test_merge_mode_shape_exc() -> None:
    """Test the merge_mode_shapes function with an exception."""
    MSarr_list = [np.array([[1, 2], [3, 4]]), np.array([[5], [7]])]
    reflist = [[0], [1], [2]]
    with pytest.raises(ValueError) as excinfo:
        Gen_funct.merge_mode_shapes(MSarr_list, reflist)
    assert "All mode shape arrays must have the same number of modes." in str(
        excinfo.value
    )


def test_MPC() -> None:
    """Test the MPC function."""
    phi = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    assert Gen_funct.MPC(phi) == pytest.approx(1.0)


def test_MPD() -> None:
    """Test the MPD function."""
    phi = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    assert Gen_funct.MPD(phi) == pytest.approx(0.052641260122719684)


def test_MSF() -> None:
    """Test the MSF function."""
    phi_1 = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    phi_2 = np.array([2 + 3j, 3 + 4j, 4 + 5j])
    assert Gen_funct.MSF(phi_1, phi_2) == pytest.approx([1.35342466])


def test_MSF_exc() -> None:
    """Test the MSF function with an exception."""
    phi_1 = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    phi_2 = np.array([2 + 3j, 3 + 4j, 4 + 5j, 5 + 6j])
    with pytest.raises(Exception) as excinfo:
        Gen_funct.MSF(phi_1, phi_2)
    assert "`phi_1` and `phi_2` must have the same shape" in str(excinfo.value)


def test_MCF() -> None:
    """Test the MCF function."""
    phi = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    assert Gen_funct.MCF(phi) == pytest.approx([0.01297999])


def test_MAC() -> None:
    """Test the MAC function."""
    phi_X = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    phi_A = np.array([2 + 3j, 3 + 4j, 4 + 5j])
    assert Gen_funct.MAC(phi_X, phi_A) == pytest.approx(0.9929349425964087 + 0j)


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
        Gen_funct.MAC(input_phi_X, phi_A)
    assert expected_exc_msg in str(excinfo.value)


def test_PRE_MultiSetup() -> None:
    """Test the pre_MultiSetup function."""
    DataList = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    reflist = [[0], [1]]
    result = Gen_funct.pre_MultiSetup(DataList, reflist)
    assert len(result) == len(DataList)
    assert "ref" in result[0] and "mov" in result[0]


def test_invperm() -> None:
    """Test the invperm function."""
    p = np.array([3, 0, 2, 1])
    assert np.array_equal(Gen_funct.invperm(p), np.array([1, 3, 2, 0]))


def test_find_map() -> None:
    """Test the find_map function."""
    arr1 = np.array([10, 30, 20])
    arr2 = np.array([3, 2, 1])
    assert np.array_equal(Gen_funct.find_map(arr1, arr2), np.array([2, 0, 1]))


@pytest.mark.parametrize(
    "fs, Wn, order, btype, expected_shape",
    [
        (1000, 100, 4, "lowpass", (100, 2)),
    ],
)
def test_filter_data(fs, Wn, order, btype, expected_shape) -> None:
    """Test the filter_data function."""
    data = np.random.rand(100, 2)
    filt_data = Gen_funct.filter_data(data, fs, Wn, order, btype)
    assert filt_data.shape == expected_shape
