import numpy as np
import pytest
from pyoma2.functions import ssi


@pytest.mark.parametrize(
    "Y, Yref, br, fs, method, expected",
    [
        (
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[1, 2, 3, 4, 5]]),
            1,
            1.0,
            "cov_mm",
            np.array([[6.0, 4.0], [7.5, 5.0]]),
        ),
        (
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[1, 2, 3, 4, 5]]),
            1,
            1.0,
            "cov_R",
            np.array([[10.0, 11.0], [8.66666667, 10.0]]),
        ),
        (
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[1, 2, 3, 4, 5]]),
            1,
            1.0,
            "dat",
            np.array([[2.82842712], [3.53553391]]),
        ),
        (
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[1, 2, 3, 4, 5]]),
            1,
            1.0,
            "YfYp",
            (
                np.array([[2.82842712], [3.53553391]]),
                np.array([[2.12132034], [1.41421356]]),
            ),
        ),
    ],
)
def test_BuildHank(Y, Yref, br, fs, method, expected) -> None:
    """Test the BuildHank function."""
    result = ssi.BuildHank(Y, Yref, br, fs, method)
    assert np.allclose(result, expected)


def test_BuildHank_invalid_method() -> None:
    """Test the BuildHank function with invalid method."""
    with pytest.raises(ValueError):
        ssi.BuildHank(
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[1, 2, 3, 4, 5]]),
            1,
            1.0,
            "invalid_method",
        )


def test_AC2MP() -> None:
    """Test the AC2MP function."""

    A = np.array([[-1, -2], [3, -4]])
    C = np.array([[1, 0], [0, 1]])
    dt = 0.1

    fn, xi, phi = ssi.AC2MP(A, C, dt)
    assert fn.shape == (2,)
    assert xi.shape == (2,)
    assert phi.shape == (2, 2)

    assert np.allclose(fn, np.array([4.35528095, 4.35528095]))
    assert np.allclose(xi, np.array([-0.4207166, -0.4207166]))
    assert np.allclose(
        phi, np.array([[0.5 + 0.64549722j, 1.0 + 0.0j], [0.5 - 0.64549722j, 1.0 + 0.0j]])
    )

    assert isinstance(fn, np.ndarray)
    assert isinstance(xi, np.ndarray)
    assert isinstance(phi, np.ndarray)


def test_SSI() -> None:
    """Test the SSI function."""
    # Define test input
    H = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    br = 1
    ordmax = 2
    step = 1

    # Call the function with test input
    A, C = ssi.SSI(H, br, ordmax, step)

    # Check if the output has the correct shape
    assert A[0].shape == (0, 0)
    assert A[1].shape == (1, 1)
    assert A[2].shape == (2, 2)

    assert C[0].shape == (2, 0)
    assert C[1].shape == (2, 1)
    assert C[2].shape == (2, 2)

    # Check if the output is of the correct type
    assert all([isinstance(a, np.ndarray) for a in A])
    assert all([isinstance(c, np.ndarray) for c in C])


def test_SSI_FAST() -> None:
    """Test the SSI_FAST function."""
    H = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    br = 1
    ordmax = 2
    step = 1

    # Call the function with test input
    A, C = ssi.SSI_FAST(H, br, ordmax, step)

    # Check if the output has the correct shape
    assert A[0].shape == (0, 0)
    assert A[1].shape == (1, 1)
    assert A[2].shape == (2, 2)

    assert C[0].shape == (2, 0)
    assert C[1].shape == (2, 1)
    assert C[2].shape == (2, 2)

    assert all([isinstance(a, np.ndarray) for a in A])
    assert all([isinstance(c, np.ndarray) for c in C])


def test_SSI_Poles() -> None:
    """Test the SSI_Poles function."""
    # Define test input
    A = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    C = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]
    ordmax = 2
    dt = 0.1
    step = 1

    # Call the function with test input
    Fn, Sm, Ms = ssi.SSI_Poles(A, C, ordmax, dt, step)

    # Check if the output has the correct shape
    assert Fn.shape == (ordmax, int((ordmax) / step + 1))
    assert Sm.shape == (ordmax, int((ordmax) / step + 1))
    assert Ms.shape == (ordmax, int((ordmax) / step + 1), C[0].shape[0])

    # Check if the output is of the correct type
    assert isinstance(Fn, np.ndarray)
    assert isinstance(Sm, np.ndarray)
    assert isinstance(Ms, np.ndarray)


def test_SSI_MulSet() -> None:
    """Test the SSI_MulSet function."""
    # Define test data
    Y = [
        {"ref": np.random.rand(3, 10), "mov": np.random.rand(2, 10)},
        {"ref": np.random.rand(3, 10), "mov": np.random.rand(2, 10)},
    ]
    fs = 1.0
    br = 2
    ordmax = 3
    methodHank = "cov_mm"

    # Test with default step and method
    A, C = ssi.SSI_MulSet(Y, fs, br, ordmax, methodHank)
    assert isinstance(A, list) and isinstance(C, list)
    assert all(isinstance(a, np.ndarray) for a in A) and all(
        isinstance(c, np.ndarray) for c in C
    )

    # Test with non-default step and method
    A, C = ssi.SSI_MulSet(Y, fs, br, ordmax, methodHank, step=2, method="SLOW")
    assert isinstance(A, list) and isinstance(C, list)
    assert all(isinstance(a, np.ndarray) for a in A) and all(
        isinstance(c, np.ndarray) for c in C
    )

    # Test with invalid method
    with pytest.raises(ValueError):
        ssi.SSI_MulSet(Y, fs, br, ordmax, methodHank, method="INVALID")

    # Test with invalid methodHank
    with pytest.raises(ValueError):
        ssi.SSI_MulSet(Y, fs, br, ordmax, "INVALID")


# Dummy MAC function
def MAC(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_SSI_MPE() -> None:
    """Test the SSI_MPE function."""
    # Define test data
    sel_freq = [1.0, 2.0, 3.0]
    Fn_pol = np.random.rand(10, 11)
    Sm_pol = np.random.rand(10, 11)
    Ms_pol = np.random.rand(10, 11, 12)
    order = 5
    Lab = np.random.randint(0, 8, size=(10, 11))

    # Test with default parameters
    Fn, Xi, Phi, order_out = ssi.SSI_MPE(sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab)
    assert isinstance(Fn, np.ndarray)
    assert isinstance(Xi, np.ndarray)
    assert isinstance(Phi, np.ndarray)
    assert isinstance(order_out, (int, np.ndarray))

    # Test with order as 'find_min'
    Fn, Xi, Phi, order_out = ssi.SSI_MPE(
        sel_freq, Fn_pol, Sm_pol, Ms_pol, "find_min", Lab
    )
    assert isinstance(Fn, np.ndarray)
    assert isinstance(Xi, np.ndarray)
    assert isinstance(Phi, np.ndarray)
    assert isinstance(order_out, (int, np.ndarray)) if order_out is not None else True

    # Test with order as list of int
    order = [1, 2, 3]
    Fn, Xi, Phi, order_out = ssi.SSI_MPE(sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab)
    assert isinstance(Fn, np.ndarray)
    assert isinstance(Xi, np.ndarray)
    assert isinstance(Phi, np.ndarray)
    assert isinstance(order_out, (int, np.ndarray)) if order_out is not None else True

    # Test with invalid order
    with pytest.raises(ValueError):
        ssi.SSI_MPE(sel_freq, Fn_pol, Sm_pol, Ms_pol, "invalid", Lab)

    # Test with order='find_min' but Lab is None
    with pytest.raises(ValueError):
        ssi.SSI_MPE(sel_freq, Fn_pol, Sm_pol, Ms_pol, "find_min", None)
