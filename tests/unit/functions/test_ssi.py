import typing

import numpy as np
import pytest
from pyoma2.functions import ssi

from tests.factory import assert_array_equal_with_nan


@pytest.mark.parametrize(
    "Y, Yref, br, method, calc_unc, expected_hank, expected_uncertainty_is_none, expected_error",
    [
        (
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[1, 2, 3, 4, 5]]),
            1,
            "cov",
            True,
            np.array([[6.0, 4.0], [7.5, 5.0]]),
            None,
            False,
        ),
        (
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[1, 2, 3, 4, 5]]),
            1,
            "cov",
            False,
            np.array([[6.0, 4.0], [7.5, 5.0]]),
            True,
            False,
        ),
        (
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[1, 2, 3, 4, 5]]),
            1,
            "cov_R",
            True,
            None,
            None,
            True,
        ),
        (
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[1, 2, 3, 4, 5]]),
            1,
            "cov_R",
            False,
            np.array([[10.0, 11.0], [8.66666667, 10.0]]),
            True,
            False,
        ),
        (
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[1, 2, 3, 4, 5]]),
            1,
            "dat",
            True,
            None,
            None,
            True,
        ),
        (
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[1, 2, 3, 4, 5]]),
            1,
            "dat",
            False,
            np.array([[2.82842712], [3.53553391]]),
            True,
            False,
        ),
        (
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[1, 2, 3, 4, 5]]),
            1,
            "YfYp",
            True,
            None,
            None,
            True,
        ),
    ],
)
def test_build_hank(
    Y: np.ndarray,
    Yref: np.ndarray,
    br: int,
    method: str,
    calc_unc: bool,
    expected_hank: typing.Union[np.ndarray, None],
    expected_uncertainty_is_none: bool,
    expected_error: bool,
) -> None:
    """Test the build_hank function."""
    if expected_error:
        with pytest.raises(AttributeError) as e:
            ssi.build_hank(
                Y=Y, Yref=Yref, br=br, method=method, calc_unc=calc_unc, nb=100
            )
            assert (
                e.value == "Uncertainty calculations are only available for 'cov' method"
            )
    else:
        hank, uncertainty = ssi.build_hank(
            Y=Y, Yref=Yref, br=br, method=method, calc_unc=calc_unc, nb=100
        )
        assert np.allclose(hank, expected_hank)
        if expected_uncertainty_is_none:
            assert uncertainty is None
        else:
            assert uncertainty is not None


def test_build_hank_invalid_method() -> None:
    """Test the build_hank function with invalid method."""
    with pytest.raises(AttributeError) as e:
        ssi.build_hank(
            Y=np.array([[1, 2, 3, 4, 5]]),
            Yref=np.array([[1, 2, 3, 4, 5]]),
            br=1,
            method="invalid_method",
            calc_unc=False,
            nb=100,
        )
        assert e.value == "Uncertainty calculations are only available for 'cov' method"


def test_ac2mp() -> None:
    """Test the ac2mp function."""

    A = np.array([[-1, -2], [3, -4]])
    C = np.array([[1, 0], [0, 1]])
    dt = 0.1

    fn, xi, phi, *_ = ssi.ac2mp(A, C, dt)
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


def test_SSI_fast() -> None:
    """Test the SSI_fast function."""
    H = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    br = 1
    ordmax = 2
    step = 1

    # Call the function with test input
    A, C, *_ = ssi.SSI_fast(H, br, ordmax, step)

    # Check if the output has the correct shape
    assert A[0].shape == (2,)
    assert A[1].shape == (2,)
    assert A[2].shape == (2,)

    assert C[0].shape == (0, 0)
    assert C[1].shape == (1, 1)
    assert C[2].shape == (2, 2)

    assert all([isinstance(a, np.ndarray) for a in A])
    assert all([isinstance(c, np.ndarray) for c in C])


# def test_SSI_poles() -> None:
#     """Test the SSI_poles function."""
#     # Define test input
#     A = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
#     C = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]
#     ordmax = 2
#     dt = 0.1
#     step = 1

#     # Call the function with test input
#     Fn, Sm, Ms = ssi.SSI_poles(A, C, ordmax, dt, step)

#     # Check if the output has the correct shape
#     assert Fn.shape == (ordmax, int((ordmax) / step + 1))
#     assert Sm.shape == (ordmax, int((ordmax) / step + 1))
#     assert Ms.shape == (ordmax, int((ordmax) / step + 1), C[0].shape[0])

#     # Check if the output is of the correct type
#     assert isinstance(Fn, np.ndarray)
#     assert isinstance(Sm, np.ndarray)
#     assert isinstance(Ms, np.ndarray)


@pytest.mark.parametrize(
    "Obs, AA, CC, ordmax, dt, "
    "step, calc_unc, Q1, Q2, Q3, Q4, "
    "expected_Fn, expected_Xi, expected_Phi, "
    "expected_lambdas, expected_Fn_cov, expected_Xi_cov, expected_Phi_cov",
    [
        # Test case 1: Basic functionality without uncertainty calculation
        (
            np.array([[1]]),  # Obs
            [np.array([[1]]), np.array([[7]])],  # AA
            [np.array([[1]]), np.array([[1]])],  # CC
            1,  # ordmax
            0.01,  # dt
            1,  # step
            False,  # calc_unc
            None,
            None,
            None,
            None,  # Q1, Q2, Q3, Q4
            np.array([[np.nan, 30.9701219]]),  # expected_Fn
            np.array([[np.nan, -1.0]]),  # expected_Xi
            np.array([[[np.nan + 0.0j], [1.0 + 0.0j]]]),  # expected_Phi
            np.array([[np.nan + 0.0j, 194.59101491 + 0.0j]]),  # expected_lambdas
            None,  # expected_Fn_cov
            None,  # expected_Xi_cov
            None,  # expected_Phi_cov
        ),
    ],
)
def test_SSI_poles(
    Obs,
    AA,
    CC,
    ordmax,
    dt,
    step,
    calc_unc,
    Q1,
    Q2,
    Q3,
    Q4,
    expected_Fn,
    expected_Xi,
    expected_Phi,
    expected_lambdas,
    expected_Fn_cov,
    expected_Xi_cov,
    expected_Phi_cov,
) -> None:
    Fn, Xi, Phi, Lambds, Fn_cov, Xi_cov, Phi_cov = ssi.SSI_poles(
        Obs=Obs,
        AA=AA,
        CC=CC,
        ordmax=ordmax,
        dt=dt,
        step=step,
        calc_unc=calc_unc,
        Q1=Q1,
        Q2=Q2,
        Q3=Q3,
        Q4=Q4,
    )
    assert assert_array_equal_with_nan(Fn, expected_Fn)
    assert assert_array_equal_with_nan(Xi, expected_Xi)
    assert assert_array_equal_with_nan(Phi, expected_Phi)
    assert assert_array_equal_with_nan(Lambds, expected_lambdas)

    if expected_Fn_cov:
        assert assert_array_equal_with_nan(Fn_cov, expected_Fn_cov)
    else:
        assert Fn_cov is None
    if expected_Xi_cov:
        assert assert_array_equal_with_nan(Xi_cov, expected_Xi_cov)
    else:
        assert Xi_cov is None
    if expected_Phi_cov:
        assert assert_array_equal_with_nan(Phi_cov, expected_Phi_cov)
    else:
        assert Phi_cov is None


def test_SSI_multi_setup() -> None:
    """Test the SSI_multi_setup function."""
    # Define test data
    Y = [
        {"ref": np.random.rand(3, 10), "mov": np.random.rand(2, 10)},
        {"ref": np.random.rand(3, 10), "mov": np.random.rand(2, 10)},
    ]
    fs = 1.0
    br = 2
    ordmax = 3
    methodHank = "cov"

    # Test with default step and method
    A, C, *_ = ssi.SSI_multi_setup(Y, fs, br, ordmax, methodHank)
    assert isinstance(A, np.ndarray) and isinstance(C, list)
    assert all(isinstance(a, np.ndarray) for a in A) and all(
        isinstance(c, np.ndarray) for c in C
    )

    # Test with non-default step and method
    Obs_all, A, C = ssi.SSI_multi_setup(Y, fs, br, ordmax, step=2, method_hank="cov")

    assert isinstance(Obs_all, np.ndarray) and isinstance(A, list) and isinstance(C, list)
    assert all(isinstance(a, np.ndarray) for a in A) and all(
        isinstance(c, np.ndarray) for c in C
    )

    # Test with invalid method
    with pytest.raises(AttributeError):
        ssi.SSI_multi_setup(Y, fs, br, ordmax, method_hank="INVALID")

    # Test with invalid methodHank
    with pytest.raises(AttributeError):
        ssi.SSI_multi_setup(Y, fs, br, ordmax, "INVALID")


# Dummy MAC function
def MAC(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_SSI_mpe() -> None:
    """Test the SSI_mpe function."""
    # Define test data
    sel_freq = [1.0, 2.0, 3.0]
    Fn_pol = np.random.rand(10, 11)
    Sm_pol = np.random.rand(10, 11)
    Ms_pol = np.random.rand(10, 11, 12)
    order = 5
    Lab = np.random.randint(0, 8, size=(10, 11))

    # Test with default parameters
    Fn, Xi, Phi, order_out, *_ = ssi.SSI_mpe(sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab)
    assert isinstance(Fn, np.ndarray)
    assert isinstance(Xi, np.ndarray)
    assert isinstance(Phi, np.ndarray)
    assert isinstance(order_out, (int, np.ndarray))

    # Test with order as 'find_min'
    Fn, Xi, Phi, order_out, *_ = ssi.SSI_mpe(
        sel_freq, Fn_pol, Sm_pol, Ms_pol, "find_min", Lab
    )
    assert isinstance(Fn, np.ndarray)
    assert isinstance(Xi, np.ndarray)
    assert isinstance(Phi, np.ndarray)
    assert isinstance(order_out, (int, np.ndarray)) if order_out is not None else True

    # Test with order as list of int
    order = [1, 2, 3]
    Fn, Xi, Phi, order_out, *_ = ssi.SSI_mpe(sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab)
    assert isinstance(Fn, np.ndarray)
    assert isinstance(Xi, np.ndarray)
    assert isinstance(Phi, np.ndarray)
    assert isinstance(order_out, (int, np.ndarray)) if order_out is not None else True

    # Test with invalid order
    with pytest.raises(AttributeError):
        ssi.SSI_mpe(sel_freq, Fn_pol, Sm_pol, Ms_pol, "invalid", Lab)

    # Test with order='find_min' but Lab is None
    with pytest.raises(AttributeError):
        ssi.SSI_mpe(sel_freq, Fn_pol, Sm_pol, Ms_pol, "find_min", None)
