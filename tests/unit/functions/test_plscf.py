import numpy as np
import pytest
from pyoma2.functions import plscf


@pytest.mark.parametrize("input_sgn_basf", [-1, 1])
def test_pLSCF(input_sgn_basf: int):
    """
    Test the pLSCF function.
    """
    # Define test inputs
    Sy = np.random.rand(3, 3, 100)  # Spectral density matrix
    dt = 0.1  # Time step
    ordmax = 5  # Maximum model order

    # Call the function with test inputs
    Ad, Bn = plscf.pLSCF(Sy, dt, ordmax, input_sgn_basf)

    # Assert that the outputs have the expected shape
    assert [el.shape for el in Ad] == [
        (2, 3, 3),
        (3, 3, 3),
        (4, 3, 3),
        (5, 3, 3),
        (6, 3, 3),
    ]
    assert [el.shape for el in Bn] == [
        (2, 3, 3),
        (3, 3, 3),
        (4, 3, 3),
        (5, 3, 3),
        (6, 3, 3),
    ]

    # Assert that the outputs are of the correct type
    assert all([isinstance(el, np.ndarray) for el in Ad])
    assert all([isinstance(el, np.ndarray) for el in Bn])


def test_pLSCF_Poles() -> None:
    """
    Test the pLSCF_Poles function.
    """
    Ad = np.array([[[[1, -0.5], [1, -0.7]]]])
    Bn = np.array([[[[7, 8], [9, 10]]]])
    dt = 0.01
    methodSy = "per"
    nxseg = 10
    Fns, Xis, Phi1 = plscf.pLSCF_Poles(Ad, Bn, dt, methodSy, nxseg)

    # Check if output types are correct
    assert isinstance(Fns, np.ndarray)
    assert isinstance(Xis, np.ndarray)
    assert isinstance(Phi1, np.ndarray)

    # Check shapes of output arrays
    assert Fns.shape == (2, 1)
    assert Xis.shape == (2, 1)
    assert Phi1.shape == (2, 1, 2)


def test_rmfd2AC() -> None:
    """Test the rmfd2AC function."""
    # Define test data
    A_den = np.array([[[1, 2], [3, 4]]])
    B_num = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])

    # Call the function with test data

    A, C = plscf.rmfd2AC(A_den, B_num)

    # Define expected output
    assert np.allclose(
        A,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        ),
    )
    assert np.allclose(C, ([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))


def test_AC2MP_poly() -> None:
    """Test the AC2MP_poly function."""
    # Define test inputs
    A = np.array([[-1, -2], [1, 0]])
    C = np.array([[1, 0], [0, 1]])
    dt = 0.1
    methodSy = "cor"
    nxseg = 100

    # Call the function with test inputs
    fn, xi, phi = plscf.AC2MP_poly(A, C, dt, methodSy, nxseg)

    assert fn.shape == (2,)
    assert xi.shape == (2,)
    assert phi.shape == (2, 2)


@pytest.mark.parametrize(
    "input_order, expected_order",
    [("find_min", 1), (1, 1), ([0, 1, 2], np.array([0.0, 1.0, 2.0]))],
)
def test_pLSCF_MPE(input_order, expected_order) -> None:
    """Test the pLSCF_MPE function."""
    # Define test inputs
    sel_freq = [1.0, 2.0, 3.0]
    Fn_pol = np.array(
        [
            [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]],
            [[4.0, 5.0, 6.0], [4.1, 5.1, 6.1], [4.2, 5.2, 6.2]],
            [[7.0, 8.0, 9.0], [7.1, 8.1, 9.1], [7.2, 8.2, 9.2]],
        ]
    )
    Xi_pol = np.array(
        [
            [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]],
            [[4.0, 5.0, 6.0], [4.1, 5.1, 6.1], [4.2, 5.2, 6.2]],
            [[7.0, 8.0, 9.0], [7.1, 8.1, 9.1], [7.2, 8.2, 9.2]],
        ]
    )
    Phi_pol = np.array(
        [
            [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]],
            [[4.0, 5.0, 6.0], [4.1, 5.1, 6.1], [4.2, 5.2, 6.2]],
            [[7.0, 8.0, 9.0], [7.1, 8.1, 9.1], [7.2, 8.2, 9.2]],
        ]
    )

    Lab = np.array([[1, 1, 1], [1, 1, 1], [7, 7, 7]])
    deltaf = 0.05
    rtol = 1e-2

    # Call the function with test inputs
    Fn, Xi, Phi, order_out = plscf.pLSCF_MPE(
        sel_freq, Fn_pol, Xi_pol, Phi_pol, input_order, Lab, deltaf, rtol
    )

    if isinstance(order_out, np.ndarray):
        assert np.allclose(order_out, expected_order)
    else:
        assert order_out == expected_order


def test_pLSCF_MPE_exc() -> None:
    """Test the pLSCF_MPE function. Exception case."""
    # Define test inputs
    sel_freq = [1.0, 2.0, 3.0]
    Fn_pol = Xi_pol = Phi_pol = np.array([])
    order = "find_min"
    Lab = None
    deltaf = 0.05
    rtol = 1e-2

    with pytest.raises(ValueError):
        # Call the function with test inputs
        plscf.pLSCF_MPE(sel_freq, Fn_pol, Xi_pol, Phi_pol, order, Lab, deltaf, rtol)
