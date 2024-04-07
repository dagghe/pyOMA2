import numpy as np
import pytest

from src.pyoma2.functions import plot_funct


def test_CMIF_plot() -> None:
    """Test the CMIF_plot function."""
    S_val = np.random.rand(3, 3, 10)
    freq = np.linspace(0, 10, 10)

    # Test with default parameters
    try:
        fig, ax = plot_funct.CMIF_plot(S_val, freq)
    except Exception as e:
        assert False, f"CMIF_plot raised an exception {e}"

    # Test with custom parameters
    try:
        fig, ax = plot_funct.CMIF_plot(S_val, freq, freqlim=(2, 8), nSv=2)
    except Exception as e:
        assert False, f"CMIF_plot raised an exception {e}"


def test_CMIF_plot_exc() -> None:
    """Test the CMIF_plot function with invalid input."""
    S_val = np.random.rand(3, 3, 10)
    freq = np.linspace(0, 10, 10)

    # Test with invalid nSv
    with pytest.raises(ValueError):
        plot_funct.CMIF_plot(S_val, freq, nSv="invalid")
