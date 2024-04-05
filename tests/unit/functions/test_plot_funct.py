import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.pyoma2.functions import plot_funct


def test_CMIF_plot() -> None:
    """Test the CMIF_plot function."""
    S_val = np.random.rand(3, 3, 10)
    freq = np.linspace(0, 10, 10)

    # Test with default parameters
    fig, ax = plot_funct.CMIF_plot(S_val, freq)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test with custom parameters
    fig, ax = plot_funct.CMIF_plot(S_val, freq, freqlim=(2, 8), nSv=2)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_CMIF_plot_exc() -> None:
    """Test the CMIF_plot function with invalid input."""
    S_val = np.random.rand(3, 3, 10)
    freq = np.linspace(0, 10, 10)

    # Test with invalid nSv
    with pytest.raises(ValueError):
        plot_funct.CMIF_plot(S_val, freq, nSv="invalid")
