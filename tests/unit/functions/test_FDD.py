import numpy as np
import pytest

from src.pyoma2.functions import FDD_funct


@pytest.mark.parametrize(
    "input_method",
    [
        "cor",
        "per",
    ],
)
def test_SD_PreGER(input_method: str) -> None:
    fs = 1000.0
    nxseg = 1024
    pov = 0.5

    Y = [
        {"ref": np.random.rand(2, 10000), "mov": np.random.rand(3, 10000)},
        {"ref": np.random.rand(2, 10000), "mov": np.random.rand(3, 10000)},
    ]
    freq, Sy = FDD_funct.SD_PreGER(Y, fs, nxseg=nxseg, pov=pov, method=input_method)
    assert len(freq) > 0  # Ensure frequency array is not empty

    # Check that the output is a tuple
    assert isinstance(freq, np.ndarray)
    assert isinstance(Sy, np.ndarray)
    assert Sy.shape[0] == 8  # Ensure correct shape of Sy


@pytest.mark.parametrize(
    "input_method",
    [
        "cor",
        "per",
    ],
)
def test_SD_Est(input_method: str) -> None:
    fs = 1000
    N = 1000
    Yall = np.random.rand(10, N)
    Yref = np.random.rand(5, N)
    nxseg = 1024
    pov = 0.5
    dt = 1 / fs
    freq, Sy = FDD_funct.SD_Est(Yall, Yref, dt, nxseg=nxseg, method=input_method, pov=pov)
    assert len(freq) > 0  # Ensure frequency array is not empty
    assert Sy.shape[0] == Yall.shape[0]  # Ensure correct shape of Sy


def test_FDD_MPE():
    # Generate some dummy data
    Sval = np.random.rand(2, 2, 1000)
    Svec = np.random.rand(2, 2, 1000)
    freq = np.linspace(0, 100, 1000)
    sel_freq = [25, 50, 75]
    DF = 0.1

    # Call the function with the dummy data
    Fn, Phi = FDD_funct.FDD_MPE(Sval, Svec, freq, sel_freq, DF)

    # Check that the output has the expected shape
    assert Fn.shape == (len(sel_freq),)
    assert Phi.shape == (Svec.shape[1], len(sel_freq))

    # Check that the output is a tuple
    assert isinstance(Fn, np.ndarray)
    assert isinstance(Phi, np.ndarray)


@pytest.mark.parametrize(
    "input_method",
    [
        "FSDD",
        "EFDD",
    ],
)
def test_SDOF_bellandMS(input_method: str) -> None:
    Nch = 3  # Number of channels
    Nf = 100  # Number of frequency points
    dt = 0.01  # Time interval of the data sampling
    sel_fn = 10.0  # Selected modal frequency
    cm = 1  # Number of close modes to consider in the analysis
    MAClim = 0.85  # Threshold for the Modal Assurance Criterion (MAC)
    DF = 1.0  # Frequency bandwidth around the selected frequency for analysis

    # Create a random spectral matrix
    Sy = np.random.rand(Nch, Nch, Nf) + 1j * np.random.rand(Nch, Nch, Nf)

    # Create a random mode shape
    phi_FDD = np.random.rand(Nch) + 1j * np.random.rand(Nch)
    phi_FDD /= np.linalg.norm(phi_FDD)  # Normalize the mode shape

    # Call the function with the dummy data
    SDOFbell1, SDOFms1 = FDD_funct.SDOF_bellandMS(
        Sy, dt, sel_fn, phi_FDD, input_method, cm, MAClim, DF
    )

    # Check that the output has the expected shape
    assert SDOFbell1.shape == (Sy.shape[2],)
    assert SDOFms1.shape == (Sy.shape[2], phi_FDD.shape[0])

    # Check that the output is a tuple
    assert isinstance(SDOFbell1, np.ndarray)
    assert isinstance(SDOFms1, np.ndarray)


@pytest.mark.parametrize(
    "input_method",
    [
        "cor",
        "paer",
    ],
)
def test_EFDD_MPE(input_method: str) -> None:
    # Sample data
    Sy = np.random.rand(3, 3, 100)
    freq = np.linspace(0, 1, 100)
    dt = 0.1
    sel_freq = [0.3, 0.5, 0.7]

    # Call the function
    Fn, Xi, Phi, PerPlot = FDD_funct.EFDD_MPE(
        Sy=Sy, freq=freq, dt=dt, sel_freq=sel_freq, methodSy=input_method, npmax=2
    )

    # Basic assertions to ensure the output has the correct shape
    assert Fn.shape[0] == len(sel_freq)
    assert Xi.shape[0] == len(sel_freq)
    assert Phi.shape == (Sy.shape[0], len(sel_freq))
    assert all([len(per_plot) == 9 for per_plot in PerPlot])
