from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

"""TODO fix type"""


@dataclass
class BaseResult:
    Fn: npt.NDArray[np.float32] # array of natural frequencies
    Phi: npt.NDArray[np.float32] # array of Mode shape vectors
    # Xi: npt.NDArray[np.float32] # array of damping ratios


@dataclass
class FDDResult(BaseResult):
    freq: npt.NDArray[np.float32]
    Sy: npt.NDArray[np.float32]
    S_val: npt.NDArray[np.float32]
    S_vec: npt.NDArray[np.float32]

    Freq_ind: npt.NDArray[np.float32]
    sel_freq: npt.NDArray[np.float32]


@dataclass
class EFDDResult(FDDResult):
    Xi: npt.NDArray[np.float32] # array of damping ratios
    forPlot: list


@dataclass
class SSIcovResult(BaseResult):
    Fn_poles: npt.NDArray[np.float32]
    xi_poles: npt.NDArray[np.float32]
    Phi_poles: npt.NDArray[np.float32]
    lam_poles: npt.NDArray[np.float32]


@dataclass
class SSIdatResult(SSIcovResult):
    pass


@dataclass
class pLSCFResult(SSIcovResult):
    pass
