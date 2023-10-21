from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

"""TODO fix type"""


@dataclass
class BaseResult:
    df: int
    S_val: int
    S_vec: int
    freqs: int
    Fn: int
    Phi: int
    Freq_ind: int
    PSD_matr: int


@dataclass
class FDDResult(BaseResult):
    pass


@dataclass
class EFDDResult(BaseResult):
    Method: int
    xi: int
    Figs: int


@dataclass
class SSIcovResult(BaseResult):
    Method: str
    Fn_poles: npt.NDArray[np.float32]
    xi_poles: npt.NDArray[np.float32]
    Phi_poles: npt.NDArray[np.float32]
    lam_poles: npt.NDArray[np.float32]


@dataclass
class SSIdatResult(SSIcovResult):
    pass


@dataclass
class pLSCFResult(SSIcovResult):
    ordmax: int
