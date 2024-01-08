import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

"""TODO fix type"""


class BaseResult(BaseModel):
    """Base class for output results."""
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # dopo MPE o MPE_fromPlot
    Fn: npt.NDArray[np.float32]  # array of natural frequencies
    Phi: npt.NDArray[np.float32]  # array of Mode shape vectors


class FDDResult(BaseResult):
    # dopo run
    freq: npt.NDArray[np.float32]
    Sy: npt.NDArray[np.float32]
    S_val: npt.NDArray[np.float32]
    S_vec: npt.NDArray[np.float32]

class EFDDResult(BaseResult):
    # dopo run
    freq: npt.NDArray[np.float32]
    Sy: npt.NDArray[np.float32]
    S_val: npt.NDArray[np.float32]
    S_vec: npt.NDArray[np.float32]
    # dopo MPE, MPE_forPlot
    Xi: npt.NDArray[np.float32]  # array of damping ratios
    forPlot: list

class SSIResult(BaseResult):
    # dopo run
    Fn_poles: npt.NDArray[np.float32]
    xi_poles: npt.NDArray[np.float32]
    Phi_poles: npt.NDArray[np.float32]
    # lam_poles: npt.NDArray[np.float32]
    Lab: npt.NDArray[np.float32]
    # dopo MPE, MPE_forPlot
    Xi: npt.NDArray[np.float32]  # array of damping ratios
    order_out: int
