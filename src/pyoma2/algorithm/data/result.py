import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

"""TODO fix type"""


class BaseResult(BaseModel):
    """Base class for output results."""
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
# dopo MPE o MPE_fromPlot
    Fn: npt.NDArray[np.float32] | None = None  # array of natural frequencies
    Phi: npt.NDArray[np.float32] | None = None  # array of Mode shape vectors


class FDDResult(BaseResult):
# dopo run
    freq: npt.NDArray[np.float32] | None = None
    Sy: npt.NDArray[np.float32] | None = None
    S_val: npt.NDArray[np.float32] | None = None
    S_vec: npt.NDArray[np.float32] | None = None

class EFDDResult(BaseResult):
# dopo run
    freq: npt.NDArray[np.float32] | None = None
    Sy: npt.NDArray[np.float32] | None = None
    S_val: npt.NDArray[np.float32] | None = None
    S_vec: npt.NDArray[np.float32] | None = None
# dopo MPE, MPE_forPlot
    Xi: npt.NDArray[np.float32] | None = None # array of damping ratios
    forPlot: list | None = None

class SSIResult(BaseResult):
# dopo run
    Fn_poles: npt.NDArray[np.float32] | None = None
    xi_poles: npt.NDArray[np.float32] | None = None
    Phi_poles: npt.NDArray[np.float32] | None = None
    # lam_poles: npt.NDArray[np.float32]
    Lab: npt.NDArray[np.float32] | None = None
# dopo MPE, MPE_forPlot
    Xi: npt.NDArray[np.float32] | None = None # array of damping ratios
    order_out: int | None = None
