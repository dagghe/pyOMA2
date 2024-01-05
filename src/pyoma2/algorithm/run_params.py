import typing

from pydantic import BaseModel, ConfigDict

import numpy as np
import numpy.typing as npt

class BaseRunParams(BaseModel):
    """Base class for input parameters."""
    model_config = ConfigDict(from_attributes=True)
    fs: float 


class FDDRunParams(BaseRunParams):
    nxseg: int = 1024
    method_SD: str = "cor"
    pov: float = 0.5
    DF: float = 0.1
    sel_freq: npt.NDArray[np.float32]


class SSIRunParams(BaseRunParams):
    method_hank: str 
    br: int
    ref_id: typing.Optional[list[int]] = None  # lista di indici ?
    ordmin: int = 0
    ordmax: typing.Optional[int] = None
    step: int = 1
    err_fn: float = 0.01
    err_xi: float = 0.05
    err_phi: float = 0.03
    xi_max: float = 0.1

