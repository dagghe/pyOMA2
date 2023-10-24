import typing
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

@dataclass
class BaseRunParams(BaseModel):
    """Base class for input parameters."""
    model_config = ConfigDict(from_attributes=True)

@dataclass
class FDDRunParams(BaseRunParams):
    nxseg: int = 1024
    method_SD: str = "cor"
#    pov: float = 0.5
#    window: str = "hann"

@dataclass
class SSIdatRunParams(BaseRunParams):
    br: int
    ref_id: typing.Optional[list[int]] = None # lista di indici ?
    ordmin: int = 0
    ordmax: typing.Optional[int] = None
    step: int = 1
    err_fn: float = 0.01
    err_xi: float = 0.05
    err_phi: float = 0.03
    xi_max: float = 0.1

@dataclass
class SSIcovRunParams(SSIdatRunParams):
    method_hank: str = "bias"



 

