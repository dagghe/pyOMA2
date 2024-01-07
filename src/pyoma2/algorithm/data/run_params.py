import typing

from pydantic import BaseModel, ConfigDict


class BaseRunParams(BaseModel):
    """Base class for input parameters."""

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    fs: float


class FDDRunParams(BaseRunParams):
    nxseg: int = 1024
    method_SD: str = "cor"
    pov: float = 0.5
    DF: float = 0.1
    # # FIXME sel_freq può non essere messa qui, non è un run param ma un parametro ottenuto
    # # dal metodo plot INTERATTIVO
    # sel_freq: npt.NDArray[np.float32]


class SSIRunParams(BaseRunParams):
    method_hank: str  # "bias" ?
    br: int
    ref_ind: typing.Optional[list[int]] = None  # lista di indici ?
    ordmin: int = 0
    ordmax: typing.Optional[int] = None
    step: int = 1
    err_fn: float = 0.01
    err_xi: float = 0.05
    err_phi: float = 0.03
    xi_max: float = 0.1
