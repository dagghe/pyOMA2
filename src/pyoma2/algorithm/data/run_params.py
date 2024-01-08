import typing
from pydantic import BaseModel, ConfigDict
import numpy as np
import numpy.typing as npt

## =============================================================================
## VECCHIO
## =============================================================================
# class BaseRunParams(BaseModel):
#     """Base class for input parameters."""
#     model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
#     fs: float
# 
# 
# class FDDRunParams(BaseRunParams):
#     nxseg: int = 1024
#     method_SD: str = "cor"
#     pov: float = 0.5
#     DF: float = 0.1
#     # # FIXME sel_freq può non essere messa qui, non è un run param ma un parametro ottenuto
#     # # dal metodo plot INTERATTIVO
#     # sel_freq: npt.NDArray[np.float32]
# 
# 
# class SSIRunParams(BaseRunParams):
#     method_hank: str  # "bias" ?
#     br: int
#     ref_ind: typing.Optional[list[int]] = None  # lista di indici ?
#     ordmin: int = 0
#     ordmax: typing.Optional[int] = None
#     step: int = 1
#     err_fn: float = 0.01
#     err_xi: float = 0.05
#     err_phi: float = 0.03
#     xi_max: float = 0.1
## =============================================================================
## NUOVO?
## =============================================================================
class FDDRunParams(BaseModel):
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
# METODO 1: run
    nxseg: int = 1024
    method_SD: str = "cor"
    # se method_SD = "per" allora ho anche bisogno di "pov" 
    # (eventualmente) le kwarg di scipy.signal.csd ?? tipo window ecc
    pov: float = 0.5
# FIXME
# METODO 2: MPE e MPE_fromPlot
    # io sel_freq lo metterei qui perche e un input per la funzione che viene
    # chiamata internamente quando viene usato MPE_fromPlot 
    sel_freq: npt.NDArray[np.float32]
    DF: float = 0.1
    # eventualmente parametri per plot vanno qui?
    freqlim: float | None = None,
    nSv: int | None = None,

class EFDDRunParams(BaseModel):
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
# METODO 1: run
    nxseg: int = 1024
    method_SD: str = "cor"
    # se method_SD = "per" allora ho anche bisogno di "pov" 
    # (eventualmente) le kwarg di scipy.signal.csd ?? tipo window ecc
    pov: float = 0.5
# METODO 2: MPE e MPE_fromPlot
    # io sel_freq lo metterei qui perche e un input per la funzione che viene
    # chiamata internamente quando viene usato MPE_fromPlot 
    sel_freq: npt.NDArray[np.float32]
    DF1: float = 0.1,
    DF2: float = 1.0,
    cm: int = 1,
    MAClim: float = 0.85,
    sppk: int = 3,
    npmax: int = 20,
    # eventualmente parametri per plot vanno qui?
    freqlim: float | None = None,
    nSv: int | None = None,



class SSIRunParams(BaseModel):
    method_hank: str
    br: int
    ref_ind: typing.Optional[list[int]] = None  # lista di indici ?
    ordmin: int = 0
    ordmax: typing.Optional[int] = None
    step: int = 1
    err_fn: float = 0.01
    err_xi: float = 0.05
    err_phi: float = 0.03
    xi_max: float = 0.1
# METODO 2: MPE e MPE_fromPlot
    # stessi motivi per sel_freq, vedi sopra
    sel_freq: npt.NDArray[np.float32]
    order_in: int | str = "find_min",
    deltaf: float = 0.05,
    rtol: float = 1e-2,
    # eventualmente parametri per plot vanno qui?
    freqlim: float | None = None


# class pLSCFRunParams(BaseModel):
#     # DA FARE
#     pass