from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


class FDDRunParams(BaseModel):
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # METODO 1: run
    nxseg: int = 1024
    method_SD: str = "per"
    pov: float = 0.5
    # se method_SD = "per" allora ho anche bisogno di "pov"
    # (eventualmente) le kwarg di scipy.signal.csd ?? tipo window ecc

    # METODO 2: MPE e MPE_fromPlot
    sel_freq: typing.Optional[npt.NDArray[np.float64]] = None
    DF: float = 0.1


class EFDDRunParams(BaseModel):
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # METODO 1: run
    nxseg: int = 1024
    method_SD: str = "per"
    pov: float = 0.5
    # se method_SD = "per" allora ho anche bisogno di "pov"
    # (eventualmente) le kwarg di scipy.signal.csd ?? tipo window ecc

    # METODO 2: MPE e MPE_fromPlot
    sel_freq: typing.Optional[npt.NDArray[np.float64]] = None
    DF1: float = (0.1,)
    DF2: float = (1.0,)
    cm: int = (1,)
    MAClim: float = (0.85,)
    sppk: int = (3,)
    npmax: int = (20,)


class SSIRunParams(BaseModel):
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # METODO 1: run
    br: int
    method: str = None
    ref_ind: typing.Optional[typing.List[int]] = None  # lista di indici ?
    ordmin: int = 0
    ordmax: typing.Optional[int] = None
    step: int = 1
    err_fn: float = 0.01
    err_xi: float = 0.05
    err_phi: float = 0.03
    xi_max: float = 0.1
    # METODO 2: MPE e MPE_fromPlot
    sel_freq: typing.Optional[typing.List[float]] = None
    order_in: typing.Union[int, str] = "find_min"
    deltaf: float = (0.05,)
    rtol: float = (1e-2,)


# class pLSCFRunParams(BaseModel):
#     # DA FARE
#     pass
