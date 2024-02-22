"""
This module provides classes for handling and storing various types of results data
related to the pyOMA2 module.
"""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

"""TODO fix type"""


class BaseResult(BaseModel):
    """Base class for storing results data."""

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # dopo MPE o MPE_fromPlot
    Fn: typing.Optional[npt.NDArray[np.float64]] = None  # array of natural frequencies
    Phi: typing.Optional[npt.NDArray[np.float64]] = None  # array of Mode shape vectors


class FDDResult(BaseResult):
    """Class for storing FDD results data."""

    freq: typing.Optional[npt.NDArray[np.float64]] = None
    Sy: typing.Optional[npt.NDArray[np.float64]] = None
    S_val: typing.Optional[npt.NDArray[np.float64]] = None
    S_vec: typing.Optional[npt.NDArray[np.float64]] = None


class EFDDResult(BaseResult):
    """Class for storing EFDD/FSDD results data."""

    freq: typing.Optional[npt.NDArray[np.float64]] = None
    Sy: typing.Optional[npt.NDArray[np.float64]] = None
    S_val: typing.Optional[npt.NDArray[np.float64]] = None
    S_vec: typing.Optional[npt.NDArray[np.float64]] = None
    # dopo MPE, MPE_forPlot
    Xi: typing.Optional[npt.NDArray[np.float64]] = None  # array of damping ratios
    forPlot: typing.Optional[typing.List] = None


class SSIResult(BaseResult):
    """Class for storing SSIcov/dat results data."""

    A: typing.Optional[typing.List[npt.NDArray[np.float64]]] = None
    C: typing.Optional[typing.List[npt.NDArray[np.float64]]] = None
    H: typing.Optional[npt.NDArray[np.float64]] = None

    Fn_poles: typing.Optional[npt.NDArray[np.float64]] = None
    xi_poles: typing.Optional[npt.NDArray[np.float64]] = None
    Phi_poles: typing.Optional[npt.NDArray[np.float64]] = None
    # lam_poles: npt.NDArray[np.float64]
    Lab: typing.Optional[npt.NDArray[np.float64]] = None
    # dopo MPE, MPE_forPlot
    Xi: typing.Optional[npt.NDArray[np.float64]] = None  # array of damping ratios
    order_out: typing.Union[typing.List[int], int, None] = None


class pLSCFResult(BaseResult):
    """Class for storing pLSCF results data."""

    freq: typing.Optional[npt.NDArray[np.float64]] = None
    Sy: typing.Optional[npt.NDArray[np.float64]] = None
    Ad: typing.Optional[list[npt.NDArray[np.float64]]] = None
    Bn: typing.Optional[list[npt.NDArray[np.float64]]] = None
    Fn_poles: typing.Optional[npt.NDArray[np.float64]] = None
    xi_poles: typing.Optional[npt.NDArray[np.float64]] = None
    Phi_poles: typing.Optional[npt.NDArray[np.float64]] = None
    # lam_poles: npt.NDArray[np.float64]
    Lab: typing.Optional[npt.NDArray[np.float64]] = None
    # dopo MPE, MPE_forPlot
    Xi: typing.Optional[npt.NDArray[np.float64]] = None  # array of damping ratios
    order_out: typing.Union[typing.List[int], int, None] = None


class MsPoserResult(BaseModel):
    # FIXME non molto corretto che sia sotto la cartella algorithms..
    # valutare se creare una cartella apposita per i setups
    """Base class for MultiSetup Poser result"""
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    Phi: npt.NDArray[np.float64]
    Fn: npt.NDArray[np.float64]
    Fn_cov: npt.NDArray[np.float64]
    Xi: npt.NDArray[np.float64]
    Xi_cov: npt.NDArray[np.float64]
