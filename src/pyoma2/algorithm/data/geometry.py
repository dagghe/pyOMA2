"""
Created on Mon Jan  8 10:30:00 2024

@author: dpa
"""
from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict

"""TODO fix type"""


class Geometry1(BaseModel):
    """Base class for output results."""

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # # MANDATORY
    sens_names: typing.List[str]
    sens_coord: typing.Union[
        pd.DataFrame, npt.NDArray[np.float64]
    ]  # sensors' coordinates
    sens_dir: npt.NDArray[np.float64]  # sensors' directions
    # # OPTIONAL
    sens_lines: typing.Optional[npt.NDArray[np.int64]] = None  # lines connecting sensors
    bg_nodes: typing.Optional[npt.NDArray[np.float64]] = None  # Background nodes
    bg_lines: typing.Optional[npt.NDArray[np.int64]] = None  # Background lines
    bg_surf: typing.Optional[npt.NDArray[np.float64]] = None  # Background surfaces


class Geometry2(BaseModel):
    """Base class for output results."""

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # # MANDATORY
    sens_names: typing.List[str]  # sensors' names
    pts_coord: typing.Union[pd.DataFrame, npt.NDArray[np.float64]]  # points' coordinates
    sens_map: typing.Union[pd.DataFrame, npt.NDArray[np.float64]]  # mapping
    sens_sign: typing.Union[pd.DataFrame, npt.NDArray[np.float64]]  # sign

    # # OPTIONAL
    # Order reduction uno tra ["None", "xy","xz","yz","x","y","z"]
    order_red: typing.Optional[str] = None
    sens_lines: typing.Optional[npt.NDArray[np.int64]] = None  # lines connection sensors
    bg_nodes: typing.Optional[npt.NDArray[np.float64]] = None  # Background nodes
    bg_lines: typing.Optional[npt.NDArray[np.int64]] = None  # Background lines
    bg_surf: typing.Optional[npt.NDArray[np.float64]] = None  # Background surfaces
