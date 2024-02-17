"""
This module provides classes for handling geometry-related data, specifically designed
to store and manipulate sensor and background geometry information. It includes two main
classes: Geometry1 and Geometry2, each tailored for different types of geometric data.
"""
from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict

"""TODO fix type"""


class Geometry1(BaseModel):
    """Class for storing Geometry 1 data."""

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # MANDATORY
    sens_names: typing.List[str]
    sens_coord: typing.Union[
        pd.DataFrame, npt.NDArray[np.float64]
    ]  # sensors' coordinates
    sens_dir: npt.NDArray[np.float64]  # sensors' directions
    # OPTIONAL
    sens_lines: typing.Optional[npt.NDArray[np.int64]] = None  # lines connecting sensors
    bg_nodes: typing.Optional[npt.NDArray[np.float64]] = None  # Background nodes
    bg_lines: typing.Optional[npt.NDArray[np.int64]] = None  # Background lines
    bg_surf: typing.Optional[npt.NDArray[np.float64]] = None  # Background surfaces


class Geometry2(BaseModel):
    """Class for storing Geometry 2 data."""

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # MANDATORY
    sens_names: typing.List[str]  # sensors' names
    pts_coord: typing.Union[pd.DataFrame, npt.NDArray[np.float64]]  # points' coordinates
    sens_map: typing.Union[pd.DataFrame, npt.NDArray[np.float64]]  # mapping
    sens_sign: typing.Union[pd.DataFrame, npt.NDArray[np.float64]]  # sign

    # # OPTIONAL
    # FIXME x DIEGO si puo fare cosi qui?
    # order_red: typing.Literal["xy","xz","yz","x","y","z"] = None
    order_red: typing.Optional[str] = None  # string to reduce the n* of DOF per point
    sens_lines: typing.Optional[npt.NDArray[np.int64]] = None  # lines connection sensors
    bg_nodes: typing.Optional[npt.NDArray[np.float64]] = None  # Background nodes
    bg_lines: typing.Optional[npt.NDArray[np.int64]] = None  # Background lines
    bg_surf: typing.Optional[npt.NDArray[np.float64]] = None  # Background surfaces
