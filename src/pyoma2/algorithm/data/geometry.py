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
    sens_coord: pd.DataFrame  # sensors' coordinates
    sens_dir: npt.NDArray[np.int64]  # sensors' directions
    # OPTIONAL
    sens_lines: typing.Optional[npt.NDArray[np.int64]] = None  # lines between sensors
    bg_nodes: typing.Optional[npt.NDArray[np.float64]] = None  # Background nodes
    bg_lines: typing.Optional[npt.NDArray[np.int64]] = None  # Background lines
    bg_surf: typing.Optional[npt.NDArray[np.int64]] = None  # Background surfaces


class Geometry2(BaseModel):
    """Class for storing Geometry 2 data."""

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # MANDATORY
    sens_names: typing.List[str]  # sensors' names
    pts_coord: pd.DataFrame  # points' coordinates
    sens_map: pd.DataFrame  # mapping sensors to points
    # OPTIONAL
    cstrn: typing.Optional[pd.DataFrame] = None
    sens_sign: typing.Optional[pd.DataFrame] = None  # sensors sign
    sens_lines: typing.Optional[npt.NDArray[np.int64]] = None  # lines between sensors
    sens_surf: typing.Optional[npt.NDArray[np.int64]] = None  # surfaces between sensors
    bg_nodes: typing.Optional[npt.NDArray[np.float64]] = None  # Background nodes
    bg_lines: typing.Optional[npt.NDArray[np.int64]] = None  # Background lines
    bg_surf: typing.Optional[npt.NDArray[np.int64]] = None  # Background surfaces
