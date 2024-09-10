from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict


class BaseGeometry(BaseModel):
    """
    Base class for storing and managing sensor and background geometry data.

    Attributes
    ----------
    sens_names : List[str]
        Names of the sensors.
    sens_lines : numpy.ndarray of shape (n, 2), optional
        An array representing lines between sensors, where each entry is a pair of
        sensor indices. Default is None.
    bg_nodes : numpy.ndarray of shape (m, 3), optional
        An array of background nodes in 3D space. Default is None.
    bg_lines : numpy.ndarray of shape (p, 2), optional
        An array of lines between background nodes, where each entry is a pair of
        node indices. Default is None.
    bg_surf : numpy.ndarray of shape (q, 3), optional
        An array of background surfaces, where each entry is a node index.
        Default is None.
    """

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # MANDATORY
    sens_names: typing.List[str]
    # OPTIONAL
    sens_lines: typing.Optional[npt.NDArray[np.int64]] = None  # lines between sensors
    bg_nodes: typing.Optional[npt.NDArray[np.float64]] = None  # Background nodes
    bg_lines: typing.Optional[npt.NDArray[np.int64]] = None  # Background lines
    bg_surf: typing.Optional[npt.NDArray[np.int64]] = None  # Background surfaces


class Geometry1(BaseGeometry):
    """
    Class for storing and managing sensor and background geometry data for Geometry 1.

    This class provides a structured way to store the coordinates and directions of
    sensors, as well as optional background data such as nodes, lines, and surfaces.

    Attributes
    ----------
    sens_names : List[str]
        Names of the sensors.
    sens_coord : pandas.DataFrame
        A DataFrame containing the coordinates of each sensor.
    sens_dir : numpy.ndarray of shape (n, 3)
        An array representing the direction vectors of the sensors.
    sens_lines : numpy.ndarray of shape (n, 2), optional
        An array representing lines between sensors, where each entry is a pair of
        sensor indices. Default is None.
    bg_nodes : numpy.ndarray of shape (m, 3), optional
        An array of background nodes in 3D space. Default is None.
    bg_lines : numpy.ndarray of shape (p, 2), optional
        An array of lines between background nodes, where each entry is a pair of
        node indices. Default is None.
    bg_surf : numpy.ndarray of shape (q, 3), optional
        An array of background surfaces, where each entry is a node index.
        Default is None.
    """

    # MANDATORY
    sens_coord: pd.DataFrame  # sensors' coordinates
    sens_dir: npt.NDArray[np.int64]  # sensors' directions


class Geometry2(BaseGeometry):
    """
    Class for storing and managing sensor and background geometry data for Geometry 2.

    This class is designed to store sensor data and their associated point coordinates,
    along with optional constraints and sensor surface information. It supports mapping
    sensors to specific points in space, and includes optional background data for
    further geometric analysis.

    Attributes
    ----------
    sens_names : List[str]
        Names of the sensors.
    pts_coord : pandas.DataFrame
        A DataFrame containing the coordinates of the points.
    sens_map : pandas.DataFrame
        A DataFrame mapping sensors to points locations.
    cstrn : pandas.DataFrame, optional
        A DataFrame of constraints applied to the points or sensors. Default is None.
    sens_sign : pandas.DataFrame, optional
        A DataFrame indicating the sign or orientation of the sensors. Default is None.
    sens_lines : numpy.ndarray of shape (n, 2), optional
        An array representing lines between sensors, where each entry is a pair of
        sensor indices. Default is None.
    sens_surf : numpy.ndarray of shape (p, ?), optional
        An array representing surfaces between sensors, where each entry is a node index.
        Default is None.
    bg_nodes : numpy.ndarray of shape (m, 3), optional
        An array of background nodes in 3D space. Default is None.
    bg_lines : numpy.ndarray of shape (p, 2), optional
        An array of lines between background nodes, where each entry is a pair of
        node indices. Default is None.
    bg_surf : numpy.ndarray of shape (q, 3), optional
        An array of background surfaces, where each entry is a node index.
        Default is None.
    """

    # MANDATORY
    pts_coord: pd.DataFrame  # points' coordinates
    sens_map: pd.DataFrame  # mapping sensors to points
    # OPTIONAL
    cstrn: typing.Optional[pd.DataFrame] = None
    sens_sign: typing.Optional[pd.DataFrame] = None  # sensors sign
    sens_surf: typing.Optional[npt.NDArray[np.int64]] = None  # surfaces between sensors
