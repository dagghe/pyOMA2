from __future__ import annotations

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
    sens_names: list[str]
    # OPTIONAL
    sens_lines: npt.NDArray[np.int64] | None = None  # lines between sensors
    bg_nodes: npt.NDArray[np.float64] | None = None  # Background nodes
    bg_lines: npt.NDArray[np.int64] | None = None  # Background lines
    bg_surf: npt.NDArray[np.int64] | None = None  # Background surfaces


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
    cstrn: pd.DataFrame | None = None
    sens_sign: pd.DataFrame | None = None  # sensors sign
    sens_surf: npt.NDArray[np.int64] | None = None  # surfaces between sensors


class ModeGeo1Data(BaseModel):
    """
    Headless mode-shape geometry data for a Geometry1 setup.

    Carries the complete set of arrays a geo1 mode-shape renderer needs:
    sensor coordinates and directions, the per-sensor real modal displacement,
    the deformed coordinates, plus the connectivity and background elements
    echoed from the geometry. Built by
    :func:`pyoma2.support.geometry.mode_data.build_mode_geo1_data` and returned
    by :meth:`pyoma2.support.geometry.mixin.GeometryMixin.get_mode_geo1_data`.

    Attributes
    ----------
    sens_names : list of str
        Names of the sensors; row order matches every ``(Nch, ...)`` array.
    sens_coord : numpy.ndarray of shape (Nch, 3)
        Undeformed sensor coordinates (x, y, z).
    sens_dir : numpy.ndarray of shape (Nch, 3)
        Per-sensor measurement direction.
    phi : numpy.ndarray of shape (Nch,)
        Real part of the mode shape for ``mode_nr`` (unscaled).
    mode_displ : numpy.ndarray of shape (Nch, 3)
        Per-sensor modal displacement, ``sens_dir * phi`` (unscaled — this is
        the raw quiver vector; ``scaleF`` is applied only in ``deformed_coord``).
    deformed_coord : numpy.ndarray of shape (Nch, 3)
        Deformed sensor coordinates, ``sens_coord + mode_displ * scaleF``.
    fn : float
        Natural frequency of ``mode_nr`` (Hz).
    mode_nr : int
        Mode number (1-based).
    scaleF : float
        Displacement scale factor applied in ``deformed_coord``.
    sens_lines : numpy.ndarray of shape (n, 2), optional
        Sensor-index pairs forming connection lines (0-indexed). Default None.
    bg_nodes : numpy.ndarray of shape (m, 3), optional
        Background node coordinates. Default None.
    bg_lines : numpy.ndarray of shape (p, 2), optional
        Background node-index pairs. Default None.
    bg_surf : numpy.ndarray of shape (q, 3), optional
        Background surface node-index triplets. Default None.
    """

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # MANDATORY
    sens_names: list[str]
    sens_coord: npt.NDArray[np.float64]
    sens_dir: npt.NDArray[np.int64]
    phi: npt.NDArray[np.float64]
    mode_displ: npt.NDArray[np.float64]
    deformed_coord: npt.NDArray[np.float64]
    fn: float
    mode_nr: int
    scaleF: float
    # OPTIONAL
    sens_lines: npt.NDArray[np.int64] | None = None
    bg_nodes: npt.NDArray[np.float64] | None = None
    bg_lines: npt.NDArray[np.int64] | None = None
    bg_surf: npt.NDArray[np.int64] | None = None


class ModeGeo2Data(BaseModel):
    """
    Headless mode-shape geometry data for a Geometry2 setup.

    Carries the complete set of arrays the geo2 mode-shape renderers assemble:
    point coordinates, the sensor mapping and signs, the mapped modal values,
    the per-point modal displacement, the deformed coordinates, the
    displacement magnitude, plus connectivity and background elements. Built by
    :func:`pyoma2.support.geometry.mode_data.build_mode_geo2_data` and returned
    by :meth:`pyoma2.support.geometry.mixin.GeometryMixin.get_mode_geo2_data`.

    Attributes
    ----------
    sens_names : list of str
        Names of the sensors.
    pts_coord : numpy.ndarray of shape (P, 3)
        Undeformed point coordinates (x, y, z).
    sens_map : pandas.DataFrame of shape (P, 3)
        Per point/axis: a sensor name, a constraint name, or 0.
    sens_sign : pandas.DataFrame of shape (P, 3)
        Per point/axis sign (+1 / -1 / 0).
    phi : numpy.ndarray of shape (Nch,)
        Real part of the mode shape for ``mode_nr``, pre-scaled by ``scaleF``.
    df_phi_map : pandas.DataFrame of shape (P, 3)
        ``phi`` mapped onto each point/axis via ``gen.dfphi_map_func``.
    mode_displ : numpy.ndarray of shape (P, 3)
        Per-point modal displacement, ``df_phi_map * sens_sign`` (scaled).
    deformed_coord : numpy.ndarray of shape (P, 3)
        Deformed point coordinates, ``pts_coord + mode_displ``.
    displ_magnitude : numpy.ndarray of shape (P,)
        Per-point displacement magnitude, ``norm(mode_displ, axis=1)``.
    fn : float
        Natural frequency of ``mode_nr`` (Hz).
    mode_nr : int
        Mode number (1-based).
    scaleF : float
        Displacement scale factor pre-multiplied into ``phi`` / ``mode_displ``.
    cstrn : pandas.DataFrame, optional
        Constraint matrix (index = constraint names, columns = sensor names).
        Default None.
    sens_lines : numpy.ndarray of shape (n, 2), optional
        Point-index pairs forming connection lines (0-indexed). Default None.
    sens_surf : numpy.ndarray of shape (p, 3), optional
        Point-index triplets forming surface faces (0-indexed). Default None.
    bg_nodes : numpy.ndarray of shape (m, 3), optional
        Background node coordinates. Default None.
    bg_lines : numpy.ndarray of shape (p, 2), optional
        Background node-index pairs. Default None.
    bg_surf : numpy.ndarray of shape (q, 3), optional
        Background surface node-index triplets. Default None.
    """

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    # MANDATORY
    sens_names: list[str]
    pts_coord: npt.NDArray[np.float64]
    sens_map: pd.DataFrame
    sens_sign: pd.DataFrame
    phi: npt.NDArray[np.float64]
    df_phi_map: pd.DataFrame
    mode_displ: npt.NDArray[np.float64]
    deformed_coord: npt.NDArray[np.float64]
    displ_magnitude: npt.NDArray[np.float64]
    fn: float
    mode_nr: int
    scaleF: float
    # OPTIONAL
    cstrn: pd.DataFrame | None = None
    sens_lines: npt.NDArray[np.int64] | None = None
    sens_surf: npt.NDArray[np.int64] | None = None
    bg_nodes: npt.NDArray[np.float64] | None = None
    bg_lines: npt.NDArray[np.int64] | None = None
    bg_surf: npt.NDArray[np.int64] | None = None
