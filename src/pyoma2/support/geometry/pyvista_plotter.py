# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 21:25:39 2024

@author: dagpa
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
from numpy.typing import NDArray

from pyoma2.algorithms.data.result import BaseResult
from pyoma2.functions import gen
from pyoma2.support.geometry.data import Geometry2

from .plotter import BasePlotter

# Default visualization settings
_UNDEF_SETT: dict = {"color": "gray", "opacity": 0.7}
_DEF_MODE_SETT: dict = {"cmap": "plasma", "opacity": 0.7, "show_scalar_bar": False}
_UNDEF_MODE_SETT: dict = {"color": "gray", "opacity": 0.3}


class PvGeoPlotter(BasePlotter[Geometry2]):
    """
    Visualize and animate 3D mode shapes using PyVista.

    Attributes
    ----------
    geo : Geometry2
        Geometric model containing sensor positions and topology.
    res : Optional[BaseResult]
        Modal analysis results (mode shapes, frequencies).
    """

    def __init__(self, geo: Geometry2, res: Optional[BaseResult] = None):
        """
        Initialize the plotter.

        Parameters
        ----------
        geo : Geometry2
            Geometric model with sensor coordinates and connectivity.
        res : Optional[BaseResult], default=None
            Modal analysis results containing mode shapes and frequencies.

        Raises
        ------
        ImportError
            If PyVista or PyVistaQt is not installed.
        """
        super().__init__(geo, res)
        if pv is None or pvqt is None:
            logging.error("PyVista or PyVistaQt not available.")
            raise ImportError("Install 'pyvista' and 'pyvistaqt' to use PvGeoPlotter.")

    @staticmethod
    def _make_plotter(notebook: bool, background: bool) -> pv.Plotter:
        """
        Create a PyVista Plotter based on execution context.

        Parameters
        ----------
        notebook : bool
            Whether to create a notebook-compatible plotter.
        background : bool
            Whether to use a background Qt plotter.

        Returns
        -------
        pv.Plotter
            Configured Plotter instance.
        """
        if notebook:
            return pv.Plotter(notebook=True)
        return pvqt.BackgroundPlotter() if background else pv.Plotter()

    @staticmethod
    def _encode_mesh(
        points: NDArray[np.float64],
        lines_list: Optional[np.ndarray],
        faces_list: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Convert raw connectivity lists into PyVista-friendly arrays.

        Parameters
        ----------
        points : ndarray
            Point coordinates (n_points x 3).
        lines_list : Optional[list of int pairs]
            Sensor line connectivity.
        faces_list : Optional[list of int triplets]
            Surface face connectivity.

        Returns
        -------
        Tuple[Optional[ndarray], Optional[ndarray]]
            Encoded lines and faces arrays.
        """
        lines = None
        faces = None
        if lines_list is not None:
            lines = np.array([np.hstack(([2], line)) for line in lines_list], dtype=int)
        if faces_list is not None:
            faces = np.array([np.hstack(([3], face)) for face in faces_list], dtype=int)
        return lines, faces

    def _get_sensor_arrows(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]:
        """
        Compute arrow origins, directions, and labels for each sensor channel.

        Parameters
        ----------
        None

        Returns
        -------
        positions : ndarray
            Arrow start points (n_arrows x 3).
        directions : ndarray
            Arrow direction vectors (n_arrows x 3).
        labels : list of str
            Channel names corresponding to each arrow.
        """
        sens_map = self.geo.sens_map.to_numpy()
        sens_sign = self.geo.sens_sign.to_numpy()
        pts = self.geo.pts_coord.to_numpy()

        positions, directions, labels = [], [], []
        n_pts, n_axes = sens_map.shape
        for i in range(n_pts):
            for axis in range(n_axes):
                name = sens_map[i, axis]
                if isinstance(name, str) and name.lower() != "nan":
                    positions.append(pts[i])
                    vec = np.zeros(3, dtype=float)
                    vec[axis] = sens_sign[i, axis]
                    directions.append(vec)
                    labels.append(name)
        return np.array(positions), np.array(directions), labels

    def plot_geo(
        self,
        *,
        scaleF: float = 1.0,
        col_sens: str = "red",
        show_points: bool = True,
        points_sett: Optional[dict] = None,
        show_lines: bool = True,
        lines_sett: Optional[dict] = None,
        show_surf: bool = True,
        surf_sett: Optional[dict] = None,
        pl: Optional[pv.Plotter] = None,
        background: bool = True,
        notebook: bool = False,
    ) -> pv.Plotter:
        """
        Plot the raw geometry: points, lines, surfaces, and sensor arrows.

        Parameters
        ----------
        scaleF : float, default=1.0
            Scale factor for arrow length.
        col_sens : str, default="red"
            Color for sensor arrows.
        show_points : bool, default=True
            Whether to render sensor points.
        points_sett : dict or None, default=None
            Plot settings for points; falls back to default.
        show_lines : bool, default=True
            Whether to render sensor connection lines.
        lines_sett : dict or None, default=None
            Plot settings for lines; falls back to default.
        show_surf : bool, default=True
            Whether to render surface faces.
        surf_sett : dict or None, default=None
            Plot settings for surfaces; falls back to default.
        pl : pv.Plotter or None, default=None
            Existing plotter instance to use.
        background : bool, default=True
            Whether to use a background Qt plotter if creating new.
        notebook : bool, default=False
            Whether to create a notebook-compatible plotter.

        Returns
        -------
        pv.Plotter
            The configured plotter with raw geometry.
        """
        pl = pl or self._make_plotter(notebook, background)

        pts = self.geo.pts_coord.to_numpy()
        lines_arr, faces_arr = self._encode_mesh(
            pts, self.geo.sens_lines, self.geo.sens_surf
        )

        # Apply defaults
        points_sett = points_sett or _UNDEF_SETT.copy()
        lines_sett = lines_sett or _UNDEF_SETT.copy()
        surf_sett = surf_sett or _UNDEF_SETT.copy()

        if show_points:
            pl.add_points(pts, **points_sett)
        if show_lines and lines_arr is not None:
            pl.add_mesh(pv.PolyData(pts, lines=lines_arr), **lines_sett)
        if show_surf and faces_arr is not None:
            pl.add_mesh(pv.PolyData(pts, faces=faces_arr), **surf_sett)

        # Sensor arrows
        pos, dirs, labels = self._get_sensor_arrows()
        pl.add_arrows(pos, dirs, mag=scaleF, color=col_sens)
        pl.add_point_labels(
            pos + dirs * scaleF,
            labels,
            font_size=12,
            always_visible=True,
            shape_color="white",
        )

        pl.add_axes(line_width=2)
        pl.show()
        return pl

    def plot_mode(
        self,
        mode_nr: int = 1,
        scaleF: float = 1.0,
        show_lines: bool = True,
        show_surf: bool = True,
        def_sett: Optional[dict] = None,
        undef_sett: Optional[dict] = None,
        pl: Optional[pv.Plotter] = None,
        background: bool = True,
        notebook: bool = False,
    ) -> pv.Plotter:
        """
        Plot a single mode shape with optional undeformed geometry.

        Parameters
        ----------
        mode_nr : int, default=1
            Mode number to visualize (1-based).
        scaleF : float, default=1.0
            Scale factor for deformation amplitude.
        show_lines : bool, default=True
            Whether to render connection lines on mode shape.
        show_surf : bool, default=True
            Whether to render surface faces on mode shape.
        def_sett : dict or None, default=None
            Plot settings for deformed shape; falls back to default.
        undef_sett : dict or None, default=None
            Plot settings for undeformed shape; falls back to default.
        pl : pv.Plotter or None, default=None
            Existing plotter instance to use.
        background : bool, default=True
            Whether to use a background Qt plotter if creating new.
        notebook : bool, default=False
            Whether to create a notebook-compatible plotter.

        Raises
        ------
        ValueError
            If modal results (`res`) are not provided or mode_nr is out of range.

        Returns
        -------
        pv.Plotter
            The configured plotter with mode shape.
        """
        if self.res is None:
            raise ValueError("Modal result data is required to plot mode shapes.")
        n_modes = self.res.Phi.shape[1]
        if not 1 <= mode_nr <= n_modes:
            raise ValueError(f"mode_nr must be between 1 and {n_modes}")

        pl = pl or self._make_plotter(notebook, background)
        pts = self.geo.pts_coord.to_numpy()
        lines_arr, faces_arr = self._encode_mesh(
            pts, self.geo.sens_lines, self.geo.sens_surf
        )

        def_sett = def_sett or _DEF_MODE_SETT.copy()
        undef_sett = undef_sett or _UNDEF_MODE_SETT.copy()

        # Compute deformation
        phi = self.res.Phi[:, mode_nr - 1].real * scaleF
        df_map = gen.dfphi_map_func(
            phi, self.geo.sens_names, self.geo.sens_map, cstrn=self.geo.cstrn
        )
        new_pts = pts + df_map.to_numpy() * self.geo.sens_sign.to_numpy()

        # Undeformed
        pl.add_points(pts, **undef_sett)
        if show_lines and lines_arr is not None:
            pl.add_mesh(pv.PolyData(pts, lines=lines_arr), **undef_sett)
        if show_surf and faces_arr is not None:
            pl.add_mesh(pv.PolyData(pts, faces=faces_arr), **undef_sett)

        # Deformed with scalars
        pl.add_points(new_pts, scalars=df_map.values, **def_sett)
        if show_lines and lines_arr is not None:
            pl.add_mesh(
                pv.PolyData(new_pts, lines=lines_arr), scalars=df_map.values, **def_sett
            )
        if show_surf and faces_arr is not None:
            pl.add_mesh(
                pv.PolyData(new_pts, faces=faces_arr), scalars=df_map.values, **def_sett
            )

        freq = self.res.Fn[mode_nr - 1]
        pl.add_text(f"Mode {mode_nr}: {freq:.3f} Hz", position="upper_edge")
        pl.add_axes(line_width=2)
        pl.show()
        return pl

    def animate_mode(
        self,
        mode_nr: int = 1,
        scaleF: float = 1.0,
        show_lines: bool = True,
        show_surf: bool = True,
        def_sett: Optional[dict] = None,
        save_gif: bool = False,
        pl: Optional[pv.Plotter] = None,
    ) -> Union[pv.Plotter, str]:
        """
        Animate a mode shape oscillation. Optionally save as GIF.

        Parameters
        ----------
        mode_nr : int, default=1
            Mode number to animate (1-based).
        scaleF : float, default=1.0
            Scale factor for oscillation amplitude.
        show_lines : bool, default=True
            Whether to render connection lines during animation.
        show_surf : bool, default=True
            Whether to render surface faces during animation.
        def_sett : dict or None, default=None
            Plot settings for animation frames; falls back to default.
        save_gif : bool, default=False
            If True, saves animation as a GIF and returns its filepath.
        pl : pv.Plotter or None, default=None
            Existing plotter instance to use. If None, a new one is created.

        Raises
        ------
        ValueError
            If modal results (`res`) are not provided.

        Returns
        -------
        pv.Plotter or str
            Plotter instance for live animation, or filepath string if GIF saved.
        """
        if self.res is None:
            raise ValueError("Modal result data is required to animate mode shapes.")

        # Configure plotter
        new_plotter = False
        if pl is None:
            pl = pv.Plotter(off_screen=True) if save_gif else pvqt.BackgroundPlotter()
            new_plotter = True

        def_sett = def_sett or _DEF_MODE_SETT.copy()

        # Prepare geometry and scalars
        pts = self.geo.pts_coord.to_numpy()
        lines_arr, faces_arr = self._encode_mesh(
            pts, self.geo.sens_lines, self.geo.sens_surf
        )
        phi = self.res.Phi[:, mode_nr - 1].real * scaleF
        df_map = gen.dfphi_map_func(
            phi, self.geo.sens_names, self.geo.sens_map, cstrn=self.geo.cstrn
        )
        sens_sign = self.geo.sens_sign.to_numpy()

        # Initial displaced mesh
        pts_mesh = pv.PolyData(pts)
        base_disp = df_map.to_numpy() * sens_sign
        amps = np.linalg.norm(base_disp, axis=1)
        pts_mesh.point_data["amplitude"] = amps
        pl.add_mesh(pts_mesh, scalars="amplitude", **def_sett)

        # Optional line and surface meshes
        line_mesh = None
        if show_lines and lines_arr is not None:
            line_mesh = pv.PolyData(pts, lines=lines_arr)
            line_mesh.point_data["amplitude"] = amps
            pl.add_mesh(line_mesh, scalars="amplitude", **def_sett)
        face_mesh = None
        if show_surf and faces_arr is not None:
            face_mesh = pv.PolyData(pts, faces=faces_arr)
            face_mesh.point_data["amplitude"] = amps
            pl.add_mesh(face_mesh, scalars="amplitude", **def_sett)

        # Annotation
        freq = self.res.Fn[mode_nr - 1]
        pl.add_text(f"Mode {mode_nr}: {freq:.3f} Hz", position="upper_edge")
        pl.add_axes(line_width=2)

        # Animation loop
        n_frames = 30
        frames = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
        idx = {"frame": 0}

        def _update():
            phase = frames[idx["frame"]]
            disp = base_disp * np.cos(phase)
            new_pts = pts + disp
            inst_amp = np.linalg.norm(disp, axis=1)

            pts_mesh.points = new_pts
            pts_mesh.point_data["amplitude"] = inst_amp
            if line_mesh is not None:
                line_mesh.points = new_pts
                line_mesh.point_data["amplitude"] = inst_amp
            if face_mesh is not None:
                face_mesh.points = new_pts
                face_mesh.point_data["amplitude"] = inst_amp

            pl.render()
            idx["frame"] = (idx["frame"] + 1) % n_frames

        if save_gif:
            gif_path = f"Mode_{mode_nr}.gif"
            pl.open_gif(gif_path)
            for _ in range(n_frames):
                _update()
                pl.write_frame()
            return gif_path
        else:
            pl.add_callback(_update, interval=100)
            if new_plotter:
                pl.show()
            return pl
