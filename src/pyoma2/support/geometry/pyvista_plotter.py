# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 21:25:39 2024

@author: dagpa
"""

import typing
import warnings

import numpy as np

from pyoma2.support.geometry.data import Geometry2

# import numpy.typing as npt
try:
    import pyvista as pv
    import pyvistaqt as pvqt
except ImportError:
    warnings.warn(
        "Optional package 'pyvista' is not installed. Some features may not be available.",
        ImportWarning,
        stacklevel=2,
    )
    warnings.warn(
        "Install 'pyvista' with 'pip install pyvista' or 'pip install pyoma_2[pyvista]'",
        ImportWarning,
        stacklevel=2,
    )
    pv = None
    pvqt = None
from pyoma2.algorithms.data.result import BaseResult
from pyoma2.functions import gen

from .plotter import BasePlotter

if typing.TYPE_CHECKING:
    from pyoma2.support.geometry import Geometry2


class PvGeoPlotter(BasePlotter[Geometry2]):
    """
    A class to visualize and animate mode shapes in 3D using `pyvista`.

    This class provides methods for plotting geometry, mode shapes, and animating
    mode shapes, utilizing the `pyvista` and `pyvistaqt` libraries for visualization.

    Parameters
    ----------
    geo : Geometry2
        The geometric data of the model, which includes sensor coordinates and other
        structural information.
    Res : Union[BaseResult, MsPoserResult], optional
        The result data containing mode shapes and frequency data (default is None).

    Raises
    ------
    ImportError
        If `pyvista` or `pyvistaqt` are not installed, an error is raised when attempting
        to instantiate the class.
    """

    def __init__(self, geo: Geometry2, res: typing.Optional[BaseResult] = None):
        """
        Initialize the class with geometric and result data.
        Ensure that the `pyvista` and `pyvistaqt` libraries are installed.
        """
        super().__init__(geo, res)
        if pv is None or pvqt is None:
            raise ImportError(
                "Optional package 'pyvista' is not installed. Some features may not be available."
                "Install 'pyvista' with 'pip install pyvista' or 'pip install pyoma_2[pyvista]'"
            )

    def plot_geo(
        self,
        scaleF=1,
        col_sens="red",
        plot_points=True,
        points_sett="default",
        plot_lines=True,
        lines_sett="default",
        plot_surf=True,
        surf_sett="default",
        pl=None,
        bg_plotter: bool = True,
        notebook: bool = False,
    ):
        """
        Plot the 3D geometry of the model, including points, lines, and surfaces.

        Parameters
        ----------
        scaleF : float, optional
            Scale factor for the sensor vectors (default is 1).
        col_sens : str, optional
            Color for the sensor points and arrows (default is 'red').
        plot_points : bool, optional
            Whether to plot sensor points (default is True).
        points_sett : dict or str, optional
            Settings for plotting points (default is 'default', which applies preset settings).
        plot_lines : bool, optional
            Whether to plot lines representing connections between sensors (default is True).
        lines_sett : dict or str, optional
            Settings for plotting lines (default is 'default', which applies preset settings).
        plot_surf : bool, optional
            Whether to plot surfaces (default is True).
        surf_sett : dict or str, optional
            Settings for plotting surfaces (default is 'default', which applies preset settings).
        pl : pyvista.Plotter, optional
            Existing plotter instance to use (default is None, which creates a new plotter).
        bg_plotter : bool, optional
            Whether to use a background plotter for visualization (default is True).
        notebook : bool, optional
            If True, a plotter for use in Jupyter notebooks is created (default is False).

        Returns
        -------
        pyvista.Plotter
            The plotter object used for visualization.

        Notes
        -----
        If `pyvistaqt` is used, a background plotter will be created. If running in
        a notebook environment, a `pyvista` plotter with notebook support is used.
        """
        # import geometry
        geo = self.geo

        # define the plotter object type
        if pl is None:
            if notebook:
                pl = pv.Plotter(notebook=True)
            elif bg_plotter:
                pl = pvqt.BackgroundPlotter()
            else:
                pl = pv.Plotter()

        # define default settings for plot
        undef_sett = dict(
            color="gray",
            opacity=0.7,
        )

        if points_sett == "default":
            points_sett = undef_sett

        if lines_sett == "default":
            lines_sett = undef_sett

        if surf_sett == "default":
            surf_sett = undef_sett

        # GEOMETRY
        points = geo.pts_coord.to_numpy()
        lines = geo.sens_lines
        surfs = geo.sens_surf
        # geometry in pyvista format
        if lines is not None:
            lines = np.array([np.hstack([2, line]) for line in lines])
        if surfs is not None:
            surfs = np.array([np.hstack([3, surf]) for surf in surfs])

        # PLOTTING
        if plot_points:
            pl.add_points(points, **points_sett)
        if plot_lines:
            line_mesh = pv.PolyData(points, lines=lines)
            pl.add_mesh(line_mesh, **lines_sett)
        if plot_surf:
            face_mesh = pv.PolyData(points, faces=surfs)
            pl.add_mesh(face_mesh, **surf_sett)

        # # Add axes
        # pl.add_axes(line_width=5, labels_off=False)
        # pl.show()

        # add sensor points + arrows for direction
        sens_names = geo.sens_names
        ch_names = geo.sens_map.to_numpy()
        ch_names = np.array(
            [name if name in sens_names else "nan" for name in ch_names.flatten()]
        ).reshape(ch_names.shape)

        ch_names_fl = ch_names.flatten()[ch_names.flatten() != "nan"]
        ch_names_fl = [str(ele) for ele in ch_names_fl]
        # Plot points where ch_names_1 is not np.nan
        valid_indices = ch_names != "nan"  # FIXME
        valid_points = points[np.any(valid_indices, axis=1)]

        pl.add_points(
            valid_points,
            render_points_as_spheres=True,
            color=col_sens,
            point_size=10,
        )

        points_new = []
        directions = []
        for i, (row1, row2) in enumerate(zip(ch_names, points)):
            for j, elem in enumerate(row1):
                if elem != "nan":
                    vector = [0, 0, 0]
                    # vector[j] = 1
                    vector[j] = geo.sens_sign.values[i, j]
                    directions.append(vector)
                    points_new.append(row2)

        points_new = np.array(points_new)
        directions = np.array(directions)
        # Add arrow to plotter
        pl.add_arrows(points_new, directions, mag=scaleF, color=col_sens)
        pl.add_point_labels(
            points_new + directions * scaleF,
            ch_names_fl,
            font_size=20,
            always_visible=True,
            shape_color="white",
        )

        # Add axes
        pl.add_axes(line_width=5, labels_off=False)
        pl.show()

        return pl

    def plot_mode(
        self,
        mode_nr: int = 1,
        scaleF: float = 1.0,
        plot_lines: bool = True,
        plot_surf: bool = True,
        plot_undef: bool = True,
        def_sett: dict = "default",
        undef_sett: dict = "default",
        pl=None,
        bg_plotter: bool = True,
        notebook: bool = False,
    ):
        """
        Plot the mode shape of the structure for a given mode number.

        Parameters
        ----------
        mode_nr : int, optional
            The mode number to plot (default is 1).
        scaleF : float, optional
            Scale factor for the deformation (default is 1.0).
        plot_lines : bool, optional
            Whether to plot lines connecting sensor points (default is True).
        plot_surf : bool, optional
            Whether to plot surface meshes (default is True).
        plot_undef : bool, optional
            Whether to plot the undeformed shape of the structure (default is True).
        def_sett : dict or str, optional
            Settings for the deformed plot (default is 'default', which applies preset settings).
        undef_sett : dict or str, optional
            Settings for the undeformed plot (default is 'default', which applies preset settings).
        pl : pyvista.Plotter, optional
            Existing plotter instance to use (default is None, which creates a new plotter).
        bg_plotter : bool, optional
            Whether to use a background plotter for visualization (default is True).
        notebook : bool, optional
            If True, a plotter for use in Jupyter notebooks is created (default is False).

        Returns
        -------
        pyvista.Plotter
            The plotter object used for visualization.

        Raises
        ------
        ValueError
            If the result (`Res`) data is not provided when plotting a mode shape.
        """
        # import geometry and results
        geo = self.geo
        res = self.res

        # define the plotter object type
        if pl is None:
            if notebook:
                pl = pv.Plotter(notebook=True)
            elif bg_plotter:
                pl = pvqt.BackgroundPlotter()
            else:
                pl = pv.Plotter()

        # define default settings for plot
        def_settings = dict(cmap="plasma", opacity=0.7, show_scalar_bar=False)
        undef_settings = dict(color="gray", opacity=0.3)

        if def_sett == "default":
            def_sett = def_settings

        if undef_sett == "default":
            undef_sett = undef_settings

        # GEOMETRY
        points = geo.pts_coord.to_numpy()
        lines = geo.sens_lines
        surfs = geo.sens_surf
        # geometry in pyvista format
        if lines is not None:
            lines = np.array([np.hstack([2, line]) for line in lines])
        if surfs is not None:
            surfs = np.array([np.hstack([3, surf]) for surf in surfs])

        # Mode shape
        if res is not None:
            phi = res.Phi[:, int(mode_nr - 1)].real * scaleF
        else:
            raise ValueError("You must pass the Res class to plot a mode shape!")

        # APPLY POINTS TO SENSOR MAPPING
        df_phi_map = gen.dfphi_map_func(
            phi, geo.sens_names, geo.sens_map, cstrn=geo.cstrn
        )
        # calculate deformed shape (NEW POINTS)
        newpoints = points + df_phi_map.to_numpy() * geo.sens_sign.to_numpy()

        # If true plot undeformed shape
        if plot_undef:
            pl.add_points(points, **undef_sett)
            if plot_lines:
                line_mesh = pv.PolyData(points, lines=lines)
                pl.add_mesh(line_mesh, **undef_sett)
            if plot_surf:
                face_mesh = pv.PolyData(points, faces=surfs)
                pl.add_mesh(face_mesh, **undef_sett)

        # PLOT MODE SHAPE
        pl.add_points(newpoints, scalars=df_phi_map.values, **def_sett)
        if plot_lines:
            line_mesh = pv.PolyData(newpoints, lines=lines)
            pl.add_mesh(line_mesh, scalars=df_phi_map.values, **def_sett)
        if plot_surf:
            face_mesh = pv.PolyData(newpoints, faces=surfs)
            pl.add_mesh(face_mesh, scalars=df_phi_map.values, **def_sett)

        pl.add_text(
            rf"Mode nr. {mode_nr}, fn = {res.Fn[mode_nr-1]:.3f}Hz",
            position="upper_edge",
            color="black",
            # font_size=26,
        )
        pl.add_axes(line_width=5, labels_off=False)
        pl.show()

        return pl

    def animate_mode(
        self,
        mode_nr: int = 1,
        scaleF: float = 1.0,
        plot_lines: bool = True,
        plot_surf: bool = True,
        def_sett: dict = "default",
        saveGIF: bool = False,
        pl=None,
    ) -> "pv.Plotter":
        """
        Animate the mode shape for the given mode number.

        Parameters
        ----------
        mode_nr : int, optional
            The mode number to animate (default is 1).
        scaleF : float, optional
            Scale factor for the deformation (default is 1.0).
        plot_lines : bool, optional
            Whether to plot lines connecting sensor points (default is True).
        plot_surf : bool, optional
            Whether to plot surface meshes (default is True).
        def_sett : dict or str, optional
            Settings for the deformed plot (default is 'default', which applies preset settings).
        saveGIF : bool, optional
            If True, the animation is saved as a GIF (default is False).
        pl : pyvista.Plotter, optional
            Existing plotter instance to use (default is None, which creates a new plotter).

        Returns
        -------
        pyvista.Plotter
            The plotter object used for the animation.
        """
        # define default settings for plot
        def_settings = dict(cmap="plasma", opacity=0.7, show_scalar_bar=False)

        if def_sett == "default":
            def_sett = def_settings

        # import geometry and results
        geo = self.geo
        res = self.res
        points = pv.pyvista_ndarray(geo.pts_coord.to_numpy())
        lines = geo.sens_lines
        surfs = geo.sens_surf
        # geometry in pyvista format
        if lines is not None:
            lines = np.array([np.hstack([2, line]) for line in lines])
        if surfs is not None:
            surfs = np.array([np.hstack([3, surf]) for surf in surfs])

        # Mode shape
        phi = res.Phi[:, int(mode_nr - 1)].real * scaleF

        # mode shape mapped to points
        df_phi_map = gen.dfphi_map_func(
            phi, geo.sens_names, geo.sens_map, cstrn=geo.cstrn
        )
        # add together coordinates and mode shape displacement
        # newpoints = (points + df_phi_map.to_numpy() * geo.sens_sign.to_numpy() )

        # copy the dataset as we will modify its coordinates
        points_c = points.copy()

        if pl is None:
            pl = pv.Plotter(off_screen=False) if saveGIF else pvqt.BackgroundPlotter()

        # PLOT MODE SHAPE
        def_pts = pl.add_points(points_c, scalars=df_phi_map.values, **def_sett)

        if plot_lines:
            line_mesh = pv.PolyData(points_c, lines=lines)
            pl.add_mesh(line_mesh, scalars=df_phi_map.values, **def_sett)
        if plot_surf:
            face_mesh = pv.PolyData(points_c, faces=surfs)
            pl.add_mesh(face_mesh, scalars=df_phi_map.values, **def_sett)

        pl.add_text(
            rf"Mode nr. {mode_nr}, fn = {res.Fn[mode_nr-1]:.3f}Hz",
            position="upper_edge",
            color="black",
            # font_size=26,
        )

        if saveGIF:
            pl.enable_anti_aliasing("fxaa")
            n_frames = 30
            pl.open_gif(f"Mode nr. {mode_nr}.gif")
            for phase in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
                def_pts.mapper.dataset.points = (
                    points
                    + df_phi_map.to_numpy() * geo.sens_sign.to_numpy() * np.cos(phase)
                )
                line_mesh.points = (
                    points
                    + df_phi_map.to_numpy() * geo.sens_sign.to_numpy() * np.cos(phase)
                )
                face_mesh.points = (
                    points
                    + df_phi_map.to_numpy() * geo.sens_sign.to_numpy() * np.cos(phase)
                )
                pl.add_axes(line_width=5, labels_off=False)
                pl.write_frame()
            pl.show(auto_close=False)
        else:

            def update_shape():
                n_frames = 30
                for phase in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
                    def_pts.mapper.dataset.points = (
                        points
                        + df_phi_map.to_numpy() * geo.sens_sign.to_numpy() * np.cos(phase)
                    )
                    line_mesh.points = (
                        points
                        + df_phi_map.to_numpy() * geo.sens_sign.to_numpy() * np.cos(phase)
                    )
                    face_mesh.points = (
                        points
                        + df_phi_map.to_numpy() * geo.sens_sign.to_numpy() * np.cos(phase)
                    )
                    pl.add_axes(line_width=5, labels_off=False)
                    pl.update()

            pl.add_callback(update_shape, interval=100)
            # pl.show()
        # pl.close()

        return pl
