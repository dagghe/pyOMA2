from __future__ import annotations

import typing
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from pyoma2.algorithms.data.result import BaseResult
from pyoma2.functions.gen import (
    check_on_geo1,
    check_on_geo2,
    read_excel_file,
)

from .data import Geometry1, Geometry2
from .mpl_plotter import Geo1MplPlotter, Geo2MplPlotter
from .pyvista_plotter import PvGeoPlotter

if typing.TYPE_CHECKING:
    try:
        import pyvista as pv
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


class GeometryMixin:
    """
    Mixin that gives the ability to define the geometry the instance of the setup class.

    This mixin provides methods to define the geometry setups for the instance, including
    sensor names, coordinates, directions, and optional elements like lines, surfaces, and
    background nodes. It also includes methods to plot the geometry setups using Matplotlib
    and PyVista for 2D and 3D visualization, respectively.
    """

    geo1: typing.Optional[Geometry1] = None
    geo2: typing.Optional[Geometry2] = None

    def def_geo1(
        self,
        # # MANDATORY
        sens_names: typing.Union[
            typing.List[str],
            typing.List[typing.List[str]],
            pd.DataFrame,
            npt.NDArray[np.str_],
        ],  # sensors' names
        sens_coord: pd.DataFrame,  # sensors' coordinates
        sens_dir: npt.NDArray[np.int64],  # sensors' directions
        # # OPTIONAL
        sens_lines: npt.NDArray[np.int64] = None,  # lines connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None,  # Background nodes
        bg_lines: npt.NDArray[np.int64] = None,  # Background lines
        bg_surf: npt.NDArray[np.float64] = None,  # Background surfaces
    ) -> None:
        """
        Defines the first geometry setup (geo1) for the instance.

        This method sets up the geometry involving sensors' names, coordinates, directions,
        and optional elements like sensor lines, background nodes, lines, and surfaces.

        Parameters
        ----------
        sens_names : Union[numpy.ndarray of string, List of string]
            An array or list containing the names of the sensors.
        sens_coord : pandas.DataFrame
            A DataFrame containing the coordinates of the sensors.
        sens_dir : numpy.ndarray of int64
            An array defining the directions of the sensors.
        sens_lines : numpy.ndarray of int64, optional
            An array defining lines connecting sensors. Default is None.
        bg_nodes : numpy.ndarray of float64, optional
            An array defining background nodes. Default is None.
        bg_lines : numpy.ndarray of int64, optional
            An array defining background lines. Default is None.
        bg_surf : numpy.ndarray of float64, optional
            An array defining background surfaces. Default is None.
        """
        # Get reference index (if any)
        ref_ind = getattr(self, "ref_ind", None)

        # Assemble dictionary for check function
        file_dict = {
            "sensors names": sens_names,
            "sensors coordinates": sens_coord,
            "sensors directions": sens_dir,
            "sensors lines": sens_lines if sens_lines is not None else pd.DataFrame(),
            "BG nodes": bg_nodes if bg_nodes is not None else pd.DataFrame(),
            "BG lines": bg_lines if bg_lines is not None else pd.DataFrame(),
            "BG surfaces": bg_surf if bg_surf is not None else pd.DataFrame(),
        }

        # check on input
        res_ok = check_on_geo1(file_dict, ref_ind=ref_ind)

        self.geo1 = Geometry1(
            sens_names=res_ok[0],
            sens_coord=res_ok[1],
            sens_dir=res_ok[2],
            sens_lines=res_ok[3],
            bg_nodes=res_ok[4],
            bg_lines=res_ok[5],
            bg_surf=res_ok[6],
        )

    # metodo per definire geometria 2
    def def_geo2(
        self,
        # MANDATORY
        sens_names: typing.Union[
            typing.List[str],
            typing.List[typing.List[str]],
            pd.DataFrame,
            npt.NDArray[np.str_],
        ],  # sensors' names
        pts_coord: pd.DataFrame,  # points' coordinates
        sens_map: pd.DataFrame,  # mapping
        # OPTIONAL
        cstr: pd.DataFrame = None,
        sens_sign: pd.DataFrame = None,
        sens_lines: npt.NDArray[np.int64] = None,  # lines connecting sensors
        sens_surf: npt.NDArray[np.int64] = None,  # surf connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None,  # Background nodes
        bg_lines: npt.NDArray[np.float64] = None,  # Background lines
        bg_surf: npt.NDArray[np.float64] = None,  # Background lines
    ) -> None:
        """
        Defines the second geometry setup (geo2) for the instance.

        This method sets up an alternative geometry configuration, including sensors' names,
        points' coordinates, mapping, sign data, and optional elements like constraints,
        sensor lines, background nodes, lines, and surfaces.

        Parameters
        ----------
        sens_names : Union[list of str, list of list of str, pandas.DataFrame, numpy.ndarray of str]
            Sensors' names. It can be a list of strings, a list of lists of strings, a DataFrame, or a NumPy array.
        pts_coord : pandas.DataFrame
            A DataFrame containing the coordinates of the points.
        sens_map : pandas.DataFrame
            A DataFrame containing the mapping data for sensors.
        cstrn : pandas.DataFrame, optional
            A DataFrame containing constraints. Default is None.
        sens_sign : pandas.DataFrame, optional
            A DataFrame containing sign data for the sensors. Default is None.
        sens_lines : numpy.ndarray of int64, optional
            An array defining lines connecting sensors. Default is None.
        bg_nodes : numpy.ndarray of float64, optional
            An array defining background nodes. Default is None.
        bg_lines : numpy.ndarray of float64, optional
            An array defining background lines. Default is None.
        bg_surf : numpy.ndarray of float64, optional
            An array defining background surfaces. Default is None.

        Notes
        -----
        This method adapts indices for 0-indexed lines in `bg_lines`, `sens_lines`, and `bg_surf`.
        """
        # Get reference index
        ref_ind = getattr(self, "ref_ind", None)

        # Assemble dictionary for check function
        file_dict = {
            "sensors names": sens_names,
            "points coordinates": pts_coord,
            "mapping": sens_map,
            "constraints": cstr if cstr is not None else pd.DataFrame(),
            "sensors sign": sens_sign if sens_sign is not None else pd.DataFrame(),
            "sensors lines": sens_lines if sens_lines is not None else pd.DataFrame(),
            "sensors surfaces": sens_surf if sens_surf is not None else pd.DataFrame(),
            "BG nodes": bg_nodes if bg_nodes is not None else pd.DataFrame(),
            "BG lines": bg_lines if bg_lines is not None else pd.DataFrame(),
            "BG surfaces": bg_surf if bg_surf is not None else pd.DataFrame(),
        }

        # check on input
        res_ok = check_on_geo2(file_dict, ref_ind=ref_ind)

        # Save to geometry
        self.geo2 = Geometry2(
            sens_names=res_ok[0],
            pts_coord=res_ok[1].astype(float),
            sens_map=res_ok[2],
            cstrn=res_ok[3],
            sens_sign=res_ok[4],
            sens_lines=res_ok[5],
            sens_surf=res_ok[6],
            bg_nodes=res_ok[7],
            bg_lines=res_ok[8],
            bg_surf=res_ok[9],
        )

    def _def_geo_by_file(
        self, geo_type: str, path: str, **read_excel_file_kwargs
    ) -> None:
        """
        Defines the geometry setup from an Excel file.

        This method reads an Excel file to extract sensor information, including sensor names,
        coordinates, and other optional geometry elements such as lines and background nodes.
        The information is used to set up the geometry for the instance.

        Parameters
        ----------
        geo_type : str
            The type of geometry to define: 'geo1' or 'geo2'.
        path : str
            The file path to the Excel file containing the geometry data.
        read_excel_file_kwargs : dict, optional
            Additional keyword arguments to pass to the `read_excel_file` function.

        Raises
        ------
        ValueError
            If the input data is invalid or missing required fields.
        """
        # Get reference index
        ref_ind = getattr(self, "ref_ind", None)

        # Read the Excel file
        file_dict = read_excel_file(path=path, **read_excel_file_kwargs)

        # Check on input
        if geo_type == "geo1":
            res_ok = check_on_geo1(file_dict, ref_ind=ref_ind)
            self.geo1 = Geometry1(
                sens_names=res_ok[0],
                sens_coord=res_ok[1],
                sens_dir=res_ok[2],
                sens_lines=res_ok[3],
                bg_nodes=res_ok[4],
                bg_lines=res_ok[5],
                bg_surf=res_ok[6],
            )
        elif geo_type == "geo2":
            res_ok = check_on_geo2(file_dict, ref_ind=ref_ind)
            self.geo2 = Geometry2(
                sens_names=res_ok[0],
                pts_coord=res_ok[1].astype(float),
                sens_map=res_ok[2],
                cstrn=res_ok[3],
                sens_sign=res_ok[4],
                sens_lines=res_ok[5],
                sens_surf=res_ok[6],
                bg_nodes=res_ok[7],
                bg_lines=res_ok[8],
                bg_surf=res_ok[9],
            )
        else:
            raise ValueError(f"Invalid geometry type: {geo_type}")

    # metodo per definire geometria 1 da file
    def def_geo1_by_file(self, path: str, **read_excel_file_kwargs) -> None:
        """
        Defines the first geometry (geo1) from an Excel file.

        This method reads an Excel file to extract sensor information, including sensor names,
        coordinates, and other optional geometry elements such as lines and background nodes.
        The information is used to set up the geometry for the instance.

        Parameters
        ----------
        path : str
            The file path to the Excel file containing the geometry data.
        read_excel_file_kwargs : dict, optional
            Additional keyword arguments to pass to the `read_excel_file` function.

        Raises
        ------
        ValueError
            If the input data is invalid or missing required fields.
        """
        self._def_geo_by_file(geo_type="geo1", path=path, **read_excel_file_kwargs)

    def def_geo2_by_file(self, path: str, **read_excel_file_kwargs) -> None:
        """
        Defines the second geometry (geo2) from an Excel file.

        This method reads an Excel file to extract information related to the geometry configuration,
        including sensor names, points' coordinates, mapping, and optional background nodes and surfaces.
        The information is used to set up the second geometry for the instance.

        Parameters
        ----------
        path : str
            The file path to the Excel file containing the geometry data.
        read_excel_file_kwargs : dict, optional
            Additional keyword arguments to pass to the `read_excel_file` function.

        Raises
        ------
        ValueError
            If the input data is invalid or missing required fields.
        """
        self._def_geo_by_file(geo_type="geo2", path=path, **read_excel_file_kwargs)

    # PLOT GEO1 - mpl plotter
    def plot_geo1(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz"] = "3D",
        col_sns: str = "red",
        col_sns_lines: str = "red",
        col_BG_nodes: str = "gray",
        col_BG_lines: str = "gray",
        col_BG_surf: str = "gray",
        col_txt: str = "red",
    ) -> typing.Tuple[plt.Figure, plt.Axes]:
        """
        Plots the first geometry setup (geo1) using Matplotlib.

        This method creates a 2D or 3D plot of the first geometry, including sensors, lines, background nodes,
        and surfaces, using customizable color schemes for each element.

        Parameters
        ----------
        scaleF : int, optional
            Scaling factor for the plot. Default is 1.
        view : {'3D', 'xy', 'xz', 'yz'}, optional
            The view angle of the plot. Default is '3D'.
        col_sns : str, optional
            Color of the sensors. Default is 'red'.
        col_sns_lines : str, optional
            Color of the lines connecting sensors. Default is 'red'.
        col_BG_nodes : str, optional
            Color of the background nodes. Default is 'gray'.
        col_BG_lines : str, optional
            Color of the background lines. Default is 'gray'.
        col_BG_surf : str, optional
            Color of the background surfaces. Default is 'gray'.
        col_txt : str, optional
            Color of the text labels for sensors. Default is 'red'.

        Returns
        -------
        tuple
            A tuple containing the Matplotlib figure and axes objects.

        Raises
        ------
        ValueError
            If `geo1` is not defined.
        """

        if self.geo1 is None:
            raise ValueError("geo1 is not defined. Call def_geo1 first.")

        Plotter = Geo1MplPlotter(self.geo1)

        fig, ax = Plotter.plot_geo(
            scaleF,
            view,
            col_sns,
            col_sns_lines,
            col_BG_nodes,
            col_BG_lines,
            col_BG_surf,
            col_txt,
        )
        return fig, ax

    # PLOT GEO2 - PyVista plotter
    def plot_geo2(
        self,
        scaleF: int = 1,
        col_sens: str = "red",
        plot_lines: bool = True,
        plot_surf: bool = True,
        points_sett: dict = "default",
        lines_sett: dict = "default",
        surf_sett: dict = "default",
        bg_plotter: bool = True,
        notebook: bool = False,
    ) -> "pv.Plotter":
        """
        Plots the second geometry setup (geo2) using PyVista for 3D visualization.

        This method creates a 3D interactive plot of the second geometry setup with options
        to visualize sensor points, connecting lines, and surfaces. It provides various
        customization options for coloring and rendering.

        Parameters
        ----------
        scaleF : int, optional
            Scaling factor for the plot. Default is 1.
        col_sens : str, optional
            Color of the sensors. Default is 'red'.
        plot_lines : bool, optional
            Whether to plot lines connecting sensors. Default is True.
        plot_surf : bool, optional
            Whether to plot surfaces connecting sensors. Default is True.
        points_sett : dict, optional
            Settings for the points' appearance. Default is 'default'.
        lines_sett : dict, optional
            Settings for the lines' appearance. Default is 'default'.
        surf_sett : dict, optional
            Settings for the surfaces' appearance. Default is 'default'.
        bg_plotter : bool, optional
            Whether to include a background plotter. Default is True.
        notebook : bool, optional
            Whether to render the plot in a Jupyter notebook environment. Default is False.

        Returns
        -------
        pyvista.Plotter
            A PyVista Plotter object with the geometry visualization.

        Raises
        ------
        ValueError
            If `geo2` is not defined.
        """
        if self.geo2 is None:
            raise ValueError("geo2 is not defined. Call def_geo2 first.")

        Plotter = PvGeoPlotter(self.geo2)

        pl = Plotter.plot_geo(
            scaleF=scaleF,
            col_sens=col_sens,
            plot_lines=plot_lines,
            plot_surf=plot_surf,
            points_sett=points_sett,
            lines_sett=lines_sett,
            surf_sett=surf_sett,
            pl=None,
            bg_plotter=bg_plotter,
            notebook=notebook,
        )
        return pl

    # PLOT GEO2 - Matplotlib plotter
    def plot_geo2_mpl(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        col_sns: str = "red",
        col_sns_lines: str = "black",
        col_sns_surf: str = "lightcoral",
        col_BG_nodes: str = "gray",
        col_BG_lines: str = "gray",
        col_BG_surf: str = "gray",
        col_txt: str = "red",
    ) -> typing.Tuple[plt.Figure, plt.Axes]:
        """
        Plots the second geometry setup (geo2) using Matplotlib.

        This method creates a 2D or 3D plot of the second geometry, including sensors, lines,
        surfaces, background nodes, and surfaces, with customizable colors.

        Parameters
        ----------
        scaleF : int, optional
            Scaling factor for the plot. Default is 1.
        view : {'3D', 'xy', 'xz', 'yz', 'x', 'y', 'z'}, optional
            The view angle of the plot. Default is '3D'.
        col_sns : str, optional
            Color of the sensors. Default is 'red'.
        col_sns_lines : str, optional
            Color of the lines connecting sensors. Default is 'black'.
        col_sns_surf : str, optional
            Color of the surfaces connecting sensors. Default is 'lightcoral'.
        col_BG_nodes : str, optional
            Color of the background nodes. Default is 'gray'.
        col_BG_lines : str, optional
            Color of the background lines. Default is 'gray'.
        col_BG_surf : str, optional
            Color of the background surfaces. Default is 'gray'.
        col_txt : str, optional
            Color of the text labels for sensors. Default is 'red'.

        Returns
        -------
        tuple
            A tuple containing the Matplotlib figure and axes objects.

        Raises
        ------
        ValueError
            If `geo2` is not defined.
        """
        if self.geo2 is None:
            raise ValueError("geo2 is not defined. Call def_geo2 first.")

        Plotter = Geo2MplPlotter(self.geo2)

        fig, ax = Plotter.plot_geo(
            scaleF,
            view,
            col_sns,
            col_sns_lines,
            col_sns_surf,
            col_BG_nodes,
            col_BG_lines,
            col_BG_surf,
            col_txt,
        )
        return fig, ax

    def plot_mode_geo1(
        self,
        algo_res: BaseResult,
        mode_nr: int,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz"] = "3D",
        col_sns: str = "red",
        col_sns_lines: str = "red",
        col_BG_nodes: str = "gray",
        col_BG_lines: str = "gray",
        col_BG_surf: str = "gray",
    ) -> typing.Tuple[plt.Figure, plt.Axes]:
        """
        Plots the mode shapes for the first geometry setup (geo1) using Matplotlib.

        This method visualizes the mode shapes corresponding to the specified mode number, with customizable
        colors and scaling for different geometrical elements such as sensors, lines, and background surfaces.

        Parameters
        ----------
        algo_res : BaseResult
            The result object containing modal parameters and mode shape data.
        mode_nr : int
            The mode number to be plotted.
        scaleF : int, optional
            Scaling factor to adjust the size of the mode shapes. Default is 1.
        view : {'3D', 'xy', 'xz', 'yz'}, optional
            The viewing plane or angle for the plot. Default is '3D'.
        col_sns : str, optional
            Color of the sensors in the plot. Default is 'red'.
        col_sns_lines : str, optional
            Color of the lines connecting the sensors. Default is 'red'.
        col_BG_nodes : str, optional
            Color of the background nodes in the plot. Default is 'gray'.
        col_BG_lines : str, optional
            Color of the background lines in the plot. Default is 'gray'.
        col_BG_surf : str, optional
            Color of the background surfaces in the plot. Default is 'gray'.

        Returns
        -------
        tuple
            A tuple containing the Matplotlib figure and axes objects for further customization or saving.

        Raises
        ------
        ValueError
            If `geo1` is not defined or if the algorithm results are missing.
        """
        if self.geo1 is None:
            raise ValueError("geo1 is not defined. Call def_geo1 first.")

        if algo_res.Fn is None:
            raise ValueError("Run algorithm first")
        Plotter = Geo1MplPlotter(self.geo1, algo_res)

        fig, ax = Plotter.plot_mode(
            mode_nr,
            scaleF,
            view,
            col_sns,
            col_sns_lines,
            col_BG_nodes,
            col_BG_lines,
            col_BG_surf,
        )
        return fig, ax

    # PLOT MODI - PyVista plotter
    def plot_mode_geo2(
        self,
        algo_res: BaseResult,
        mode_nr: int = 1,
        scaleF: float = 1.0,
        plot_lines: bool = True,
        plot_surf: bool = True,
        plot_undef: bool = True,
        def_sett: dict = "default",
        undef_sett: dict = "default",
        bg_plotter: bool = True,
        notebook: bool = False,
        *args,
        **kwargs,
    ) -> "pv.Plotter":
        """
        Plots the mode shapes for the second geometry setup (geo2) using PyVista for interactive 3D visualization.

        This method uses PyVista for creating an interactive 3D plot of the mode shapes corresponding
        to the specified mode number. The plot can include options for visualizing lines, surfaces, and
        undeformed geometries, with customization for appearance settings.

        Parameters
        ----------
        algo_res : BaseResult
            The result object containing modal parameters and mode shape data.
        mode_nr : int, optional
            The mode number to be plotted. Default is 1.
        scaleF : float, optional
            Scaling factor for the mode shape visualization. Default is 1.0.
        plot_lines : bool, optional
            Whether to plot lines connecting sensors. Default is True.
        plot_surf : bool, optional
            Whether to plot surfaces connecting sensors. Default is True.
        plot_undef : bool, optional
            Whether to plot the undeformed geometry. Default is True.
        def_sett : dict, optional
            Settings for the deformed mode shapes. Default is 'default'.
        undef_sett : dict, optional
            Settings for the undeformed mode shapes. Default is 'default'.
        bg_plotter : bool, optional
            Whether to include a background plotter. Default is True.
        notebook : bool, optional
            Whether to render the plot in a Jupyter notebook. Default is False.

        Returns
        -------
        pyvista.Plotter
            A PyVista plotter object with the interactive 3D visualization.

        Raises
        ------
        ValueError
            If `geo2` is not defined or if the algorithm results are missing (e.g., `Fn` is None).
        """

        if self.geo2 is None:
            raise ValueError("geo2 is not defined. Call def_geo2 first.")

        if algo_res.Fn is None:
            raise ValueError("Run algorithm first")

        Plotter = PvGeoPlotter(self.geo2, algo_res)

        pl = Plotter.plot_mode(
            mode_nr=mode_nr,
            scaleF=scaleF,
            plot_lines=plot_lines,
            plot_surf=plot_surf,
            plot_undef=plot_undef,
            def_sett=def_sett,
            undef_sett=undef_sett,
            pl=None,
            bg_plotter=bg_plotter,
            notebook=notebook,
        )
        return pl

    # PLOT MODI - Matplotlib plotter
    def plot_mode_geo2_mpl(
        self,
        algo_res: BaseResult,
        mode_nr: typing.Optional[int],
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz"] = "3D",
        color: str = "cmap",
        *args,
        **kwargs,
    ) -> typing.Tuple[plt.Figure, plt.Axes]:
        """
        Plots the mode shapes for the second geometry setup (geo2) using Matplotlib.

        This method visualizes the mode shapes for geo2, with customizable scaling, color, and viewing options.
        The plot can be configured for different modes and color maps.

        Parameters
        ----------
        algo_res : BaseResult
            The result object containing modal parameters and mode shape data.
        mode_nr : int, optional
            The mode number to be plotted. If None, the default mode is plotted.
        scaleF : int, optional
            Scaling factor to adjust the size of the mode shapes. Default is 1.
        view : {'3D', 'xy', 'xz', 'yz'}, optional
            The viewing plane or angle for the plot. Default is '3D'.
        color : str, optional
            Color scheme or colormap to be used for the mode shapes. Default is 'cmap'.

        Returns
        -------
        tuple
            A tuple containing the Matplotlib figure and axes objects for further customization or saving.

        Raises
        ------
        ValueError
            If `geo2` is not defined or if the algorithm results are missing (e.g., `Fn` is None).
        """
        if self.geo2 is None:
            raise ValueError("geo2 is not defined. Call def_geo2 first.")

        if algo_res.Fn is None:
            raise ValueError("Run algorithm first")

        Plotter = Geo2MplPlotter(self.geo2, algo_res)

        fig, ax = Plotter.plot_mode(mode_nr, scaleF, view, color)
        return fig, ax

    # PLOT MODI - PyVista plotter
    def anim_mode_g2(
        self,
        algo_res: BaseResult,
        mode_nr: int = 1,
        scaleF: float = 1.0,
        pl=None,
        plot_points: bool = True,
        plot_lines: bool = True,
        plot_surf: bool = True,
        def_sett: dict = "default",
        saveGIF: bool = False,
        *args,
        **kwargs,
    ) -> "pv.Plotter":
        """
        Creates an animation of the mode shapes for the second geometry setup (geo2) using PyVista.

        This method animates the mode shapes corresponding to the specified mode number, using
        PyVista for interactive 3D visualization. It supports saving the animation as a GIF.

        Parameters
        ----------
        algo_res : BaseResult
            The result object containing modal parameters and mode shape data.
        mode_nr : int, optional
            The mode number to animate. Default is 1.
        scaleF : float, optional
            Scaling factor for the mode shape animation. Default is 1.0.
        pl : pyvista.Plotter, optional
            An existing PyVista plotter object for the animation. If None, a new plotter is created.
        plot_points : bool, optional
            Whether to plot sensor points. Default is True.
        plot_lines : bool, optional
            Whether to plot lines connecting sensors. Default is True.
        plot_surf : bool, optional
            Whether to plot surfaces connecting sensors. Default is True.
        def_sett : dict, optional
            Settings for the deformed mode shapes. Default is 'default'.
        saveGIF : bool, optional
            Whether to save the animation as a GIF. Default is False.

        Returns
        -------
        pyvista.Plotter
            A PyVista plotter object with the animated 3D visualization.

        Raises
        ------
        ValueError
            If `geo2` is not defined or if the algorithm results are missing (e.g., `Fn` is None).
        """
        if self.geo2 is None:
            raise ValueError("geo2 is not defined. Call def_geo2 first.")

        if algo_res.Fn is None:
            raise ValueError("Run algorithm first")

        Plotter = PvGeoPlotter(self.geo2, algo_res)

        pl = Plotter.animate_mode(
            mode_nr=mode_nr,
            scaleF=scaleF,
            plot_lines=plot_lines,
            plot_surf=plot_surf,
            def_sett=def_sett,
            saveGIF=saveGIF,
            pl=None,
        )
        return pl
