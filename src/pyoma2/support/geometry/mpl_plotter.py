# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:48:34 2024

@author: dagpa
"""

from __future__ import annotations

import typing

import matplotlib.pyplot as plt
import numpy as np

from pyoma2.functions.gen import dfphi_map_func
from pyoma2.functions.plot import (
    plt_lines,
    plt_nodes,
    plt_quiver,
    plt_surf,
    set_ax_options,
    set_view,
)

from .data import Geometry1, Geometry2
from .plotter import BasePlotter, T_Geo


class MplPlotter(BasePlotter[T_Geo]):
    """An abstract base class for plotting geometry and mode shapes using Matplotlib."""

    def _create_figure(self):
        """Create and return a new figure and 3D axis."""
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        return fig, ax

    def _set_common_options(self, ax, scaleF, view):
        """Set common axis options and view."""
        set_ax_options(
            ax,
            bg_color="w",
            remove_fill=True,
            remove_grid=True,
            remove_axis=True,
            scaleF=scaleF,
        )
        set_view(ax, view=view)

    def _plot_background(self, ax, col_BG_nodes, col_BG_lines, col_BG_surf):
        """Plot background nodes, lines, and surfaces if they exist."""
        if self.geo.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, self.geo.bg_nodes, color=col_BG_nodes, alpha=0.5)
            # Check that BG lines are defined
            if self.geo.bg_lines is not None:
                # if True plot
                plt_lines(
                    ax,
                    self.geo.bg_nodes,
                    self.geo.bg_lines,
                    color=col_BG_lines,
                    alpha=0.5,
                )
            if self.geo.bg_surf is not None:
                # if True plot
                plt_surf(
                    ax, self.geo.bg_nodes, self.geo.bg_surf, color=col_BG_surf, alpha=0.1
                )


class Geo1MplPlotter(MplPlotter[Geometry1]):
    """A class to plot mode shapes in 3D using Geometry1."""

    def plot_geo(
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
        Plots the geometry (type 1) of tested structure.

        This method visualizes the geometry of a structure, including sensor placements and directions.
        It allows customization of the plot through various parameters such as scaling factor,
        view type, and options to remove fill, grid, and axis from the plot.

        Parameters
        ----------
        scaleF : int, optional
            The scaling factor for the sensor direction quivers. A higher value results in
            longer quivers. Default is 1.
        view : {'3D', 'xy', 'xz', 'yz'}, optional
            The type of view for plotting the geometry. Options include 3D and 2D projections
            on various planes. Default is "3D".
        remove_fill : bool, optional
            If True, removes the fill from the plot. Default is True.
        remove_grid : bool, optional
            If True, removes the grid from the plot. Default is True.
        remove_axis : bool, optional
            If True, removes the axis labels and ticks from the plot. Default is True.

        Raises
        ------
        ValueError
            If Geo is not defined in the setup.

        Returns
        -------
        tuple
            A tuple containing the figure and axis objects of the plot. This can be used for
            further customization or saving the plot externally.

        """
        fig, ax = self._create_figure()

        # plot sensors' nodes
        sens_coord = self.geo.sens_coord[["x", "y", "z"]].to_numpy()
        plt_nodes(ax, sens_coord, color=col_sns)

        # plot sensors' directions
        plt_quiver(
            ax,
            sens_coord,
            self.geo.sens_dir,
            scaleF=scaleF,
            names=self.geo.sens_names,
            color=col_sns,
            color_text=col_txt,
            method="2",
        )

        self._plot_background(ax, col_BG_nodes, col_BG_lines, col_BG_surf)

        # check for sens_lines
        if self.geo.sens_lines is not None:
            plt_lines(ax, sens_coord, self.geo.sens_lines, color=col_sns_lines)

        self._set_common_options(ax, scaleF, view)

        return fig, ax

    def plot_mode(
        self,
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
        Plots a 3D mode shape for a specified mode number using the Geometry1 object.

        Parameters
        ----------
        Geo : Geometry1
            Geometry object containing sensor coordinates and other information.
        mode_nr : int
            Mode number to visualize.
        scaleF : int, optional
            Scale factor for mode shape visualization. Default is 1.
        view : {'3D', 'xy', 'xz', 'yz'}, optional
            View for the 3D plot. Default is '3D'.
        remove_fill : bool, optional
            Whether to remove fill from the plot. Default is True.
        remove_grid : bool, optional
            Whether to remove grid from the plot. Default is True.
        remove_axis : bool, optional
            Whether to remove axis from the plot. Default is True.

        Returns
        -------
        typing.Any
            A tuple containing the matplotlib figure and axes of the mode shape plot.
        """
        if self.res.Fn is None:
            raise ValueError("Run algorithm first")

        # Select the (real) mode shape
        phi = self.res.Phi[:, int(mode_nr - 1)].real
        fn = self.res.Fn[int(mode_nr - 1)]

        fig, ax = self._create_figure()
        # Set title
        ax.set_title(f"Mode nr. {mode_nr}, $f_n$={fn:.3f}Hz")

        # plot sensors' nodes
        sens_coord = self.geo.sens_coord[["x", "y", "z"]].to_numpy()
        plt_nodes(ax, sens_coord, color="red")

        # plot Mode shape
        plt_quiver(
            ax,
            sens_coord,
            self.geo.sens_dir * phi.reshape(-1, 1),
            scaleF=scaleF,
            method="2",
            color=col_sns,
        )

        self._plot_background(ax, col_BG_nodes, col_BG_lines, col_BG_surf)

        # check for sens_lines
        if self.geo.sens_lines is not None:
            # if True plot
            plt_lines(ax, sens_coord, self.geo.sens_lines, color=col_sns_lines)

        self._set_common_options(ax, scaleF, view)

        return fig, ax


class Geo2MplPlotter(MplPlotter[Geometry2]):
    """A class to plot mode shapes in 3D using Geometry2."""

    def plot_geo(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz"] = "3D",
        col_sns: str = "red",
        col_sns_lines: str = "black",
        col_sns_surf: str = "lightcoral",
        col_BG_nodes: str = "gray",
        col_BG_lines: str = "gray",
        col_BG_surf: str = "gray",
        col_txt: str = "red",
    ) -> typing.Tuple[plt.Figure, plt.Axes]:
        """
        Plots the geometry (type 2) of tested structure.

        This method creates a 3D or 2D plot of a specific geometric configuration (geo2) with
        customizable features such as scaling factor, view type, and visibility options for
        fill, grid, and axes. It involves plotting sensor points, directions, and additional
        geometric elements if available.

        Parameters
        ----------
        scaleF : int, optional
            Scaling factor for the quiver plots representing sensors' directions. Default is 1.
        view : {'3D', 'xy', 'xz', 'yz'}, optional
            Specifies the type of view for the plot. Can be a 3D view or 2D projections on
            various planes. Default is "3D".
        remove_fill : bool, optional
            If True, the plot's fill is removed. Default is True.
        remove_grid : bool, optional
            If True, the plot's grid is removed. Default is True.
        remove_axis : bool, optional
            If True, the plot's axes are removed. Default is True.

        Raises
        ------
        ValueError
            If geo2 is not defined in the setup.

        Returns
        -------
        tuple
            Returns a tuple containing the figure and axis objects of the matplotlib plot.
            This allows for further customization or saving outside the method.
        """
        fig, ax = self._create_figure()

        # plot sensors'
        pts = self.geo.pts_coord.to_numpy()[:, :]
        plt_nodes(ax, pts, color="red")

        # plot sensors' directions
        ch_names = self.geo.sens_map.to_numpy()
        s_sign = self.geo.sens_sign.to_numpy().astype(float)

        zero2 = np.zeros((s_sign.shape[0], 2))
        s_sign[s_sign == 0] = np.nan
        s_signs = [
            np.hstack((s_sign[:, 0].reshape(-1, 1), zero2)),
            np.insert(zero2, 1, s_sign[:, 1], axis=1),
            np.hstack((zero2, s_sign[:, 2].reshape(-1, 1))),
        ]

        for i, s_sign_direction in enumerate(s_signs):
            valid_indices = ch_names[:, i] != 0
            if np.any(valid_indices):
                plt_quiver(
                    ax,
                    pts[valid_indices],
                    s_sign_direction[valid_indices],
                    scaleF=scaleF,
                    names=ch_names[valid_indices, i],
                    color=col_sns,
                    color_text=col_txt,
                    method="2",
                )

        self._plot_background(ax, col_BG_nodes, col_BG_lines, col_BG_surf)

        # check for sens_lines
        if self.geo.sens_lines is not None:
            # if True plot
            plt_lines(ax, pts, self.geo.sens_lines, color=col_sns_lines)

        if self.geo.sens_surf is not None:
            # if True plot
            plt_surf(
                ax,
                self.geo.pts_coord.values,
                self.geo.sens_surf,
                color=col_sns_surf,
                alpha=0.3,
            )

        self._set_common_options(ax, scaleF, view)

        return fig, ax

    def plot_mode(
        self,
        mode_nr: typing.Optional[int],
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz"] = "3D",
        color: str = "cmap",
        *args,
        **kwargs,
    ) -> typing.Tuple[plt.Figure, plt.Axes]:
        """
        Plots a 3D mode shape for a specified mode number using the Geometry2 object.

        Parameters
        ----------
        geo2 : Geometry2
            Geometry object containing nodes, sensor information, and additional geometrical data.
        mode_nr : int
            Mode number to visualize.
        scaleF : int, optional
            Scale factor for mode shape visualization. Default is 1.
        view : {'3D', 'xy', 'xz', 'yz', 'x', 'y', 'z'}, optional
            View for the 3D plot. Default is '3D'.
        remove_fill : bool, optional
            Whether to remove fill from the plot. Default is True.
        remove_grid : bool, optional
            Whether to remove grid from the plot. Default is True.
        remove_axis : bool, optional
            Whether to remove axis from the plot. Default is True.
        *args, **kwargs
            Additional arguments for customizations.

        Returns
        -------
        typing.Tuple[plt.Figure, plt.Axes]
            A tuple containing the matplotlib figure and axes of the mode shape plot.
        """
        if self.res.Fn is None:
            raise ValueError("Run algorithm first")

        # Select the (real) mode shape
        fn = self.res.Fn[int(mode_nr - 1)]
        phi = self.res.Phi[:, int(mode_nr - 1)].real * scaleF

        # APPLY POINTS TO SENSOR MAPPING
        df_phi_map = dfphi_map_func(
            phi, self.geo.sens_names, self.geo.sens_map, cstrn=self.geo.cstrn
        )
        # add together coordinates and mode shape displacement
        newpoints = (
            self.geo.pts_coord.to_numpy()
            + df_phi_map.to_numpy() * self.geo.sens_sign.to_numpy()
        )

        # create fig and ax
        fig, ax = self._create_figure()
        ax.set_title(f"Mode nr. {mode_nr}, $f_n$={fn:.3f}Hz")

        self._plot_background(ax, "gray", "gray", "gray")

        # PLOT MODE SHAPE
        if color == "cmap":
            oldpoints = self.geo.pts_coord.to_numpy()[:, :]
            plt_nodes(ax, newpoints, color="cmap", initial_coord=oldpoints)

        else:
            plt_nodes(ax, newpoints, color=color)
        # check for sens_lines
        if self.geo.sens_lines is not None:
            if color == "cmap":
                plt_lines(
                    ax,
                    newpoints,
                    self.geo.sens_lines,
                    color="cmap",
                    initial_coord=oldpoints,
                )
            else:
                plt_lines(ax, newpoints, self.geo.sens_lines, color=color)

        if self.geo.sens_surf is not None:
            if color == "cmap":
                plt_surf(
                    ax,
                    newpoints,
                    self.geo.sens_surf,
                    color="cmap",
                    initial_coord=oldpoints,
                    alpha=0.4,
                )
            else:
                plt_surf(ax, newpoints, self.geo.sens_surf, color=color, alpha=0.4)

        self._set_common_options(ax, scaleF, view)

        return fig, ax


# # =============================================================================
# # TEST
# # =============================================================================
# # START - IMPORT DATA

# # r"C:\Users\dpa\
# # r"X:\
# _geo1=r"C:\Users\dpa\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyomaTEST\HTC_geom\geo1.xlsx"
# _geo2=r"C:\Users\dpa\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyomaTEST\HTC_geom\Geo2_noBG.xlsx"
# _file=r"C:\Users\dpa\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyomaTEST\HTC_geom\PHI.npy"
# ref_ind = [[4, 5], [6, 7], [6, 7], [6, 7]]

# # Load mode shape
# Phi=np.load(_file)

# # Load geometry file
# Geo = import_excel_GEO1(_geo1,ref_ind)
# # Geo = import_excel_GEO2(_geo2,ref_ind)

# # =============================================================================
# # DEFINE GEOMETRY

# Geo1 = Geometry1(
#         sens_names=Geo[0],
#         sens_coord=Geo[1],
#         sens_dir=Geo[2].values,
#         sens_lines=Geo[3],
#         bg_nodes=Geo[4],
#         bg_lines=Geo[5],
#         bg_surf=Geo[6],
#         )

# # Geo2 = Geometry2(
# #             sens_names=Geo[0],
# #             pts_coord=Geo[1],
# #             sens_map=Geo[2],
# #             cstrn=Geo[3],
# #             sens_sign=Geo[4],
# #             sens_lines=Geo[5],
# #             sens_surf=Geo[6],
# #             bg_nodes=Geo[7],
# #             bg_lines=Geo[8],
# #             bg_surf=Geo[9],
# #         )

# Res = BaseResult(
#     Fn= np.arange(Phi.shape[1]),
#     Phi=Phi)


# # CREATE PLOTTER
# PlotterGeo1 = Geo1MplPlotter(Geo1, Res)
# PlotterGeo2 = Geo2MplPlotter(Geo2, Res)

# # =============================================================================
# # GEO1
# PlotterGeo1.plot_geo(scaleF=8000)
# PlotterGeo1.plot_mode(mode_nr=6, scaleF=8000)

# # =============================================================================
# # GEO2
# # PlotterGeo2.plot_geo(scaleF=8000)
# # PlotterGeo2.plot_mode(mode_nr=6, scaleF=8000)
