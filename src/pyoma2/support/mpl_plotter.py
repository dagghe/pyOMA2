# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:48:34 2024

@author: dagpa
"""

import typing

import matplotlib.pyplot as plt
import numpy as np

from pyoma2.algorithms.data.result import BaseResult, MsPoserResult
from pyoma2.functions.gen import dfphi_map_func
from pyoma2.functions.plot import (
    plt_lines,
    plt_nodes,
    plt_quiver,
    plt_surf,
    set_ax_options,
    set_view,
)
from pyoma2.support.geometry import Geometry1, Geometry2


class MplGeoPlotter:
    """ """

    def __init__(
        self,
        Geo: typing.Union[Geometry1, Geometry2],
        Res: typing.Union[BaseResult, MsPoserResult] = None,
    ) -> typing.Any:
        self.Geo = Geo
        self.Res = Res

    # metodo per plottare geometria 1
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
    ):
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
        if type(self.Geo) != Geometry1:
            raise ValueError("geo1 is not defined. Call def_geo1 first.")
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Plot of the geometry and sensors' placement and direction")
        # plot sensors' nodes
        sens_coord = self.Geo.sens_coord[["x", "y", "z"]].to_numpy()
        plt_nodes(ax, sens_coord, color=col_sns)

        # plot sensors' directions
        plt_quiver(
            ax,
            sens_coord,
            self.Geo.sens_dir,
            scaleF=scaleF,
            names=self.Geo.sens_names,
            color=col_sns,
            color_text=col_txt,
            method="2",
        )

        # Check that BG nodes are defined
        if self.Geo.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, self.Geo.bg_nodes, color=col_BG_nodes, alpha=0.5)
            # Check that BG lines are defined
            if self.Geo.bg_lines is not None:
                # if True plot
                plt_lines(
                    ax,
                    self.Geo.bg_nodes,
                    self.Geo.bg_lines,
                    color=col_BG_lines,
                    alpha=0.5,
                )
            if self.Geo.bg_surf is not None:
                # if True plot
                plt_surf(
                    ax, self.Geo.bg_nodes, self.Geo.bg_surf, alpha=0.1, color=col_BG_surf
                )

        # check for sens_lines
        if self.Geo.sens_lines is not None:
            # if True plot
            plt_lines(ax, sens_coord, self.Geo.sens_lines, color=col_sns_lines)

        # Set ax options
        set_ax_options(
            ax,
            bg_color="w",
            remove_fill=True,
            remove_grid=True,
            remove_axis=True,
            scaleF=scaleF,
        )

        # Set view
        set_view(ax, view=view)

        return fig, ax

    def plot_mode_g1(
        self,
        mode_numb: int,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz"] = "3D",
        col_sns: str = "red",
        col_sns_lines: str = "red",
        col_BG_nodes: str = "gray",
        col_BG_lines: str = "gray",
        col_BG_surf: str = "gray",
    ) -> typing.Any:
        """
        Plots a 3D mode shape for a specified mode number using the Geometry1 object.

        Parameters
        ----------
        Geo : Geometry1
            Geometry object containing sensor coordinates and other information.
        mode_numb : int
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
        if not isinstance(self.Geo, Geometry1):
            raise ValueError(
                f"geo1 is not defined. cannot plot geometry on {self}. Call def_geo1 first."
            )

        if self.Res.Fn is None:
            raise ValueError("Run algorithm first")

        Geo = self.Geo
        # Select the (real) mode shape
        phi = self.Res.Phi[:, int(mode_numb - 1)].real
        fn = self.Res.Fn[int(mode_numb - 1)]

        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        # set title
        ax.set_title(f"Mode nr. {mode_numb}, $f_n$={fn:.3f}Hz")

        # plot sensors' nodes
        sens_coord = Geo.sens_coord[["x", "y", "z"]].to_numpy()
        plt_nodes(ax, sens_coord, color="red")

        # plot Mode shape
        plt_quiver(
            ax,
            sens_coord,
            Geo.sens_dir * phi.reshape(-1, 1),
            scaleF=scaleF,
            method="2",
            color=col_sns,
            #            names=Geo.sens_names,
        )

        # Check that BG nodes are defined
        if Geo.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, Geo.bg_nodes, color=col_BG_nodes, alpha=0.5)
            # Check that BG lines are defined
            if Geo.bg_lines is not None:
                # if True plot
                plt_lines(
                    ax,
                    Geo.bg_nodes,
                    Geo.bg_lines,
                    color=col_BG_lines,
                    alpha=0.5,
                )
            if Geo.bg_surf is not None:
                # if True plot
                plt_surf(ax, Geo.bg_nodes, Geo.bg_surf, alpha=0.1, color=col_BG_surf)

        # check for sens_lines
        if Geo.sens_lines is not None:
            # if True plot
            plt_lines(ax, sens_coord, Geo.sens_lines, color=col_sns_lines)

        # Set ax options
        set_ax_options(
            ax,
            bg_color="w",
            remove_fill=True,
            remove_grid=True,
            remove_axis=True,
            scaleF=scaleF,
        )

        # Set view
        set_view(ax, view=view)
        return fig, ax

    def plot_geo2(
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
    ):
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
        if type(self.Geo) != Geometry2:
            raise ValueError("geo2 is not defined. Call def_geo2 first.")
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Plot of the geometry and sensors' placement and direction")
        # plot sensors'
        pts = self.Geo.pts_coord.to_numpy()[:, :]
        plt_nodes(ax, pts, color="red")

        # plot sensors' directions
        ch_names = self.Geo.sens_map.to_numpy()
        s_sign = self.Geo.sens_sign.to_numpy().astype(float)  # array of signs

        zero2 = np.zeros((s_sign.shape[0], 2))

        s_sign[s_sign == 0] = np.nan

        s_sign1 = np.hstack((s_sign[:, 0].reshape(-1, 1), zero2))
        s_sign2 = np.insert(zero2, 1, s_sign[:, 1], axis=1)
        s_sign3 = np.hstack((zero2, s_sign[:, 2].reshape(-1, 1)))

        valid_indices1 = ch_names[:, 0] != 0
        valid_indices2 = ch_names[:, 1] != 0
        valid_indices3 = ch_names[:, 2] != 0

        if np.any(valid_indices1):
            plt_quiver(
                ax,
                pts[valid_indices1],
                s_sign1[valid_indices1],
                scaleF=scaleF,
                names=ch_names[valid_indices1, 0],
                color=col_sns,
                color_text=col_txt,
                method="2",
            )
        if np.any(valid_indices2):
            plt_quiver(
                ax,
                pts[valid_indices2],
                s_sign2[valid_indices2],
                scaleF=scaleF,
                names=ch_names[valid_indices2, 1],
                color=col_sns,
                color_text=col_txt,
                method="2",
            )
        if np.any(valid_indices3):
            plt_quiver(
                ax,
                pts[valid_indices3],
                s_sign3[valid_indices3],
                scaleF=scaleF,
                names=ch_names[valid_indices3, 2],
                color=col_sns,
                color_text=col_txt,
                method="2",
            )

        # Check that BG nodes are defined
        if self.Geo.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, self.Geo.bg_nodes, color=col_BG_nodes, alpha=0.5)
            # Check that BG lines are defined
            if self.Geo.bg_lines is not None:
                # if True plot
                plt_lines(
                    ax,
                    self.Geo.bg_nodes,
                    self.Geo.bg_lines,
                    color=col_BG_lines,
                    alpha=0.5,
                )
            if self.Geo.bg_surf is not None:
                # if True plot
                plt_surf(
                    ax, self.Geo.bg_nodes, self.Geo.bg_surf, color=col_BG_surf, alpha=0.1
                )

        # check for sens_lines
        if self.Geo.sens_lines is not None:
            # if True plot
            plt_lines(ax, pts, self.Geo.sens_lines, color=col_sns_lines)

        if self.Geo.sens_surf is not None:
            # if True plot
            plt_surf(
                ax,
                self.Geo.pts_coord.values,
                self.Geo.sens_surf,
                color=col_sns_surf,
                alpha=0.3,
            )

        # Set ax options
        set_ax_options(
            ax,
            bg_color="w",
            remove_fill=True,
            remove_grid=True,
            remove_axis=True,
            scaleF=scaleF,
        )

        # Set view
        set_view(ax, view=view)
        return fig, ax

    def plot_mode_g2(
        self,
        mode_numb: typing.Optional[int],
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz"] = "3D",
        color: str = "cmap",
        *args,
        **kwargs,
    ) -> typing.Any:
        """
        Plots a 3D mode shape for a specified mode number using the Geometry2 object.

        Parameters
        ----------
        geo2 : Geometry2
            Geometry object containing nodes, sensor information, and additional geometrical data.
        mode_numb : int
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
        typing.Any
            A tuple containing the matplotlib figure and axes of the mode shape plot.
        """
        if type(self.Geo) != Geometry2:
            raise ValueError("geo2 is not defined. Call def_geo2 first.")

        geo2 = self.Geo

        if self.Res.Fn is None:
            raise ValueError("Run algorithm first")

        # Select the (real) mode shape
        fn = self.Res.Fn[int(mode_numb - 1)]
        phi = self.Res.Phi[:, int(mode_numb - 1)].real * scaleF

        # APPLY POINTS TO SENSOR MAPPING
        df_phi_map = dfphi_map_func(phi, geo2.sens_names, geo2.sens_map, cstrn=geo2.cstrn)
        # add together coordinates and mode shape displacement
        newpoints = (
            geo2.pts_coord.to_numpy() + df_phi_map.to_numpy() * geo2.sens_sign.to_numpy()
        )

        # create fig and ax
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        ax.set_title(f"Mode nr. {mode_numb}, $f_n$={fn:.3f}Hz")

        # Check that BG nodes are defined
        if geo2.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, geo2.bg_nodes, color="gray", alpha=0.5)
            # Check that BG lines are defined
            if geo2.bg_lines is not None:
                # if True plot
                plt_lines(ax, geo2.bg_nodes, geo2.bg_lines, color="gray", alpha=0.5)
            if geo2.bg_surf is not None:
                # if True plot
                plt_surf(ax, geo2.bg_nodes, geo2.bg_surf, alpha=0.1)
        # PLOT MODE SHAPE
        if color == "cmap":
            oldpoints = geo2.pts_coord.to_numpy()[:, :]
            plt_nodes(ax, newpoints, color="cmap", initial_coord=oldpoints)

        else:
            plt_nodes(ax, newpoints, color=color)
        # check for sens_lines
        if geo2.sens_lines is not None:
            if color == "cmap":
                plt_lines(
                    ax, newpoints, geo2.sens_lines, color="cmap", initial_coord=oldpoints
                )
            else:
                plt_lines(ax, newpoints, geo2.sens_lines, color=color)

        if geo2.sens_surf is not None:
            if color == "cmap":
                plt_surf(
                    ax,
                    newpoints,
                    geo2.sens_surf,
                    color="cmap",
                    initial_coord=oldpoints,
                    alpha=0.4,
                )
            else:
                plt_surf(ax, newpoints, geo2.sens_surf, color=color, alpha=0.4)

        # Set ax options
        set_ax_options(
            ax,
            bg_color="w",
            remove_fill=True,
            remove_grid=True,
            remove_axis=True,
            scaleF=scaleF,
        )

        # Set view
        set_view(ax, view=view)

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

# Geo = Geometry1(
#         sens_names=Geo[0],
#         sens_coord=Geo[1],
#         sens_dir=Geo[2].values,
#         sens_lines=Geo[3],
#         bg_nodes=Geo[4],
#         bg_lines=Geo[5],
#         bg_surf=Geo[6],
#         )

# # Geo = Geometry2(
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
# Plotter = MplGeoPlotter(Geo,Res)

# # =============================================================================
# # GEO1
# Plotter.plot_geo1(scaleF=8000, )
# Plotter.plot_mode_g1(mode_numb=6, scaleF=8000)

# # =============================================================================
# # GEO2
# # Plotter.plot_geo2(scaleF=8000, )
# # Plotter.plot_mode_g2(mode_numb=6, scaleF=8000)
