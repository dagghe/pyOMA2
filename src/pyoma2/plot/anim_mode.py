"""
Animation Of Mode Shapes module.
Part of the pyOMA2 package.
Authors:
Dag Pasca
Diego Margoni
"""

from __future__ import annotations

import tkinter as tk
import typing

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3D
from pyoma2.algorithm.data.geometry import Geometry2
from pyoma2.algorithm.data.result import BaseResult, MsPoserResult
from pyoma2.functions import Gen_funct, plot_funct


def animate_scatters(
    iteration, Data, scatters, lines=None, sens_lines=None, colormap="plasma", scaleF=1
):
    """
    Update the data held by the scatter plot and the lines connecting them, with colors based on their distance from the starting position.
    """
    # Prepare colormap
    cmap = plt.get_cmap(colormap)
    norm_fn = Normalize(vmin=0, vmax=scaleF)

    # Calculate distances from starting positions using numpy.linalg.norm
    distances = np.linalg.norm(Data[:, :, iteration] - Data[:, :, 0], axis=1)
    distances_normalized = norm_fn(distances)

    # Update scatter points
    for i, scatter in enumerate(scatters):
        if Data.shape[1] == 3:
            scatter._offsets3d = (
                Data[i, 0:1, iteration],
                Data[i, 1:2, iteration],
                Data[i, 2:, iteration],
            )
        else:
            scatter.set_offsets((Data[i, 0, iteration], Data[i, 1, iteration]))

        # Set color based on normalized distance
        scatter.set_color(cmap(distances_normalized[i]))

    # Update lines if applicable
    if lines is not None and sens_lines is not None:
        for line, (idx1, idx2) in zip(lines, sens_lines):
            line.set_data(
                Data[[idx1, idx2], 0, iteration], Data[[idx1, idx2], 1, iteration]
            )
            line.set_3d_properties(Data[[idx1, idx2], 2, iteration])

            # Set line color based on the average normalized distance of the two points
            avg_distance = np.mean(
                [distances_normalized[idx1], distances_normalized[idx2]]
            )
            line.set_color(cmap(avg_distance))

    return scatters + lines if lines is not None else scatters


# =============================================================================
# PLOTTING CLASS
# =============================================================================


class AniMode:
    """
    A class for animating 3D mode shapes in Operational Modal Analysis (OMA).

    Parameters
    ----------
    Geo : Geometry2
        Geometry object containing nodes and sensor information.
    Res : Union[BaseResult, MsPoserResult]
        Result object containing modal analysis data.
    mode_numb : int
        Mode number to visualize.
    scaleF : int, optional
        Scale factor for mode shape animation. Default is 1.
    view : str, optional
        View for the 3D plot. Can be '3D', 'xy', 'xz', 'yz', 'x', 'y', 'z'. Default is '3D'.
    remove_fill : bool, optional
        If True, removes the fill from the plot. Default is True.
    remove_grid : bool, optional
        If True, removes the grid from the plot. Default is True.
    remove_axis : bool, optional
        If True, turns off the axis lines, labels, and ticks. Default is True.
    saveGIF : bool, optional
        If True, saves the animation as a GIF file. Default is False.
    args, kwargs : additional arguments
        Additional arguments passed to the underlying plotting function.

    Attributes
    ----------
    root : tkinter.Tk
        The Tkinter root window for the animation.
    fig : matplotlib.figure.Figure
        The matplotlib figure object for the animation.
    ax : matplotlib.axes.Axes
        The axes object for the 3D plot.
    ani : matplotlib.animation.FuncAnimation
        The animation object.

    Methods
    -------
    plot3D() :
        Sets up and starts the 3D animation of the mode shape.
    on_closing() :
        Handles the closing event of the Tkinter window.
    """

    def __init__(
        self,
        Geo: Geometry2,
        Res: typing.Union[BaseResult, MsPoserResult],
        mode_numb: int,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
        saveGIF: bool = False,
        *args,
        **kwargs,
    ) -> typing.Any:
        self.Geo = Geo
        self.Res = Res

        self.mode_numb = mode_numb
        self.scaleF = scaleF
        self.view = view
        self.remove_axis = remove_axis
        self.remove_fill = remove_fill
        self.remove_grid = remove_grid
        self.saveGIF = saveGIF

        self.nodes_coord = self.Geo.pts_coord
        self.sens_names = self.Geo.sens_names

        self.Fn = self.Res.Fn
        self.fn = self.Res.Fn[int(self.mode_numb - 1)]

        self.phi = self.Res.Phi[:, int(self.mode_numb - 1)]

        if self.Res.Xi is not None:
            self.Xi = self.Res.Xi

        self.MCF = Gen_funct.MCF(self.phi)[0]

        self.root = tk.Tk()
        # self.root.title(f'Mode n*{self.mode_numb}, MCF = {self.MCF}')
        self.fig = Figure(figsize=(8, 8), tight_layout=True)

        canvas = FigureCanvasTkAgg(self.fig, self.root)
        canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

        # =============================================================================
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        NavigationToolbar2Tk(canvas, self.root)

        canvas.draw()
        self.plot3D()
        # =============================================================================

        # Integrate matplotlib figure
        self.root.mainloop()

    def plot3D(self):
        """
        Sets up and starts the 3D animation of the mode shape.

        This method creates a 3D animated plot of the selected mode shape, scaled and oriented
        according to the specified parameters. It uses the geometry and mode shape data to
        animate the mode shape over time.
        """
        nr_iter = 40

        # create mode shape dataframe
        df_phi = pd.DataFrame(
            {"sName": self.sens_names, "Phi": self.phi.real * self.scaleF},
        )
        if self.Geo.cstrn is not None:
            aa = self.Geo.cstrn.to_numpy(na_value=0)[:, :]
            aa = np.nan_to_num(aa, copy=True, nan=0.0)
            val = aa @ self.phi.real
            ctn_df = pd.DataFrame(
                {"cName": self.Geo.cstrn.index, "val": val},
            )

            mapping = dict(zip(df_phi["sName"], df_phi["Phi"]))
            mapping1 = dict(zip(ctn_df["cName"], ctn_df["val"]))
            mapp = dict(mapping, **mapping1)
        else:
            mapp = dict(zip(df_phi["sName"], df_phi["Phi"]))
        # reshape the mode shape dataframe to fit the pts coord
        sens_map = self.Geo.sens_map
        df_phi_map = sens_map.replace(mapp).astype(float)
        # add together coordinates and mode shape displacement
        # newpoints = self.nodes_coord.add(df_phi_map * self.Geo.sens_sign, fill_value=0)
        # newpoints1 = self.nodes_coord.add(
        #     df_phi_map * self.Geo.sens_sign * (-1), fill_value=0
        # )
        newpoints = (
            self.Geo.pts_coord.to_numpy()
            + df_phi_map.to_numpy() * self.Geo.sens_sign.to_numpy()
        )
        newpoints1 = (
            self.Geo.pts_coord.to_numpy()
            + df_phi_map.to_numpy() * self.Geo.sens_sign.to_numpy() * (-1)
        )
        # extract only the displacement array
        # newpoints = newpoints.to_numpy()[:, 1:]
        # newpoints1 = newpoints1.to_numpy()[:, 1:]
        oldpoints = self.nodes_coord.to_numpy()[:, :]
        # Create trajectories (first quarter than reverse and concatenate)
        traj1 = np.linspace(oldpoints, newpoints, int(nr_iter / 4))
        traj2 = np.concatenate([traj1, traj1[::-1]])
        # Create trajectories (third quarter than reverse and concatenate)
        traj3 = np.linspace(oldpoints, newpoints1, int(nr_iter / 4))
        traj4 = np.concatenate([traj3, traj3[::-1]])
        # concatenate all traj
        traj5 = np.concatenate([traj2, traj4])
        # Finally assemble data
        Data = np.moveaxis(traj5, 0, -1)

        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_title(f"Mode nr. {self.mode_numb}, $f_n$={self.fn:.3f}Hz")
        # Setting the axes properties
        self.ax.set_xlim3d([np.min(Data[:, 0, :]), np.max(Data[:, 0, :])])
        self.ax.set_xlabel("X")

        self.ax.set_ylim3d([np.min(Data[:, 1, :]), np.max(Data[:, 1, :])])
        self.ax.set_ylabel("Y")

        self.ax.set_zlim3d([np.min(Data[:, 2, :]), np.max(Data[:, 2, :])])
        self.ax.set_zlabel("Z")

        # Set ax options
        plot_funct.set_ax_options(
            self.ax,
            bg_color="w",
            remove_fill=self.remove_fill,
            remove_grid=self.remove_grid,
            remove_axis=self.remove_axis,
            scaleF=self.scaleF,
        )

        # Set view
        plot_funct.set_view(self.ax, view=self.view)

        # Check that BG nodes are defined
        if self.Geo.bg_nodes is not None:
            # if True plot
            plot_funct.plt_nodes(self.ax, self.Geo.bg_nodes, color="gray", alpha=0.5)
            # Check that BG lines are defined
            if self.Geo.bg_lines is not None:
                # if True plot
                plot_funct.plt_lines(
                    self.ax, self.Geo.bg_nodes, self.Geo.bg_lines, color="gray", alpha=0.5
                )
            if self.Geo.bg_surf is not None:
                # if True plot
                plot_funct.plt_surf(
                    self.ax, self.Geo.bg_nodes, self.Geo.bg_surf, alpha=0.1
                )

        # Initialize scatters
        scatters = [
            self.ax.scatter(Data[i, 0:1, 0], Data[i, 1:2, 0], Data[i, 2:, 0], color="r")
            for i in range(Data.shape[0])
        ]
        if self.Geo.sens_lines is not None:
            # Initialize lines based on sens_lines
            lines = [
                self.ax.add_line(Line3D([], [], []))
                for _ in range(len(self.Geo.sens_lines))
            ]
            # Creating the Animation object
            self.ani = animation.FuncAnimation(
                self.fig,
                animate_scatters,
                nr_iter,
                fargs=(Data, scatters, lines, self.Geo.sens_lines, "plasma", self.scaleF),
                interval=1,
                blit=False,
                repeat=True,
                # frames=len(scatters)//20
            )

        else:
            self.ani = animation.FuncAnimation(
                self.fig,
                animate_scatters,
                nr_iter,
                fargs=(Data, scatters, None, None, "plasma", self.scaleF),
                interval=1,
                blit=False,
                repeat=True,
                # frames=len(scatters)//20
            )

        plt.show()
        if self.saveGIF is True:
            writer = animation.PillowWriter(
                fps=15, metadata=dict(artist="Me"), bitrate=1800
            )
            self.ani.save(f"mode_numb_{self.mode_numb}.gif", writer=writer)

    def on_closing(self):
        """
        Handles the closing event of the Tkinter window.

        This method ensures proper termination of the animation and Tkinter window upon closing.
        """
        self.root.quit()
        self.root.destroy()
