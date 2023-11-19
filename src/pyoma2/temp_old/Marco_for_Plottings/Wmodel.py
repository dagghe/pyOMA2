from typing import Union

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from utils import *


class Wmodel:
    """
    This object type stores all information to perform 3D plots and 2D projections of a wireframe model of the monitored buidling and plotting the mode shapes for a certain monitored setup as arrows.
    """

    def __init__(
        self, nodes_coord: np.ndarray, connectivity: np.ndarray, mode: list
    ) -> None:
        """
        ----------
        Parameters
        ----------
        nodes_coord : array (1D or 2D)
            Node coordinates of the wireframe model. Units [m].
        connectivity : array (1D or 2D)
            Connectivity matrix of the nodes_coord array to properly reconstruct the wireframe model plot.
        mode : list
            List containing all the necessary information to properly plot the mode shape. The user can use the MonSetup class object to obtain the mode list in the right format.
            Example: mode = [ mode_shape , mon_nodes_dofs, freq , method] : list
                mode_shape : is the mode shape column array of interest.
                mon_nodes_dofs : is the array containing information to properly connect each mode shape component to the proper node number and dof.
                freq : the natural frequency associated to this mode.
                method : reminder of the method used to extract the mode shape of interest.
        -------
        """
        self.nodes_coord = nodes_coord
        self.connectivity = connectivity
        self.nodes_numbering = np.arange(1, self.nodes_coord.shape[0] + 1)

        self._nodes = np.hstack((self.nodes_numbering.reshape(-1, 1), self.nodes_coord))

        self.mode_shape = mode[0]
        self._mode_nodes_dofs = mode[1]
        self.freq = mode[2]
        self.method = mode[3]

        self.mode_nodes = self._mode_nodes_dofs[0, :]
        self.mode_dofs = self._mode_nodes_dofs[1:, :]

    def plot_mode_shape3D(
        self,
        scfc: Union[int, float],
        kwargsWmodel: dict,
        kwargsmarkers: dict,
        kwargsmonnodes: dict,
        kwargsannotations: dict,
        kwargsarrows: dict,
        save_to_file_path: str = "",
    ):
        """
        This method generates the 3D wireframe model of the monitored structure, indicating the monitored nodes and the mode shape like arrows according to the monitored degree of freedoms (dofs).

        ----------
        Parameters
        ----------
        scfc : int or float
            The scale factor applied to the mode shape to better visualize the arrows magnitudes. The scale factor applies equally to the three x,y,z dofs components.
        kwargsWmodel : dictionary
            Dictionary of keyword arguments to customize the plot style of lines of the wireframe model.
        kwargsmarkers : dictionary
            Dictionary of keyword arguments to customize the plot style of the markers of the wireframe model.
        kwargsmonnodes : dictionary
            Dictionary of keyword arguments to customize the plot style of the markers related to the monitored nodes.
        kwargsannotations : dictionary
            Dictionary of keyword arguments to customize the plot style of the annotations related to the monitored nodes.
        kwargsarrows : dictionary
            Dictionary of keyword arguments to customize the plot style of the arrows related to the mode shapes' components.
        save_to_file_path : string
            If provided, the plots are stored to a file in the indicated path both as a raster (png) and vectorial (pdf) format.
        -------
        Returns
        -------
        figures : 3D matplotlib figure
        """
        fig, ax = plot3Dframe(
            self.nodes_coord,
            self.connectivity,
            kwargs_plot_lines=kwargsWmodel,
            kwargs_plot_markers=kwargsmarkers,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        plotted_mon_nodes = []
        for ii, monitored_nodes in enumerate(self.mode_nodes):
            if monitored_nodes not in plotted_mon_nodes:
                plotted_mon_nodes.append(monitored_nodes)
                ax.scatter(
                    *self.nodes_coord[monitored_nodes == self.nodes_numbering, :][0],
                    **kwargsmonnodes,
                )
                annotate3d(
                    ax,
                    f"{monitored_nodes:.0f}",
                    xyz=self.nodes_coord[monitored_nodes == self.nodes_numbering, :][0],
                    **kwargsannotations,
                )
        for ii, modal_component in enumerate(self.mode_shape):
            arrow3d(
                ax,
                *self._nodes[self.mode_nodes[ii] == self.nodes_numbering, 1:][0],
                *(scfc * modal_component * self.mode_dofs[:, ii]),
                **kwargsarrows,
            )
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.title(f"Freq. {self.freq:.2f} Hz")
        if len(save_to_file_path):
            plt.savefig(
                save_to_file_path
                + os.sep
                + f"MODE_shape_3D_freq_{self.freq:.2f}Hz_{self.method}"
                + ".pdf"
            )
            plt.savefig(
                save_to_file_path
                + os.sep
                + f"MODE_shape_3D_freq_{self.freq:2f}Hz_{self.method}"
                + ".png"
            )
            pickle.dump(
                fig,
                open(
                    save_to_file_path
                    + os.sep
                    + f"MODE_shape_3D_freq_{self.freq:.2f}Hz_{self.method}.pickle",
                    "wb",
                ),
            )
        return fig

    def plot_2D_top_view(
        self,
        zlevel: float,
        scfc: Union[int, float],
        kwargsWmodel: dict,
        kwargsmarkers: dict,
        kwargsmonnodes: dict,
        kwargsannotations: dict,
        kwargsarrows: dict,
        save_to_file_path: str = "",
        zspan=0.05,
    ):
        """
        This method generates 2D plan top views of the wireframe model cutted at a certain z level, to better visualize planar components of mode shapes floor by floor.

        ----------
        Parameters
        ----------
        zlevel : float
            It is the z level the user want to generate a 2D plan top view of the model [m].
        zspan : float (default 0.05)
            zspan defines which nodes retain in the planar top view [m]. In fact, the method defines a zlim list of the type:
            zlim = [ztop, zbottom]
            And the two z levels are computed according to the given zlevel of interest and a certain selection zspan level:
            ztop = zlevel + zspan
            zbottom = zlevel - zspan
        scfc : int or float
            The scale factor applied to the mode shape to better visualize the arrows magnitudes. The scale factor applies equally to the three x,y,z dofs components.
        kwargsWmodel : dictionary
            Dictionary of keyword arguments to customize the plot style of lines of the wireframe model.
        kwargsmarkers : dictionary
            Dictionary of keyword arguments to customize the plot style of the markers of the wireframe model.
        kwargsmonnodes : dictionary
            Dictionary of keyword arguments to customize the plot style of the markers related to the monitored nodes.
        kwargsannotations : dictionary
            Dictionary of keyword arguments to customize the plot style of the annotations related to the monitored nodes.
        kwargsarrows : dictionary
            Dictionary of keyword arguments to customize the plot style of the arrows related to the mode shapes' components.
        save_to_file_path : string
            If provided, the plots are stored to a file in the indicated path both as a raster (png) and vectorial (pdf) format.
        -------
        Returns
        -------
        figures : 3D matplotlib figure
        """
        # trovare tutti i nodi e connectivity per i nodi le cui z sono comprese in zlim=[ztop,zbottom]
        zlim = [zlevel + zspan, zlevel - zspan]

        mask = (
            (self.nodes_coord[:, 2] < zlim[0]).astype(int)
            * (self.nodes_coord[:, 2] > zlim[1]).astype(int)
        ).astype(dtype=bool)
        sel_nodes = self.nodes_numbering[mask]
        # sel_coord = self.nodes_coord[mask,:]

        mask = (
            np.in1d(self.connectivity[:, 0], sel_nodes).astype(int)
            * np.in1d(self.connectivity[:, 1], sel_nodes).astype(int)
        ).astype(dtype=bool)
        sel_connectivity = self.connectivity[mask, :]

        sel_mode_nodes = np.in1d(self.mode_nodes, sel_nodes).astype(int)
        # sel_mode_nodes_coord = sel_coord[ (np.in1d(sel_nodes, self.mode_nodes).astype(int)).astype(dtype=bool), :-1 ]

        fig, ax = plot2Dframe(
            self.nodes_coord[:, :-1],
            sel_connectivity,
            kwargs_plot_lines=kwargsWmodel,
            kwargs_plot_markers=kwargsmarkers,
        )

        plotted_mon_nodes = []
        for ii, monitored_nodes in enumerate(self.mode_nodes):
            if monitored_nodes not in plotted_mon_nodes and sel_mode_nodes[ii]:
                plotted_mon_nodes.append(monitored_nodes)
                ax.scatter(
                    *self.nodes_coord[monitored_nodes == self.nodes_numbering, :][0],
                    **kwargsmonnodes,
                )
                ax.annotate(
                    f"{monitored_nodes:.0f}",
                    xy=self.nodes_coord[monitored_nodes == self.nodes_numbering, :-1][0],
                    **kwargsannotations,
                )
        for ii, modal_component in enumerate(self.mode_shape):
            if sel_mode_nodes[ii] and not np.array_equal(
                self.mode_dofs[:, ii], np.array([0.0, 0.0, 1.0])
            ):
                # arrow(*self.nodes_coord[self.mode_nodes[ii]==self.nodes_numbering,:-1][0], \
                #         *(scfc * modal_component * self.mode_dofs[:,ii]) , **kwargsarrows)
                # arrow = mpatches.Arrow(self.nodes_coord[self.mode_nodes[ii]==self.nodes_numbering,0][0], self.nodes_coord[self.mode_nodes[ii]==self.nodes_numbering,1][0], \
                #                        (scfc * modal_component * self.mode_dofs[:,ii])[0], (scfc * modal_component * self.mode_dofs[:,ii])[1])
                # ax.add_patch(arrow)
                arrow = mpatches.FancyArrow(
                    self.nodes_coord[self.mode_nodes[ii] == self.nodes_numbering, 0][0],
                    self.nodes_coord[self.mode_nodes[ii] == self.nodes_numbering, 1][0],
                    (scfc * modal_component * self.mode_dofs[:, ii])[0],
                    (scfc * modal_component * self.mode_dofs[:, ii])[1],
                    length_includes_head=True,
                    head_starts_at_zero=True,
                    **kwargsarrows,
                )
                ax.add_patch(arrow)
                # ax.annotate("", \
                #             xy=(self.nodes_coord[self.mode_nodes[ii]==self.nodes_numbering,0][0], self.nodes_coord[self.mode_nodes[ii]==self.nodes_numbering,1][0]),\
                #             xytext=(self.nodes_coord[self.mode_nodes[ii]==self.nodes_numbering,0][0] - (scfc * modal_component * self.mode_dofs[:,ii])[0] ,
                #                     self.nodes_coord[self.mode_nodes[ii]==self.nodes_numbering,1][0] - (scfc * modal_component * self.mode_dofs[:,ii])[1] ),
                #             arrowprops=kwargsarrows)
        plt.title(f"Freq. {self.freq:.2f} Hz, top view from z={zlim[0]:.1f} m")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.tight_layout()
        if len(save_to_file_path):
            plt.savefig(
                save_to_file_path
                + os.sep
                + f"MODE_shape_2D_freq_{self.freq:.2f}Hz_view_from_z_{zlim[0]:.2f}m_{self.method}"
                + ".pdf"
            )
            plt.savefig(
                save_to_file_path
                + os.sep
                + f"MODE_shape_2D_freq_{self.freq:.2f}Hz_view_from_z_{zlim[0]:.2f}m_{self.method}"
                + ".png"
            )
            pickle.dump(
                fig,
                open(
                    save_to_file_path
                    + os.sep
                    + f"MODE_shape_2D_freq_{self.freq:.2f}Hz_view_from_z_{zlim[0]:.2f}m_{self.method}.pickle",
                    "wb",
                ),
            )
        return fig

    def plot_2D_lateral_view_X(self):
        # ancora da implementare
        pass

    def plot_2D_lateral_view_Y(self):
        # ancora da implementare
        pass
