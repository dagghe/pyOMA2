from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt

import pandas as pd
import matplotlib.pyplot as plt

from pyoma2.functions.plot_funct import( 
plt_data, plt_nodes,plt_lines,plt_quiver,plt_surf,set_ax_options,set_view)
from pyoma2.functions.Gen_funct import find_map
from pyoma2.algorithm.data.geometry import Geometry1,Geometry2
from pyoma2.functions.Gen_funct import merge_mode_shapes, PRE_MultiSetup  # noqa: F401



# Questa non l ho capita
if typing.TYPE_CHECKING:
    from pyoma2.algorithm import BaseAlgorithm


class SingleSetup:
    def __init__(self, data: typing.Iterable[float], fs: float):
        self.data = data  # data
        self.fs = fs  # sampling frequency
        self.dt = 1 / fs  # sampling interval
        self.algorithms: dict[str, BaseAlgorithm] = {}  # set of algo

    # add algorithm (method) to the set.
    def add_algorithms(self, *algorithms: BaseAlgorithm):
        """
        Add algorithms to the set and set the data and fs.
        N.B:
            the algorithms must be instantiated before adding them to the set.
            algorithms names must be unique.
        """
        self.algorithms = {
            **self.algorithms,
            **{alg.name: alg.set_data(data=self.data, fs=self.fs) for alg in algorithms},
        }

    # run the whole set of algorithms (methods). METODO 1 di tutti
    def run_all(self):
        for alg_name in self.algorithms:
            self.run_by_name(name=alg_name)
        print("all done")

    # run algorithm (method) by name. QUESTO è IL METODO 1 di un singolo
    def run_by_name(self, name: str):
        """Run an algorithm by its name and save the result in the algorithm itself."""
        print(f"Running {name}...with parameters: {self[name].run_params}")
        result = self[name].run()
        print(f"...saving {name} result\n")
        self[name].set_result(result)

    # get the modal properties (all results).
    #  QUESTO è IL METODO 2 (manuale)
    def MPE(self, name: str, *args, **kwargs):
        print(f"Getting MPE modal parameters from {name}")
        self[name].mpe(*args, **kwargs)

    # get the modal properties (all results) from the plots.
    # QUESTO è IL METODO 2 (grafico)
    def MPE_fromPlot(self, name: str, *args, **kwargs):
        print(f"Getting MPE modal parameters from plot... {name}")
        self[name].mpe_fromPlot(*args, **kwargs)

    # method to plot the time histories of the data channels.
    def plot_data(
        self,
        nc: int = 1,
        names: None | list[str] = None,
        unit: str = "unit",
        show_rms: bool = False,
    ):
        data = self.data
        dt = self.dt
        nc = nc  # number of columns for subplot
        names = names  # list of names (str) of the channnels
        unit = unit  # str label for the y-axis (unit of measurement)
        show_rms = show_rms  # wheter to show or not the rms acc in the plot
        fig, ax = plt_data(data, dt, nc, names, unit, show_rms)
        return fig, ax

    # metodo per definire geometria 1
    def def_geo1(
        self,
        ## MANDATORY
        sens_names: npt.NDArray[np.string] | list[str], # sensors' names
        sens_coord: pd.DataFrame, # sensors' coordinates
        sens_dir: npt.NDArray[np.int64], # sensors' directions
        ## OPTIONAL
        sens_lines: npt.NDArray[np.int64] = None, # lines connecting sensors 
        bg_nodes: npt.NDArray[np.float64] = None, # Background nodes
        bg_lines: npt.NDArray[np.int64] = None, # Background lines
        bg_surf: npt.NDArray[np.float64] = None, # Background surfaces
    ):
# =============================================================================
# Checks on input       
        nr_s = len(sens_names)
        # check that nr_s == to data.shape[1]
        assert nr_s == self.data.shape[1]
        # check that nr_s == sens_coord.shape[0] and == sens_dir.shape[0]
        assert nr_s == sens_coord.to_numpy().shape[0]
        assert nr_s == sens_dir.shape[0]
# Altri controlli ???
# =============================================================================
        # adapt to 0 indexing
        if bg_lines is not None:
            bg_lines = np.subtract(bg_lines,1)

        # Find the indices that rearrange sens_coord to sens_names
        newIDX = find_map(sens_names, sens_coord["sName"].to_numpy())
        # reorder if necessary
        sens_coord = sens_coord.reindex(labels=newIDX)
        sens_dir = sens_dir[newIDX,:]
        # # Transform into numpy array
        # sens_coord= sens_coord[["x","y","z"]].to_numpy()

        self.Geo1 = Geometry1(sens_names=sens_names,
                              sens_coord=sens_coord,
                              sens_dir=sens_dir,
                              sens_lines=sens_lines,
                              bg_nodes=bg_nodes,
                              bg_lines=bg_lines,
                              bg_surf=bg_surf)

    # metodo per plottare geometria 1
    def plot_geo1(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D","xy","xz","yz","x","y","z"] = "3D",
        remove_fill: True | False =True, 
        remove_grid: True | False =True, 
        remove_axis: True | False =True
        ):
        
        fig = plt.figure(figsize=(10,10),tight_layout=True)
        ax = fig.add_subplot(111, projection='3d')

        # plot sensors' nodes
        sens_coord= self.Geo1.sens_coord[["x","y","z"]].to_numpy()
        plt_nodes(ax,sens_coord,color="red")

        # plot sensors' directions
        plt_quiver(ax,sens_coord,self.Geo1.sens_dir,
                   scaleF=scaleF,names=self.Geo1.sens_names)

        # Check that BG nodes are defined
        if self.Geo1.bg_nodes is not None:
            # if True plot
            plt_nodes(ax,self.Geo1.bg_nodes,color="gray",alpha=0.5)
            # Check that BG lines are defined
            if self.Geo1.bg_lines is not None:
                # if True plot
                plt_lines(ax,self.Geo1.bg_nodes,self.Geo1.bg_lines,
                      color="gray",alpha=0.5)
            if self.Geo1.bg_surf is not None:
                # if True plot
                plt_surf(ax,self.Geo1.bg_nodes,self.Geo1.bg_surf
                         ,alpha=0.1)

        # check for sens_lines
        if self.Geo1.sens_lines is not None:
            # if True plot
            plt_lines(ax,sens_coord,self.Geo1.sens_lines,
                      color="red")

        # Set ax options
        set_ax_options(ax,bg_color="w",
                       remove_fill=remove_fill,
                       remove_grid=remove_grid,
                       remove_axis=remove_axis)

        # Set view
        set_view(ax, view=view)

        return fig, ax

    # metodo per definire geometria 2
    def def_geo2(
        self,
        ## MANDATORY
        sens_names: npt.NDArray[np.string] | list[str],# sensors' names 
        pts_coord: pd.DataFrame, # points' coordinates 
        sens_map: pd.DataFrame, # mapping 
        sens_sign: pd.DataFrame,
        ## OPTIONAL
        order_red: None | typing.Literal["xy","xz","yz","x","y","z"] = None, 
        sens_lines: npt.NDArray[np.int64] = None, # lines connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None, # Background nodes
        bg_lines: npt.NDArray[np.float64] = None, # Background lines
        bg_surf: npt.NDArray[np.float64] = None, # Background lines
    ):
# =============================================================================
# Checks on input     
        if order_red == "xy" or order_red == "xz" or order_red == "yz":
            nc = 2
            assert sens_map.to_numpy()[:,1:].shape[1] == nc
            assert sens_sign.to_numpy()[:,1:].shape[1] == nc
        elif order_red == "x" or order_red == "y" or order_red == "z":
            nc = 1
            assert sens_map.to_numpy()[:,1:].shape[1] == nc
            assert sens_sign.to_numpy()[:,1:].shape[1] == nc
        elif order_red == None:
            nc = 3
            assert sens_map.to_numpy()[:,1:].shape[1] == nc
            assert sens_sign.to_numpy()[:,1:].shape[1] == nc
# FIXME Controllo su Dimensioni (DA FARE)

# =============================================================================
        # adapt to 0 indexed lines
        if bg_lines is not None:
            bg_lines = np.subtract(bg_lines,1)
        if sens_lines is not None:
            sens_lines = np.subtract(sens_lines,1)
        

        self.Geo2 = Geometry2(sens_names=sens_names,
                              pts_coord=pts_coord,
                              sens_map=sens_map,
                              sens_sign=sens_sign,
                              order_red=order_red,
                              sens_lines=sens_lines,
                              bg_nodes=bg_nodes,
                              bg_lines=bg_lines,
                              bg_surf=bg_surf)

    # metodo per plottare geometria 2
    def plot_geo2(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D","xy","xz","yz","x","y","z"] = "3D",
        remove_fill: True | False =True, 
        remove_grid: True | False =True, 
        remove_axis: True | False =True
        ):
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        # plot sensors' nodes
        plt_nodes(ax,self.Geo2.pts_coord,color="red")
        
        # plot sensors' directions
        plt_quiver(ax,self.Geo2.pts_coord,self.Geo2.sens_map,
                   scaleF=scaleF,names=self.Geo2.sens_names)

        # Check that BG nodes are defined
        if self.Geo2.bg_nodes is not None:
            # if True plot
            plt_nodes(ax,self.Geo2.bg_nodes,color="gray",alpha=0.5)
            # Check that BG lines are defined
            if self.Geo2.bg_lines is not None:
                # if True plot
                plt_lines(ax,self.Geo2.bg_nodes,self.Geo2.bg_lines,
                      color="gray",alpha=0.5)
            if self.Geo2.bg_surf is not None:
                # if True plot
                plt_surf(ax,self.Geo2.bg_nodes,self.Geo2.bg_surf
                         ,alpha=0.1)

        # check for sens_lines
        if self.Geo2.sens_lines is not None:
            # if True plot
            plt_lines(ax,self.Geo2.pts_coord,self.Geo2.sens_lines,
                      color="red")

        # Set ax options
        set_ax_options(ax,bg_color="w",
                       remove_fill=remove_fill,
                       remove_grid=remove_grid,
                       remove_axis=remove_axis)

        # Set view
        set_view(ax, view=view)

    def __getitem__(self, name: str) -> BaseAlgorithm:
        """
        Retrieve an algorithm from the set by its name.
        Raises a KeyError if the algorithm does not exist.
        """
        if name in self.algorithms:
            return self.algorithms[name]
        else:
            raise KeyError(f"No algorithm named '{name}' exists.")

    def get(
        self, name: str, default: BaseAlgorithm | None = None
    ) -> BaseAlgorithm | None:
        """
        Retrieve an algorithm from the set by its name.
        Returns the default value if the algorithm does not exist.
        """
        return self.algorithms.get(name, default)


# LA CLASSE MULTISETUP VA PROBABILMENTE RIVISTA...
class MultiSetupPost_V1:
    """Prende una lista di single setups e mergia tutti i risultati di tutti gli algoritmi di tutti i setup."""

    def __init__(self, fs: float, ref_ind: list[list[int]] , single_setups: list[SingleSetup] | None = None):
        self.setups = [el for el in self._init_setups(single_setups)] if single_setups else []
        self.fs = fs
        self.ref_ind = ref_ind

    def _init_setups(self, setups: list[SingleSetup]) -> typing.Generator[SingleSetup, None, None]:
        """Set the data and fs for all the setups."""
        for setup in setups:
            # force the same fs for all the setups
            setup.fs = self.fs
            # ensure that all the algorithms have the same fs
            for alg in setup.algorithms.values():
                if alg.result is None:
                    alg.run()
            yield setup

    def add_setups(self, *setups: SingleSetup):
        self.setups.extend(
            [el for el in self._init_setups(setups)]
        )

    def run_all(self):
        for setup in self.setups:
            setup.run_all()
        print("all done")

    def merge_results(self) -> npt.NDArray[np.float64]:
        results = []
        for setup in self.setups:
            for alg in setup.algorithms.values():
                print(f"Merging {alg.name} results")
                results.append(alg.result.Phi)

        return merge_mode_shapes(MSarr_list=results, reflist=self.ref_ind)
    


class MultiSetupPre_V1:
    """
    Qui non ho capito, abbiamo dei dati che arrivano da dove? poi su questi dati facciamo
    un merge e sul merge applichiamo un algoritmo comune 
    """