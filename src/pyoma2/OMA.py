from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt


from pyoma2.functions.plot_funct import plt_data

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
        sens_name: npt.NDArray[np.string] | list[str],
        # FIXME sens coor e meglio se lo facciamo come dataframe
        sens_coord: npt.NDArray[np.float32], # sensors' coordinates
        sens_dir: npt.NDArray[np.float32], # sensors' directions (array(n,3))
        ## OPTIONAL
        sens_lines: npt.NDArray[np.float32] = None, # lines connection sensors (array(n,2))
        bg_nodes: npt.NDArray[np.float32] = None, # Background nodes
        bg_lines: npt.NDArray[np.float32] = None, # Background lines

    ):
        pass

    # metodo per plottare geometria 1
    def plot_geo1(
        self,
        ):
        pass

    # metodo per definire geometria 2
    def def_geo2(
        self,
        ## MANDATORY
        # uno tra ["None", "xy","xz","yz","x","y","z"]
        sens_name: npt.NDArray[np.string],# sensors' names (n, 1)
        # FIXME sens coor e meglio se lo facciamo come dataframe
        pts_coord: npt.NDArray[np.float32], # points' coordinates (n, 4or3)
        # sens_sign: npt.NDArray[np.float32], # sensors' sign (n, 1)
        sens_map: npt.NDArray[np.float32], # mapping (n, 3)
        ## OPTIONAL
        order_red: None | str = None, 
        sens_dir: npt.NDArray[np.float32] = None, # sensors' directions (array(n,3))
        sens_lines: npt.NDArray[np.float32] = None, # lines connection sensors (array(n,2))
        bg_nodes: npt.NDArray[np.float32] = None, # Background nodes
        bg_lines: npt.NDArray[np.float32] = None, # Background lines
        # bg_surf: npt.NDArray[np.float32], # Background surfaces
    ):
        pass

    # metodo per plottare geometria 2
    def plot_geo2(
        self,
        ):
        pass

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
class MultiSetup:
    """Come il Single setup ma prende nell'init una lista degli stessi argomenti"""

    def __init__(self, data: list[typing.Iterable[float]], fs: list[float]):
        self.data = data
        self.fs = fs
        self.setups = []

    def add_setups(self, *setups: SingleSetup):
        self.setups.extend(setups)

    def run_all(self):
        for setup in self.setups:
            setup.run_all()
        print("all done")

    def merge_results(self):
        for setup in self.setups:
            for alg in setup.algorithms:
                if not alg.result:
                    raise ValueError(f"Run {alg.name} first")
                print(f"Merging {alg.name} results")
