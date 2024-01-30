from __future__ import annotations

import copy
import logging
import typing

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import decimate, detrend

from pyoma2.algorithm.data.geometry import Geometry1, Geometry2
from pyoma2.algorithm.data.result import MsPoserResult
from pyoma2.functions.Gen_funct import (
    PRE_MultiSetup,
    find_map,
    merge_mode_shapes,
)
from pyoma2.functions.plot_funct import (
    plt_data,
    plt_lines,
    plt_nodes,
    plt_quiver,
    plt_surf,
    set_ax_options,
    set_view,
    plt_ch_info
)

if typing.TYPE_CHECKING:
    from pyoma2.algorithm import BaseAlgorithm

from pyoma2.plot.anim_mode import AniMode

logger = logging.getLogger(__name__)


class BaseSetup:
    algorithms: typing.Dict[str, BaseAlgorithm]
    data: typing.Optional[np.ndarray] = None  # TODO use generic typing
    fs: typing.Optional[float] = None  # sampling frequency

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
        logger.info("all done")

    # run algorithm (method) by name. QUESTO è IL METODO 1 di un singolo
    def run_by_name(self, name: str):
        """Run an algorithm by its name and save the result in the algorithm itself."""
        logger.info("Running %s...", name)
        logger.debug("...with parameters: %s", self[name].run_params)
        result = self[name].run()
        logger.debug("...saving %s result", name)
        self[name].set_result(result)

    # get the modal properties (all results).
    def MPE(self, name: str, *args, **kwargs):
        logger.info("Getting MPE modal parameters from %s", name)
        self[name].mpe(*args, **kwargs)

    # get the modal properties (all results) from the plots.
    def MPE_fromPlot(self, name: str, *args, **kwargs):
        logger.info("Getting MPE modal parameters from plot... %s", name)
        self[name].mpe_fromPlot(*args, **kwargs)

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
        self, name: str, default: typing.Optional[BaseAlgorithm] = None
    ) -> typing.Optional[BaseAlgorithm]:
        """
        Retrieve an algorithm from the set by its name.
        Returns the default value if the algorithm does not exist.
        """
        return self.algorithms.get(name, default)

    # metodo per plottare geometria 1
    def plot_geo1(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
    ):

        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        # plot sensors' nodes
        sens_coord = self.Geo1.sens_coord[["x", "y", "z"]].to_numpy()
        plt_nodes(ax, sens_coord, color="red")

        # plot sensors' directions
        plt_quiver(
            ax, sens_coord, self.Geo1.sens_dir, scaleF=scaleF, names=self.Geo1.sens_names
        )

        # Check that BG nodes are defined
        if self.Geo1.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, self.Geo1.bg_nodes, color="gray", alpha=0.5)
            # Check that BG lines are defined
            if self.Geo1.bg_lines is not None:
                # if True plot
                plt_lines(
                    ax, self.Geo1.bg_nodes, self.Geo1.bg_lines, color="gray", alpha=0.5
                )
            if self.Geo1.bg_surf is not None:
                # if True plot
                plt_surf(ax, self.Geo1.bg_nodes, self.Geo1.bg_surf, alpha=0.1)

        # check for sens_lines
        if self.Geo1.sens_lines is not None:
            # if True plot
            plt_lines(ax, sens_coord, self.Geo1.sens_lines, color="red")

        # Set ax options
        set_ax_options(
            ax,
            bg_color="w",
            remove_fill=remove_fill,
            remove_grid=remove_grid,
            remove_axis=remove_axis,
        )

        # Set view
        set_view(ax, view=view)

        return fig, ax

    # metodo per plottare geometria 2
    def plot_geo2(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
    ):

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # plot sensors' nodes
        plt_nodes(ax, self.Geo2.pts_coord, color="red")

        # plot sensors' directions
        plt_quiver(
            ax,
            self.Geo2.pts_coord,
            self.Geo2.sens_map,
            scaleF=scaleF,
            names=self.Geo2.sens_names,
        )

        # Check that BG nodes are defined
        if self.Geo2.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, self.Geo2.bg_nodes, color="gray", alpha=0.5)
            # Check that BG lines are defined
            if self.Geo2.bg_lines is not None:
                # if True plot
                plt_lines(
                    ax, self.Geo2.bg_nodes, self.Geo2.bg_lines, color="gray", alpha=0.5
                )
            if self.Geo2.bg_surf is not None:
                # if True plot
                plt_surf(ax, self.Geo2.bg_nodes, self.Geo2.bg_surf, alpha=0.1)

        # check for sens_lines
        if self.Geo2.sens_lines is not None:
            # if True plot
            plt_lines(ax, self.Geo2.pts_coord, self.Geo2.sens_lines, color="red")

        # Set ax options
        set_ax_options(
            ax,
            bg_color="w",
            remove_fill=remove_fill,
            remove_grid=remove_grid,
            remove_axis=remove_axis,
        )

        # Set view
        set_view(ax, view=view)

        return fig, ax


class SingleSetup(BaseSetup):
    def __init__(self, data: typing.Iterable[float], fs: float):
        self.data = data  # data
        self.fs = fs  # sampling frequency
        self.dt = 1 / fs  # sampling interval
        self.algorithms: typing.Dict[str, BaseAlgorithm] = {}  # set of algo

    # method to plot the time histories of the data channels.
    def plot_data(
        self,
        nc: int = 1,
        names: typing.Optional[typing.List[str]] = None,
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

    # method to plot TH, PSD and KDE for each channel
    def plot_ch_info(
            self,
            ch_idx: str | list[int] = "all",
            ch_names: typing.Optional[typing.List[str]] = None,
            freqlim: float | None = None,
            logscale: bool = True,
            nxseg: float | None = None,
            pov: float = 0.,
            window: str = "boxcar"
    ):

        data = self.data
        fs = self.fs

        fig, ax = plt_ch_info(data, fs, ch_idx, ch_names=ch_names,
                              freqlim=freqlim, logscale=logscale,
                              nxseg=nxseg, pov=pov, window=window)
        return fig, ax


    # method to decimate data
    def decimate_data(
            self,
            q:int,
            n: int|None=None,
            ftype:typing.Literal["iir", "fir"]='iir',
            axis:int =-1,
            zero_phase: bool =True):
        """
wrapper method for scipy.signal.decimate function
"""

        self.data = decimate(self.data,q,n,ftype,axis,zero_phase)
        self.fs = self.fs / q
        self.dt = 1 / self.fs

    # method to detrend data
    def detrend_data(
            self,
            axis:int =-1,
            type:typing.Literal["linear", "constant"]='linear',
            bp:  int | npt.NDArray[np.int64] = 0,
            ):
        """
wrapper method for scipy.signal.detrend function
"""
        self.data = detrend(self.data,axis,type,bp)

    # metodo per definire geometria 1
    def def_geo1(
        self,
        # # MANDATORY
        sens_names: typing.Union[
            npt.NDArray[np.string], typing.List[str]
        ],  # sensors' names
        sens_coord: pd.DataFrame,  # sensors' coordinates
        sens_dir: npt.NDArray[np.int64],  # sensors' directions
        # # OPTIONAL
        sens_lines: npt.NDArray[np.int64] = None,  # lines connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None,  # Background nodes
        bg_lines: npt.NDArray[np.int64] = None,  # Background lines
        bg_surf: npt.NDArray[np.float64] = None,  # Background surfaces
    ):
        # ---------------------------------------------------------------------
        # Checks on input
        nr_s = len(sens_names)
        # check that nr_s == to data.shape[1]
        assert nr_s == self.data.shape[1]
        # check that nr_s == sens_coord.shape[0] and == sens_dir.shape[0]
        assert nr_s == sens_coord.to_numpy().shape[0]
        assert nr_s == sens_dir.shape[0]
        # Altri controlli ???
        # ---------------------------------------------------------------------
        # adapt to 0 indexing
        if bg_lines is not None:
            bg_lines = np.subtract(bg_lines, 1)

        # Find the indices that rearrange sens_coord to sens_names
        newIDX = find_map(sens_names, sens_coord["sName"].to_numpy())
        # reorder if necessary
        sens_coord = sens_coord.reindex(labels=newIDX)
        sens_dir = sens_dir[newIDX, :]
        # # Transform into numpy array
        # sens_coord= sens_coord[["x","y","z"]].to_numpy()

        self.Geo1 = Geometry1(
            sens_names=sens_names,
            sens_coord=sens_coord,
            sens_dir=sens_dir,
            sens_lines=sens_lines,
            bg_nodes=bg_nodes,
            bg_lines=bg_lines,
            bg_surf=bg_surf,
        )

    # metodo per definire geometria 2
    def def_geo2(
        self,
        # # MANDATORY
        sens_names: typing.Union[
            npt.NDArray[np.string], typing.List[str]
        ],  # sensors' names
        pts_coord: pd.DataFrame,  # points' coordinates
        sens_map: pd.DataFrame,  # mapping
        sens_sign: pd.DataFrame,
        # # OPTIONAL
        order_red: typing.Optiona[typing.Literal["xy", "xz", "yz", "x", "y", "z"]] = None,
        sens_lines: npt.NDArray[np.int64] = None,  # lines connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None,  # Background nodes
        bg_lines: npt.NDArray[np.float64] = None,  # Background lines
        bg_surf: npt.NDArray[np.float64] = None,  # Background lines
    ):
        # ---------------------------------------------------------------------
        # Checks on input
        if order_red == "xy" or order_red == "xz" or order_red == "yz":
            nc = 2
            assert sens_map.to_numpy()[:, 1:].shape[1] == nc
            assert sens_sign.to_numpy()[:, 1:].shape[1] == nc
        elif order_red == "x" or order_red == "y" or order_red == "z":
            nc = 1
            assert sens_map.to_numpy()[:, 1:].shape[1] == nc
            assert sens_sign.to_numpy()[:, 1:].shape[1] == nc
        elif order_red is None:
            nc = 3
            assert sens_map.to_numpy()[:, 1:].shape[1] == nc
            assert sens_sign.to_numpy()[:, 1:].shape[1] == nc
        # altri controlli??
        # ---------------------------------------------------------------------
        # adapt to 0 indexed lines
        if bg_lines is not None:
            bg_lines = np.subtract(bg_lines, 1)
        if sens_lines is not None:
            sens_lines = np.subtract(sens_lines, 1)

        self.Geo2 = Geometry2(
            sens_names=sens_names,
            pts_coord=pts_coord,
            sens_map=sens_map,
            sens_sign=sens_sign,
            order_red=order_red,
            sens_lines=sens_lines,
            bg_nodes=bg_nodes,
            bg_lines=bg_lines,
            bg_surf=bg_surf,
        )

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
        self, name: str, default: typing.Optional[BaseAlgorithm] = None
    ) -> typing.Optional[BaseAlgorithm]:
        """
        Retrieve an algorithm from the set by its name.
        Returns the default value if the algorithm does not exist.
        """
        return self.algorithms.get(name, default)


# =============================================================================
# MULTISETUP
# =============================================================================
class MultiSetup_PoSER:
    """
    Multi setup merging with "Post Separate Estimation Re-scaling" approach
    """

    __result: typing.Optional[typing.Dict[str, MsPoserResult]] = None
    __alg_ref: typing.Optional[typing.Dict[type[BaseAlgorithm], str]] = None

    def __init__(
        self,
        ref_ind: typing.List[typing.List[int]],
        single_setups: typing.List[SingleSetup],  # | None = None,
    ):
        self._setups = (
            [el for el in self._init_setups(single_setups)] if single_setups else []
        )
        self.ref_ind = ref_ind
        self.__result = None

    @property
    def setups(self):
        return self._setups

    @setups.setter
    def setups(self, setups):
        # not allow to set setups after initialization
        if hasattr(self, "_setups"):
            raise AttributeError("Cannot set setups after initialization")
        self._setups = setups

    @property
    def result(self) -> typing.Dict[str, MsPoserResult]:
        if self.__result is None:
            raise ValueError("You must run merge_results() first")
        return self.__result

    def _init_setups(
        self, setups: typing.List[SingleSetup]
    ) -> typing.Generator[SingleSetup, None, None]:
        """Ensure that each setup has run its algorithms and that internally consistent algorithms.

        Parameters
        ----------
        setups : list[SingleSetup]

        Raises
        ------
        ValueError
            if no setups are passed
        ValueError
            if any setup has no algorithms
        ValueError
            if any setup has multiple algorithms of same type
        ValueError
            if any setup has algorithms that have not been run
        """
        if len(setups) == 0:
            raise ValueError("You must pass at least one setup")
        if any(not setup.algorithms for setup in setups):
            raise ValueError("You must pass setups with at least one algorithm")

        self.__alg_ref: typing.Dict[type[BaseAlgorithm], str] = {
            alg.__class__: alg.name for alg in setups[0].algorithms.values()
        }

        for i, setup in enumerate(setups):
            tot_alg = len(setup.algorithms)
            tot__alg_ref = len(self.__alg_ref)
            if tot_alg > tot__alg_ref:
                # check for duplicates algorithms in a setup
                duplicates = [
                    (alg.__class__, alg.name)
                    for alg in setup.algorithms.values()
                    if alg.name not in self.__alg_ref.values()
                ]
                raise ValueError(
                    f"You must pass distinct algorithms for setup {i+1}. Duplicates: {duplicates}"
                )
            if tot_alg < tot__alg_ref:
                # check for missing algorithms in a setup
                setup_algs = [alg.__class__ for alg in setup.algorithms.values()]
                missing = [
                    alg_cl for alg_cl in self.__alg_ref.keys() if alg_cl not in setup_algs
                ]
                raise ValueError(
                    f"You must pass all algorithms for setup {i+1}. Missing: {missing}"
                )

            logger.debug("Initializing %s/%s setups", i + 1, len(setups))
            for alg in setup.algorithms.values():
                if not alg.result or alg.result.Fn is None:
                    raise ValueError(
                        "You must pass Single setups that have already been run"
                    )
            yield setup

    def merge_results(self) -> typing.Dict[str, MsPoserResult]:
        # group algorithms by type
        alg_groups: typing.Dict[type[BaseAlgorithm], BaseAlgorithm] = {}
        for setup in self.setups:
            for alg in setup.algorithms.values():
                alg_groups.setdefault(alg.__class__, []).append(alg)

        for alg_cl, algs in alg_groups.items():
            logger.info("Merging %s results", alg_cl.__name__)
            # get the reference algorithm
            all_fn = []
            all_xi = []
            results = []
            for alg in algs:
                logger.info("Merging %s results", alg.name)
                all_fn.append(alg.result.Fn)
                all_xi.append(alg.result.Xi)
                results.append(alg.result.Phi)

            # Convert lists to numpy arrays
            all_fn = np.array(all_fn)
            all_xi = np.array(all_xi)

            # Calculate mean and covariance
            fn_mean = np.mean(all_fn, axis=0)
            xi_mean = np.mean(all_xi, axis=0)

            fn_cov = np.std(all_fn, axis=0) / fn_mean
            xi_cov = np.std(all_xi, axis=0) / xi_mean
            Phi = merge_mode_shapes(MSarr_list=results, reflist=self.ref_ind)

            if self.__result is None:
                self.__result = {}

            self.__result[alg_cl.__name__] = MsPoserResult(
                Phi=Phi,
                Fn=fn_mean,
                Fn_cov=fn_cov,
                Xi=xi_mean,
                Xi_cov=xi_cov,
            )
        return self.__result

    # metodo per definire geometria 1
    def def_geo1(
        self,
        # # MANDATORY
        sens_names: typing.List[typing.List[str]],  # sensors' names MS
        sens_coord: pd.DataFrame,  # sensors' coordinates
        sens_dir: npt.NDArray[np.int64],  # sensors' directions
        # # OPTIONAL
        sens_lines: npt.NDArray[np.int64] = None,  # lines connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None,  # Background nodes
        bg_lines: npt.NDArray[np.int64] = None,  # Background lines
        bg_surf: npt.NDArray[np.float64] = None,  # Background surfaces
    ):
        # ---------------------------------------------------------------------
        sens_names_c = copy.deepcopy(sens_names)
        ref_ind = self.ref_ind
        ini = [sens_names_c[0][ref_ind[0][ii]] for ii in range(len(ref_ind[0]))]

        # Iterate and remove indices
        for string_list, index_list in zip(sens_names_c, ref_ind):
            for index in sorted(index_list, reverse=True):
                if 0 <= index < len(string_list):
                    string_list.pop(index)

        # flatten (reduced) sens_name list
        fr_sens_names = [x for xs in sens_names_c for x in xs]
        sens_names_final = ini + fr_sens_names

        # Checks on input
        # Altri controlli ???
        # ---------------------------------------------------------------------
        # adapt to 0 indexing
        if bg_lines is not None:
            bg_lines = np.subtract(bg_lines, 1)

        # Find the indices that rearrange sens_coord to sens_names
        newIDX = find_map(sens_names_final, sens_coord["sName"].to_numpy())
        # reorder if necessary
        sens_coord = sens_coord.reindex(labels=newIDX)
        sens_dir = sens_dir[newIDX, :]
        # # Transform into numpy array
        # sens_coord= sens_coord[["x","y","z"]].to_numpy()

        self.Geo1 = Geometry1(
            sens_names=sens_names_final,
            sens_coord=sens_coord,
            sens_dir=sens_dir,
            sens_lines=sens_lines,
            bg_nodes=bg_nodes,
            bg_lines=bg_lines,
            bg_surf=bg_surf,
        )

    # metodo per plottare geometria 1 - OK
    def plot_geo1(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
    ):

        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        # plot sensors' nodes
        sens_coord = self.Geo1.sens_coord[["x", "y", "z"]].to_numpy()
        plt_nodes(ax, sens_coord, color="red")

        # plot sensors' directions
        plt_quiver(
            ax, sens_coord, self.Geo1.sens_dir, scaleF=scaleF, names=self.Geo1.sens_names
        )

        # Check that BG nodes are defined
        if self.Geo1.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, self.Geo1.bg_nodes, color="gray", alpha=0.5)
            # Check that BG lines are defined
            if self.Geo1.bg_lines is not None:
                # if True plot
                plt_lines(
                    ax, self.Geo1.bg_nodes, self.Geo1.bg_lines, color="gray", alpha=0.5
                )
            if self.Geo1.bg_surf is not None:
                # if True plot
                plt_surf(ax, self.Geo1.bg_nodes, self.Geo1.bg_surf, alpha=0.1)

        # check for sens_lines
        if self.Geo1.sens_lines is not None:
            # if True plot
            plt_lines(ax, sens_coord, self.Geo1.sens_lines, color="red")

        # Set ax options
        set_ax_options(
            ax,
            bg_color="w",
            remove_fill=remove_fill,
            remove_grid=remove_grid,
            remove_axis=remove_axis,
        )

        # Set view
        set_view(ax, view=view)

        return fig, ax

    # metodo per definire geometria 2
    def def_geo2(
        self,
        # # MANDATORY
        sens_names: typing.List[typing.List[str]],  # sensors' names MS
        pts_coord: pd.DataFrame,  # points' coordinates
        sens_map: pd.DataFrame,  # mapping
        sens_sign: pd.DataFrame,
        # # OPTIONAL
        order_red: typing.Optional[
            typing.Literal["xy", "xz", "yz", "x", "y", "z"]
        ] = None,
        sens_lines: npt.NDArray[np.int64] = None,  # lines connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None,  # Background nodes
        bg_lines: npt.NDArray[np.float64] = None,  # Background lines
        bg_surf: npt.NDArray[np.float64] = None,  # Background lines
    ):
        # ---------------------------------------------------------------------
        sens_names_c = copy.deepcopy(sens_names)
        ref_ind = self.ref_ind
        ini = [sens_names_c[0][ref_ind[0][ii]] for ii in range(len(ref_ind[0]))]

        # Iterate and remove indices
        for string_list, index_list in zip(sens_names_c, ref_ind):
            for index in sorted(index_list, reverse=True):
                if 0 <= index < len(string_list):
                    string_list.pop(index)

        # flatten (reduced) sens_name list
        fr_sens_names = [x for xs in sens_names_c for x in xs]
        sens_names_final = ini + fr_sens_names

        # Check that length of sens_names_final == len(result.Phi[:,i])

        # Checks on input
        if order_red == "xy" or order_red == "xz" or order_red == "yz":
            nc = 2
            assert sens_map.to_numpy()[:, 1:].shape[1] == nc
            assert sens_sign.to_numpy()[:, 1:].shape[1] == nc
        elif order_red == "x" or order_red == "y" or order_red == "z":
            nc = 1
            assert sens_map.to_numpy()[:, 1:].shape[1] == nc
            assert sens_sign.to_numpy()[:, 1:].shape[1] == nc
        elif order_red is None:
            nc = 3
            assert sens_map.to_numpy()[:, 1:].shape[1] == nc
            assert sens_sign.to_numpy()[:, 1:].shape[1] == nc
        # ---------------------------------------------------------------------
        # adapt to 0 indexed lines
        if bg_lines is not None:
            bg_lines = np.subtract(bg_lines, 1)
        if sens_lines is not None:
            sens_lines = np.subtract(sens_lines, 1)

        self.Geo2 = Geometry2(
            sens_names=sens_names_final,
            pts_coord=pts_coord,
            sens_map=sens_map,
            sens_sign=sens_sign,
            order_red=order_red,
            sens_lines=sens_lines,
            bg_nodes=bg_nodes,
            bg_lines=bg_lines,
            bg_surf=bg_surf,
        )

    # metodo per plottare geometria 2 - OK
    def plot_geo2(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
    ):

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # plot sensors' nodes
        plt_nodes(ax, self.Geo2.pts_coord, color="red")

        # plot sensors' directions
        plt_quiver(
            ax,
            self.Geo2.pts_coord,
            self.Geo2.sens_map,
            scaleF=scaleF,
            names=self.Geo2.sens_names,
        )

        # Check that BG nodes are defined
        if self.Geo2.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, self.Geo2.bg_nodes, color="gray", alpha=0.5)
            # Check that BG lines are defined
            if self.Geo2.bg_lines is not None:
                # if True plot
                plt_lines(
                    ax, self.Geo2.bg_nodes, self.Geo2.bg_lines, color="gray", alpha=0.5
                )
            if self.Geo2.bg_surf is not None:
                # if True plot
                plt_surf(ax, self.Geo2.bg_nodes, self.Geo2.bg_surf, alpha=0.1)

        # check for sens_lines
        if self.Geo2.sens_lines is not None:
            # if True plot
            plt_lines(ax, self.Geo2.pts_coord, self.Geo2.sens_lines, color="red")

        # Set ax options
        set_ax_options(
            ax,
            bg_color="w",
            remove_fill=remove_fill,
            remove_grid=remove_grid,
            remove_axis=remove_axis,
        )

        # Set view
        set_view(ax, view=view)

        return fig, ax

    def plot_mode_g1(
        self,
        Algo_Res: MsPoserResult,
        Geo1: Geometry1,
        mode_numb: int,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
    ) -> typing.Any:
        """Tobe implemented, plot for FDD, EFDD, FSDD
        Mode Identification Function (MIF)
        """

        if Algo_Res.Fn is None:
            raise ValueError("Run algorithm first")

        # Select the (real) mode shape
        phi = Algo_Res.Phi[:, int(mode_numb - 1)].real
        fn = Algo_Res.Fn[int(mode_numb - 1)]

        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        # set title
        ax.set_title(f"Mode nr. {mode_numb}, $f_n$={fn:.3f}Hz")

        # plot sensors' nodes
        sens_coord = Geo1.sens_coord[["x", "y", "z"]].to_numpy()
        plt_nodes(ax, sens_coord, color="red")

        # plot Mode shape
        plt_quiver(
            ax,
            sens_coord,
            Geo1.sens_dir * phi.reshape(-1, 1),
            scaleF=scaleF,
            names=Geo1.sens_names,
        )

        # Check that BG nodes are defined
        if Geo1.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, Geo1.bg_nodes, color="gray", alpha=0.5)
            # Check that BG lines are defined
            if Geo1.bg_lines is not None:
                # if True plot
                plt_lines(ax, Geo1.bg_nodes, Geo1.bg_lines, color="gray", alpha=0.5)
            if Geo1.bg_surf is not None:
                # if True plot
                plt_surf(ax, Geo1.bg_nodes, Geo1.bg_surf, alpha=0.1)

        # check for sens_lines
        if Geo1.sens_lines is not None:
            # if True plot
            plt_lines(ax, sens_coord, Geo1.sens_lines, color="red")

        # Set ax options
        set_ax_options(
            ax,
            bg_color="w",
            remove_fill=remove_fill,
            remove_grid=remove_grid,
            remove_axis=remove_axis,
        )

        # Set view
        set_view(ax, view=view)
        return fig, ax

    def plot_mode_g2(
        self,
        Algo_Res: MsPoserResult,
        Geo2: Geometry2,
        mode_numb: typing.Optional[int],
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
        *args,
        **kwargs,
    ) -> typing.Any:
        """Tobe implemented, plot for FDD, EFDD, FSDD
        Mode Identification Function (MIF)
        """
        if Algo_Res.Fn is None:
            raise ValueError("Run algorithm first")

        # Select the (real) mode shape
        fn = Algo_Res.Fn[int(mode_numb - 1)]
        phi = Algo_Res.Phi[:, int(mode_numb - 1)].real * scaleF
        # create mode shape dataframe
        df_phi = pd.DataFrame(
            {"sName": Geo2.sens_names, "Phi": phi},
        )
        mapping = dict(zip(df_phi["sName"], df_phi["Phi"]))
        # reshape the mode shape dataframe to fit the pts coord
        df_phi_map = Geo2.sens_map.replace(mapping).astype(float)
        # add together coordinates and mode shape displacement
        newpoints = Geo2.pts_coord.add(df_phi_map * Geo2.sens_sign, fill_value=0)
        # extract only the displacement array
        newpoints = newpoints.to_numpy()[:, 1:]

        # create fig and ax
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        # set title
        ax.set_title(f"Mode nr. {mode_numb}, $f_n$={fn:.3f}Hz")

        # Check that BG nodes are defined
        if Geo2.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, Geo2.bg_nodes, color="gray", alpha=0.5)
            # Check that BG lines are defined
            if Geo2.bg_lines is not None:
                # if True plot
                plt_lines(ax, Geo2.bg_nodes, Geo2.bg_lines, color="gray", alpha=0.5)
            if Geo2.bg_surf is not None:
                # if True plot
                plt_surf(ax, Geo2.bg_nodes, Geo2.bg_surf, alpha=0.1)
        # PLOT MODE SHAPE
        plt_nodes(ax, newpoints, color="red")
        # check for sens_lines
        if Geo2.sens_lines is not None:
            # if True plot
            plt_lines(ax, newpoints, Geo2.sens_lines, color="red")

        # Set ax options
        set_ax_options(
            ax,
            bg_color="w",
            remove_fill=remove_fill,
            remove_grid=remove_grid,
            remove_axis=remove_axis,
        )

        # Set view
        set_view(ax, view=view)

        return fig, ax

    def anim_mode_g2(
        self,
        Algo_Res: MsPoserResult,
        Geo2: Geometry2,
        mode_numb: typing.Optional[int],
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
        *args,
        **kwargs,
    ) -> typing.Any:
        """Tobe implemented, plot for FDD, EFDD, FSDD
        Mode Identification Function (MIF)
        """
        if Algo_Res.Fn is None:
            raise ValueError("Run algorithm first")

        logger.info("Running Anim Mode FDD")
        AniMode(
            Geo=Geo2,
            Res=Algo_Res,
            mode_numb=mode_numb,
            scaleF=scaleF,
            view=view,
            remove_axis=remove_axis,
            remove_fill=remove_fill,
            remove_grid=remove_grid,
        )
        logger.info("...end AniMode FDD...")


# -----------------------------------------------------------------------------


class MultiSetup_PreGER(BaseSetup):
    """
    Multi setup merging with "Pre Global Estimation Re-scaling" approach
    """

    def __init__(
        self,
        fs: float,  # ! list[float]
        ref_ind: typing.List[typing.List[int]],
        datasets: typing.List[npt.NDArray[np.float64]],
    ):
        self.fs = fs
        self.ref_ind = ref_ind
        Y = PRE_MultiSetup(datasets, ref_ind)
        self.data = Y
        self.algorithms: typing.Dict[str, BaseAlgorithm] = {}  # set of algo

    # metodo per definire geometria 1
    def def_geo1(
        self,
        # # MANDATORY
        sens_names: typing.List[typing.List[str]],  # sensors' names MS
        sens_coord: pd.DataFrame,  # sensors' coordinates
        sens_dir: npt.NDArray[np.int64],  # sensors' directions
        # # OPTIONAL
        sens_lines: npt.NDArray[np.int64] = None,  # lines connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None,  # Background nodes
        bg_lines: npt.NDArray[np.int64] = None,  # Background lines
        bg_surf: npt.NDArray[np.float64] = None,  # Background surfaces
    ):
        # ---------------------------------------------------------------------
        sens_names_c = copy.deepcopy(sens_names)
        ref_ind = self.ref_ind
        ini = [sens_names_c[0][ref_ind[0][ii]] for ii in range(len(ref_ind[0]))]

        # Iterate and remove indices
        for string_list, index_list in zip(sens_names_c, ref_ind):
            for index in sorted(index_list, reverse=True):
                if 0 <= index < len(string_list):
                    string_list.pop(index)

        # flatten (reduced) sens_name list
        fr_sens_names = [x for xs in sens_names_c for x in xs]
        sens_names_final = ini + fr_sens_names
        # ---------------------------------------------------------------------
        # adapt to 0 indexing
        if bg_lines is not None:
            bg_lines = np.subtract(bg_lines, 1)

        # Find the indices that rearrange sens_coord to sens_names
        newIDX = find_map(sens_names_final, sens_coord["sName"].to_numpy())
        # reorder if necessary
        sens_coord = sens_coord.reindex(labels=newIDX)
        sens_dir = sens_dir[newIDX, :]
        # # Transform into numpy array
        # sens_coord= sens_coord[["x","y","z"]].to_numpy()

        self.Geo1 = Geometry1(
            sens_names=sens_names_final,
            sens_coord=sens_coord,
            sens_dir=sens_dir,
            sens_lines=sens_lines,
            bg_nodes=bg_nodes,
            bg_lines=bg_lines,
            bg_surf=bg_surf,
        )

    # metodo per definire geometria 2
    def def_geo2(
        self,
        # # MANDATORY
        sens_names: typing.Union[
            npt.NDArray[np.string], typing.List[str]
        ],  # sensors' names
        pts_coord: pd.DataFrame,  # points' coordinates
        sens_map: pd.DataFrame,  # mapping
        sens_sign: pd.DataFrame,
        # # OPTIONAL
        order_red: typing.Union[typing.Literal["xy", "xz", "yz", "x", "y", "z"]] = None,
        sens_lines: npt.NDArray[np.int64] = None,  # lines connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None,  # Background nodes
        bg_lines: npt.NDArray[np.float64] = None,  # Background lines
        bg_surf: npt.NDArray[np.float64] = None,  # Background lines
    ):
        # ---------------------------------------------------------------------
        sens_names_c = copy.deepcopy(sens_names)
        ref_ind = self.ref_ind
        ini = [sens_names_c[0][ref_ind[0][ii]] for ii in range(len(ref_ind[0]))]

        # Iterate and remove indices
        for string_list, index_list in zip(sens_names_c, ref_ind):
            for index in sorted(index_list, reverse=True):
                if 0 <= index < len(string_list):
                    string_list.pop(index)

        # flatten (reduced) sens_name list
        fr_sens_names = [x for xs in sens_names_c for x in xs]
        sens_names_final = ini + fr_sens_names
        # ---------------------------------------------------------------------
        # adapt to 0 indexed lines
        if bg_lines is not None:
            bg_lines = np.subtract(bg_lines, 1)
        if sens_lines is not None:
            sens_lines = np.subtract(sens_lines, 1)

        self.Geo2 = Geometry2(
            sens_names=sens_names_final,
            pts_coord=pts_coord,
            sens_map=sens_map,
            sens_sign=sens_sign,
            order_red=order_red,
            sens_lines=sens_lines,
            bg_nodes=bg_nodes,
            bg_lines=bg_lines,
            bg_surf=bg_surf,
        )
