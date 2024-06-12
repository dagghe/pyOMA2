# -*- coding: utf-8 -*-
"""
Operational Modal Analysis Module for Single and Multi-Setup Configurations.
Part of the pyOMA2 package.
Authors:
Dag Pasca
Diego Margoni
"""

from __future__ import annotations

import copy
import logging
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyoma2.algorithms.data.result import MsPoserResult
from pyoma2.functions.gen import (
    find_map,
    flatten_sns_names,
    import_excel_GEO1,
    import_excel_GEO2,
    merge_mode_shapes,
    pre_MultiSetup,
)
from pyoma2.functions.plot import (
    STFT,
    plt_ch_info,
    plt_data,
)
from pyoma2.setup.base import BaseSetup
from pyoma2.setup.single import SingleSetup
from pyoma2.support.geometry import Geometry1, Geometry2
from pyoma2.support.mpl_plotter import MplGeoPlotter
from pyoma2.support.pyvista_plotter import PvGeoPlotter

if typing.TYPE_CHECKING:
    from pyoma2.algorithms import BaseAlgorithm


logger = logging.getLogger(__name__)

# =============================================================================
# POSER
# =============================================================================


class MultiSetup_PoSER:
    """
    Class for conducting Operational Modal Analysis (OMA) on multi-setup experiments using
    the Post Separate Estimation Re-scaling (PoSER) approach. This approach is designed to
    merge and analyze data from multiple experimental setups for comprehensive modal analysis.

    The PoSER method is particularly useful in situations where data from different setups
    need to be combined to enhance the understanding of the system's modal properties.

    Attributes
    ----------
    __result : Optional[Dict[str, MsPoserResult]]
        Private attribute to store the merged results from multiple setups. Each entry in the
        dictionary corresponds to a specific algorithm used across setups, with its results.
    __alg_ref : Optional[Dict[type[BaseAlgorithm], str]]
        Private attribute to store references to the algorithms used in the setups. It maps
        each algorithm type to its corresponding name.

    Methods
    -------
    merge_results()
        Merges the results from individual setups into a combined result for holistic analysis.
    plot_mode_g1(mode_number: int, scale_factor: int, view_type: str)
        Plots mode shapes for a specified mode number using the first type of geometric setup (geo1).
    plot_mode_g2(mode_number: int, scale_factor: int, view_type: str)
        Plots mode shapes for a specified mode number using the second type of geometric setup (geo2).
    anim_mode_g2(mode_number: int, scale_factor: int, view_type: str, save_as_gif: bool)
        Creates an animation of the mode shapes for a specified mode number using the second type
        of geometric setup (geo2). Option to save the animation as a GIF file.
    def_geo1(...)
        Defines the first type of geometric setup (geo1) for the instance, based on sensor placements
        and structural characteristics.
    def_geo2(...)
        Defines the second type of geometric setup (geo2) for the instance, typically involving more
        complex geometries or additional data.

    plot_geo1(...)
        Plots the geometric configuration of the structure based on the geo1 setup, including sensor
        placements and structural details.
    plot_geo2(...)
        Plots the geometric configuration of the structure based on the geo2 setup, highlighting
        more intricate details or alternative layouts.

    Warning
    -------
    The PoSER approach assumes that the setups used are compatible in terms of their experimental
    setup and data characteristics.
    """

    __result: typing.Optional[typing.Dict[str, MsPoserResult]] = None
    __alg_ref: typing.Optional[typing.Dict[type[BaseAlgorithm], str]] = None
    geo1: typing.Optional[Geometry1] = None
    geo2: typing.Optional[Geometry2] = None

    def __init__(
        self,
        ref_ind: typing.List[typing.List[int]],
        single_setups: typing.List[SingleSetup],
    ):
        """
        Initializes the MultiSetup_PoSER instance with reference indices and a list of SingleSetup instances.

        Parameters
        ----------
        ref_ind : List[List[int]]
            Reference indices for merging results from different setups.
        single_setups : List[SingleSetup]
            A list of SingleSetup instances to be merged using the PoSER approach.

        Raises
        ------
        ValueError
            If any of the provided setups are invalid or incompatible.
        """
        self._setups = [
            el for el in self._init_setups(setups=single_setups if single_setups else [])
        ]
        self.ref_ind = ref_ind
        self.__result = None

    @property
    def setups(self):
        """
        Returns the list of SingleSetup instances representing individual measurement setups.
        """
        return self._setups

    @setups.setter
    def setups(self, setups):
        """
        Sets the list of SingleSetup instances. Not allowed after initialization.

        Raises
        ------
        AttributeError
            If trying to set setups after initialization.
        """
        # not allow to set setups after initialization
        if hasattr(self, "_setups"):
            raise AttributeError("Cannot set setups after initialization")
        self._setups = setups

    @property
    def result(self) -> typing.Dict[str, MsPoserResult]:
        """
        Returns the merged results after applying the PoSER method.

        Raises
        ------
        ValueError
            If the merge_results() method has not been run yet.
        """
        if self.__result is None:
            raise ValueError("You must run merge_results() first")
        return self.__result

    def _init_setups(
        self, setups: typing.List[SingleSetup]
    ) -> typing.Generator[SingleSetup, None, None]:
        """
        Ensures each setup has run its algorithms and that algorithms are internally consistent.

        Parameters
        ----------
        setups : List[SingleSetup]
            List of SingleSetup instances to initialize.

        Yields
        ------
        Generator[SingleSetup, None, None]
            Generator yielding initialized SingleSetup instances.

        Raises
        ------
        ValueError
            If there are issues with the provided setups or algorithms.
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
                    alg_cl for alg_cl in self.__alg_ref if alg_cl not in setup_algs
                ]
                raise ValueError(
                    f"You must pass all algorithms for setup {i+1}. Missing: {missing}"
                )

            logger.debug("Initializing %s/%s setups", i + 1, len(setups))
            for alg in setup.algorithms.values():
                if not alg.result or alg.result.Fn is None:
                    raise ValueError(
                        "You must pass Single setups that have already been run"
                        " and the Modal Parameters have to be extracted (call MPE method on SingleSetup)"
                    )
            yield setup

    def merge_results(self) -> typing.Dict[str, MsPoserResult]:
        """
        Merges results from individual setups into a combined result using the PoSER method.

        Groups algorithms by type across all setups and merges their results.
        Calculates the mean and covariance of natural frequencies and damping ratios,
        and merges mode shapes.

        Returns
        -------
        Dict[str, MsPoserResult]
            A dictionary containing the merged results for each algorithm type.

        Raises
        ------
        ValueError
            If the method is called before running algorithms on the setups.
        """
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
    ):
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
        # TODO
        # assert dimensions

        # ---------------------------------------------------------------------
        ref_ind = getattr(self, "ref_ind", None)
        sens_names = flatten_sns_names(sens_names, ref_ind=ref_ind)
        # ---------------------------------------------------------------------
        # Find the indices that rearrange sens_coord to sens_names
        newIDX = find_map(sens_names, sens_coord.index.to_numpy())
        # reorder if necessary
        sens_coord = sens_coord.reindex(labels=newIDX)
        sens_dir = sens_dir[newIDX, :]

        self.geo1 = Geometry1(
            sens_names=sens_names,
            sens_coord=sens_coord,
            sens_dir=sens_dir,
            sens_lines=sens_lines,
            bg_nodes=bg_nodes,
            bg_lines=bg_lines,
            bg_surf=bg_surf,
        )

    # metodo per definire geometria 1 da file
    def def_geo1_byFILE(self, path: str):
        """ """
        ref_ind = getattr(self, "ref_ind", None)

        data = import_excel_GEO1(path, ref_ind=ref_ind)

        self.geo1 = Geometry1(
            sens_names=data[0],
            sens_coord=data[1],
            sens_dir=data[2].values,
            sens_lines=data[3],
            bg_nodes=data[4],
            bg_lines=data[5],
            bg_surf=data[6],
        )

    # metodo per definire geometria 2
    def def_geo2(
        self,
        # # MANDATORY
        sens_names: typing.Union[
            typing.List[typing.List[str]], pd.DataFrame
        ],  # sensors' names MS
        pts_coord: pd.DataFrame,  # points' coordinates
        sens_map: pd.DataFrame,  # mapping
        # # OPTIONAL
        cstrn: pd.DataFrame = None,
        sens_sign: pd.DataFrame = None,
        sens_lines: npt.NDArray[np.int64] = None,  # lines connecting sensors
        sens_surf: npt.NDArray[np.int64] = None,  # surf connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None,  # Background nodes
        bg_lines: npt.NDArray[np.float64] = None,  # Background lines
        bg_surf: npt.NDArray[np.float64] = None,  # Background lines
    ):
        """
        Defines the second geometry setup (geo2) for the instance, incorporating sensors' names,
        points' coordinates, mapping, sign data, and optional elements like order reduction,
        sensor lines, background nodes, lines, and surfaces.

        Parameters
        ----------
        sens_names : Union[np.ndarray, List[str]]
            An array or list containing the names of the sensors.
        pts_coord : pd.DataFrame
            A DataFrame containing the coordinates of the points. Columns should include 'x', 'y', and 'z'.
        sens_map : pd.DataFrame
            A DataFrame containing the mapping data for sensors.
        sens_sign : pd.DataFrame
            A DataFrame containing sign data for the sensors.
        sens_lines : Optional[np.ndarray], optional
            An array defining lines connecting sensors, by default None.
        bg_nodes : Optional[np.ndarray], optional
            An array defining background nodes, by default None.
        bg_lines : Optional[np.ndarray], optional
            An array defining background lines, by default None.
        bg_surf : Optional[np.ndarray], optional
            An array defining background surfaces, by default None.

        Raises
        ------
        AssertionError
            If the number of columns in mapping and sign data does not match the expected
            dimensions based on the order reduction.

        Notes
        -----
        Adapts to zero-indexing for sensor and background lines if provided.
        """
        # ---------------------------------------------------------------------

        ref_ind = getattr(self, "ref_ind", None)
        sens_names = flatten_sns_names(sens_names, ref_ind=ref_ind)

        # ---------------------------------------------------------------------
        # adapt to 0 indexed lines
        if bg_lines is not None:
            bg_lines = np.subtract(bg_lines, 1)
        if sens_lines is not None:
            sens_lines = np.subtract(sens_lines, 1)
        if bg_surf is not None:
            bg_surf = np.subtract(bg_surf, 1)
        if sens_surf is not None:
            sens_surf = np.subtract(sens_surf, 1)

        self.geo2 = Geometry2(
            sens_names=sens_names,
            pts_coord=pts_coord,
            sens_map=sens_map,
            cstrn=cstrn,
            sens_sign=sens_sign,
            sens_lines=sens_lines,
            sens_surf=sens_surf,
            bg_nodes=bg_nodes,
            bg_lines=bg_lines,
            bg_surf=bg_surf,
        )

    # metodo per definire geometria 2 da file
    def def_geo2_byFILE(self, path: str):
        """ """
        ref_ind = getattr(self, "ref_ind", None)

        data = import_excel_GEO2(path, ref_ind=ref_ind)

        self.geo2 = Geometry2(
            sens_names=data[0],
            pts_coord=data[1],
            sens_map=data[2],
            cstrn=data[3],
            sens_sign=data[4],
            sens_lines=data[5],
            sens_surf=data[6],
            bg_nodes=data[7],
            bg_lines=data[8],
            bg_surf=data[9],
        )

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
    ):
        if self.geo1 is None:
            raise ValueError("geo1 is not defined. Call def_geo1 first.")

        Plotter = MplGeoPlotter(self.geo1)

        fig, ax = Plotter.plot_geo1(
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
    ):
        if self.geo2 is None:
            raise ValueError("geo2 is not defined. Call def_geo2 first.")

        Plotter = MplGeoPlotter(self.geo2)

        fig, ax = Plotter.plot_geo2(
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

    # PLOT GEO2 - PyVista plotter
    def plot_geo2(
        self,
        scaleF: int = 1,
        plot_points: bool = True,
        points_sett: dict = "default",
        plot_lines: bool = True,
        lines_sett: dict = "default",
        plot_surf: bool = True,
        surf_sett: dict = "default",
    ):
        if self.geo2 is None:
            raise ValueError("geo2 is not defined. Call def_geo2 first.")

        Plotter = PvGeoPlotter(self.geo2)

        pl = Plotter.plot_geo(
            plot_points,
            points_sett,
        )
        return pl

    # PLOT MODI - Matplotlib plotter
    def plot_mode_g1(
        self,
        Algo_Res: MsPoserResult,
        mode_nr: int,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz"] = "3D",
        col_sns: str = "red",
        col_sns_lines: str = "red",
        col_BG_nodes: str = "gray",
        col_BG_lines: str = "gray",
        col_BG_surf: str = "gray",
    ) -> typing.Any:
        """ """
        if self.geo1 is None:
            raise ValueError("geo1 is not defined. Call def_geo1 first.")

        if Algo_Res.Fn is None:
            raise ValueError("Run algorithm first")
        Plotter = MplGeoPlotter(self.geo1, Algo_Res)

        fig, ax = Plotter.plot_mode_g1(
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

    # PLOT MODI - Matplotlib plotter
    def plot_mode_g2_mpl(
        self,
        Algo_Res: MsPoserResult,
        mode_nr: typing.Optional[int],
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz"] = "3D",
        color: str = "cmap",
        *args,
        **kwargs,
    ) -> typing.Any:
        """ """
        if self.geo2 is None:
            raise ValueError("geo2 is not defined. Call def_geo2 first.")

        if Algo_Res.Fn is None:
            raise ValueError("Run algorithm first")

        Plotter = MplGeoPlotter(self.geo2, Algo_Res)

        fig, ax = Plotter.plot_mode_g2(mode_nr, scaleF, view, color)
        return fig, ax

    # PLOT MODI - PyVista plotter
    def plot_mode_g2(
        self,
        Algo_Res: MsPoserResult,
        mode_nr: int = 1,
        scaleF: float = 1.0,
        plot_points: bool = True,
        plot_lines: bool = True,
        plot_surf: bool = True,
        plot_undef: bool = True,
        def_sett: dict = "default",
        undef_sett: dict = "default",
        *args,
        **kwargs,
    ) -> typing.Any:
        """ """
        if self.geo2 is None:
            raise ValueError("geo2 is not defined. Call def_geo2 first.")

        if Algo_Res.Fn is None:
            raise ValueError("Run algorithm first")

        Plotter = PvGeoPlotter(self.geo2, Algo_Res)

        pl = Plotter.plot_mode(
            mode_nr,
            scaleF,
            None,
            plot_points,
            plot_lines,
            plot_surf,
            plot_undef,
            def_sett,
            undef_sett,
        )
        return pl

    # PLOT MODI - PyVista plotter
    def anim_mode_g2(
        self,
        Algo_Res: MsPoserResult,
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
    ) -> typing.Any:
        """ """
        if self.geo2 is None:
            raise ValueError("geo2 is not defined. Call def_geo2 first.")

        if Algo_Res.Fn is None:
            raise ValueError("Run algorithm first")

        Plotter = PvGeoPlotter(self.geo2, Algo_Res)

        pl = Plotter.animate_mode(
            mode_nr, scaleF, None, plot_points, plot_lines, plot_surf, def_sett, saveGIF
        )
        return pl


# =============================================================================
#
# =============================================================================


class MultiSetup_PreGER(BaseSetup):
    """
    Class for conducting Operational Modal Analysis on multi-setup experiments using the
    Pre-Global Estimation Re-scaling (PreGER) approach.
    This class is tailored for handling and processing multiple datasets, applying the PreGER method
    efficiently. It offers functionalities for data visualization, preprocessing, and geometric
    configuration for the structure under analysis.

    Attributes
    ----------
    fs : float
        The common sampling frequency across all datasets.
    dt : float
        The sampling interval, calculated as the inverse of the sampling frequency.
    ref_ind : list[list[int]]
        Indices of reference sensors for each dataset, as a list of lists.
    datasets : list[npt.NDArray[np.float64]]
        The original list of datasets, each represented as a NumPy array.
    data : npt.NDArray[np.float64]
        Processed data after applying the PreGER method, ready for analysis.
    algorithms : dict[str, BaseAlgorithm]
        Dictionary storing algorithms added to the setup, keyed by their names.
    Nchs : list[int]
        A list containing the number of channels for each dataset.
    Ndats : list[int]
        A list containing the number of data points for each dataset.
    Ts : list[float]
        A list containing the total duration (in seconds) of each dataset.
    Nsetup : int
        The number of setups (or datasets) included in the analysis.

    Methods
    -------
    add_algorithms(...)
        Adds algorithms to the setup and initializes them with data and sampling frequency.
    run_all(...)
        Executes all algorithms added to the setup.
    run_by_name(...)
        Runs a specific algorithm identified by its name.
    MPE(...)
        Extracts modal parameters based on selected poles/peaks.
    MPE_fromPlot(...)
        Allows modal parameter extraction directly from interactive plot selections.
    plot_geo1(...)
        Plots the first type of geometric configuration for the structure.
    plot_geo2(...)
        Plots the second type of geometric configuration for the structure.
    plot_data(...)
        Visualizes time history data of channels for selected datasets.
    plot_ch_info(...)
        Displays Time History (TH), Power Spectral Density (PSD), and Kernel Density Estimation (KDE)
        for each channel.
    plt_STFT(...)
        Computes and plots the Short Time Fourier Transform (STFT) magnitude spectrogram for the
        specified channels.
    decimate_data(...)
        Applies data decimation using scipy.signal.decimate.
    detrend_data(...)
        Detrends data using scipy.signal.detrend.
    filter_data(...)
        Applies a Butterworth filter to the input data based on specified parameters.
    def_geo1(...)
        Defines the first type of geometric setup (geo1) for the instance.
    def_geo2(...)
        Defines the second type of geometric setup (geo2) for the instance.

    Warning
    -------
    The PreGER approach assumes that the setups used are compatible in terms of their experimental
    setup and data characteristics.
    """

    def __init__(
        self,
        fs: float,  # ! list[float]
        ref_ind: typing.List[typing.List[int]],
        datasets: typing.List[npt.NDArray[np.float64]],
    ):
        """
        Initializes the MultiSetup_PreGER instance with specified sampling frequency,
        reference indices, and datasets.

        Parameters
        ----------
        fs : float
            The sampling frequency common to all datasets.
        ref_ind : typing.List[typing.List[int]]
            Reference indices for each dataset, used in the PreGER method.
        datasets : typing.List[npt.NDArray[np.float64]]
            A list of datasets, each as a NumPy array.
        """
        self.fs = fs  # sampling frequencies
        self.dt = 1 / fs  # sampling interval
        self.ref_ind = ref_ind  # list of (list of) reference indices
        self.Nsetup = len(ref_ind)
        # Pre-process the data so to be multi-setup compatible
        Y = pre_MultiSetup(datasets, ref_ind)
        self.data = Y
        self.algorithms: typing.Dict[str, BaseAlgorithm] = {}  # set of algo
        self.datasets = datasets
        Nchs = []
        Ndats = []
        Ts = []
        for data in datasets:  # loop through each dataset in the dataset list
            Nch = data.shape[1]  # number of channels
            Ndat = data.shape[0]  # number of data points
            T = self.dt * Ndat  # Period of acquisition [sec]
            Nchs.append(Nch)
            Ndats.append(Ndat)
            Ts.append(T)
        self.Nchs = Nchs
        self.Ndats = Ndats
        self.Ts = Ts

    # method to plot the time histories of the data channels.
    def plot_data(
        self,
        data_idx: typing.Union[str, typing.List[int]] = "all",
        nc: int = 1,
        names: typing.Optional[typing.List[str]] = None,
        # names: list[list[str]] = None,
        unit: str = "unit",
        show_rms: bool = False,
    ):
        """
        Plots the time histories of the data channels for selected datasets.

        Allows for visualization of the time history of each data channel across multiple datasets.
        Users can specify which datasets to plot, configure subplot layout, and optionally display
        RMS acceleration.

        Parameters
        ----------
        data_idx : str | list[int], optional
            Indices of datasets to be plotted. Can be a list of indices or 'all' for all datasets.
            Default is 'all'.
        nc : int, optional
            Number of columns for the subplot layout. Default is 1.
        names : typing.Optional[typing.List[str]], optional
            Names for the channels in each dataset. If provided, these names are used as labels.
            Default is None.
        unit : str, optional
            Unit of measurement for the y-axis. Default is "unit".
        show_rms : bool, optional
            If True, shows the RMS acceleration in the plot. Default is False.

        Returns
        -------
        list
            A list of tuples, each containing the figure and axes objects for the plots of each dataset.
        """
        if data_idx != "all":
            datasets = [self.datasets[i] for i in data_idx]
        else:
            datasets = self.datasets

        fs = self.fs
        figs, axs = [], []
        for ii, data in enumerate(datasets):
            nc = nc  # number of columns for subplot
            nam = (
                names[ii] if names is not None else None
            )  # list of names (str) of the channnels
            unit = unit  # str label for the y-axis (unit of measurement)
            show_rms = show_rms  # wheter to show or not the rms acc in the plot
            fig, ax = plt_data(data, fs, nc, nam, unit, show_rms)
            figs.append(fig)
            axs.append(ax)
        return figs, axs

    # method to plot TH, PSD and KDE for each channel
    def plot_ch_info(
        self,
        data_idx: typing.Union[str, typing.List[int]] = "all",
        nxseg: float = 1024,
        ch_idx: typing.Union[str, typing.List[int]] = "all",
        freqlim: typing.Optional[typing.Tuple[float, float]] = None,
        logscale: bool = True,
        unit: str = "unit",
    ):
        """ """
        if data_idx != "all":
            datasets = [self.datasets[i] for i in data_idx]
        else:
            datasets = self.datasets
        fs = self.fs
        figs, axs = [], []
        for data in datasets:
            fig, ax = plt_ch_info(
                data,
                fs,
                ch_idx=ch_idx,
                freqlim=freqlim,
                logscale=logscale,
                nxseg=nxseg,
                unit=unit,
            )
            figs.append(fig)
            axs.append(ax)
        return figs, axs

    # method to plot TH, PSD and KDE for each channel
    def plot_STFT(
        self,
        data_idx: typing.Union[str, typing.List[int]] = "all",
        nxseg: float = 256,
        pov: float = 0.9,
        ch_idx: typing.Union[str, typing.List[int]] = "all",
        freqlim: typing.Optional[typing.Tuple[float, float]] = None,
        win: str = "hann",
    ):
        """ """
        if data_idx != "all":
            datasets = [self.datasets[i] for i in data_idx]
        else:
            datasets = self.datasets
        fs = self.fs
        figs, axs = [], []
        for data in datasets:
            fig, ax = STFT(
                data,
                fs,
                nxseg=nxseg,
                pov=pov,
                win=win,
                ch_idx=ch_idx,
                freqlim=freqlim,
            )
            figs.append(fig)
            axs.append(ax)
        return figs, axs

    # method to decimate data
    def decimate_data(
        self,
        q: int,
        inplace: bool = False,
        **kwargs,
    ) -> typing.Optional[tuple]:
        """
        Applies decimation to the data using the scipy.signal.decimate function.

        This method reduces the sampling rate of the data by a factor of 'q'.
        The decimation process includes low-pass filtering to reduce aliasing.
        The method updates the instance's data and sampling frequency attributes.

        Parameters
        ----------
        q : int
            The decimation factor. Must be greater than 1.
        inplace : bool, optional
            If True, updates the instance's data attribute with the decimated data.
        **kwargs : dict, optional, will be passed to scipy.signal.decimate
            Additional keyword arguments for the scipy.signal.decimate function:
            n : int, optional
                The order of the filter (if 'ftype' is 'fir') or the number of times
                to apply the filter (if 'ftype' is 'iir'). If None, a default value is used.
            ftype : {'iir', 'fir'}, optional
                The type of filter to use for decimation: 'iir' for an IIR filter
                or 'fir' for an FIR filter. Default is 'iir'.

            zero_phase : bool, optional
                If True, applies a zero-phase filter, which has no phase distortion.
                If False, uses a causal filter with some phase distortion. Default is True.

        Raises
        ------
        ValueError
            If the decimation factor 'q' is not greater than 1.

        Returns
        -------
        tuple
            A tuple containing the new datasets, the number of data points for each dataset,
            and the total duration of each dataset.
            If 'inplace' is True, returns None.

        See Also
        --------
        For further information, see `scipy.signal.decimate
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html>`_.
        """
        n = kwargs.get("n")
        ftype = kwargs.get("ftype", "iir")
        axis = kwargs.get("axis", 0)
        zero_phase = kwargs.get("zero_phase", True)
        datasets = self.datasets
        if not inplace:
            datasets = copy.deepcopy(self.datasets)
        newdatasets = []
        Ndats = []
        Ts = []
        for data in datasets:
            newdata, _, _, Ndat, T = super()._decimate_data(
                data=data,
                fs=self.fs,
                q=q,
                n=n,
                ftype=ftype,
                axis=axis,
                zero_phase=zero_phase,
                **kwargs,
            )
            newdatasets.append(newdata)
            Ndats.append(Ndat)
            Ts.append(T)

        Y = pre_MultiSetup(newdatasets, self.ref_ind)
        fs = self.fs / q
        dt = 1 / self.fs

        if inplace:
            self.data = Y
            self.fs = fs
            self.dt = dt
            self.Ndats = Ndats
            self.Ts = Ts
            return None
        return newdatasets, Y, fs, dt, Ndats, Ts

    # method to detrend data
    def filter_data(
        self,
        Wn: typing.Union[float, typing.Tuple[float, float]],
        order: int = 8,
        btype: str = "lowpass",
        inplace: bool = False,
    ) -> typing.Optional[np.ndarray]:
        """
        Applies a Butterworth filter to the input data based on specified parameters.

        This method filters the data using a Butterworth filter with the specified parameters.
        The method updates the instance's data attribute.

        Parameters
        ----------
        Wn : float | tuple
            The critical frequency or frequencies for the filter. For lowpass and highpass filters,
            Wn is a scalar; for bandpass and bandstop filters, Wn is a tuple.
        order : int, optional
            The order of the filter. Default is 8.
        btype : str, optional
            The type of filter to apply: 'lowpass', 'highpass
            'bandpass', or 'bandstop'. Default is 'lowpass'.
        inplace : bool, optional
            If True, updates the instance's data attribute with the filtered data.
            default is False.

        Raises
        ------
        ValueError
            If the order of the filter is not an integer greater than 0.

        Returns
        -------
        np.ndarray
            The filtered data.
            If 'inplace' is True, returns None.

        Notes
        -----
        For more information, see the scipy documentation for `signal.butter`
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
        and `signal.sosfiltfilt`
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
        """
        datasets = self.datasets
        if not inplace:
            datasets = copy.deepcopy(self.datasets)
        newdatasets = []
        for data in datasets:
            newdata = super()._filter_data(
                data=data,
                fs=self.fs,
                Wn=Wn,
                order=order,
                btype=btype,
            )
            newdatasets.append(newdata)

        Y = pre_MultiSetup(newdatasets, self.ref_ind)
        if inplace:
            self.data = Y
            return None
        return Y

    # method to detrend data
    def detrend_data(
        self,
        inplace: bool = False,
        **kwargs,
    ) -> typing.Optional[np.ndarray]:
        """
        Applies detrending to the data using the scipy.signal.detrend function.

        This method removes the linear trend from the data by fitting a least-squares
        polynomial to the data and subtracting it.
        The method updates the instance's data attribute.

        Parameters
        ----------
        inplace : bool, optional
            If True, updates the instance's data attribute with the detrended data.
            Default is False.
        **kwargs : dict, optional
            Additional keyword arguments for the scipy.signal.detrend function:
            axis : int, optional
                The axis along which to detrend the data. Default is 0.
            type : {'linear', 'constant'}, optional
                The type of detrending to apply: 'linear' for linear detrending
                or 'constant' for mean removal. Default is 'linear'.

        Returns
        -------
        np.ndarray
            The detrended data.
            If 'inplace' is True, returns None.

        See Also
        --------
        For further information, see `scipy.signal.detrend
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html>`_.
        """
        datasets = self.datasets
        if not inplace:
            datasets = copy.deepcopy(self.datasets)

        newdatasets = []
        for data in datasets:
            newdata = super()._detrend_data(data=data, **kwargs)
            newdatasets.append(newdata)

        Y = pre_MultiSetup(newdatasets, self.ref_ind)
        if inplace:
            self.data = Y
            return None
        return Y
