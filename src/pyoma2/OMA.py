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
import pickle
import typing

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import decimate, detrend

from pyoma2.algorithm.data.geometry import Geometry1, Geometry2
from pyoma2.algorithm.data.result import MsPoserResult
from pyoma2.functions.Gen_funct import (
    filter_data,
    find_map,
    flatten_sns_names,
    import_excel_GEO1,
    import_excel_GEO2,
    merge_mode_shapes,
    pre_MultiSetup,
)
from pyoma2.functions.plot_funct import (
    STFT,
    plt_ch_info,
    plt_data,
    plt_lines,
    plt_nodes,
    plt_quiver,
    plt_surf,
    set_ax_options,
    set_view,
)

if typing.TYPE_CHECKING:
    from pyoma2.algorithm import BaseAlgorithm

from pyoma2.plot.anim_mode import AniMode

logger = logging.getLogger(__name__)


class BaseSetup:
    """
    Base class for operational modal analysis (OMA) setups.

    This class provides foundational methods and attributes used by both
    SingleSetup and MultiSetup classes. It serves as a superclass for specific
    setup types, offering common functionalities for handling data, running algorithms, and
    extracting modal properties.

    Attributes
    ----------
    algorithms : dict[str, BaseAlgorithm]
        Dictionary storing algorithms added to the setup, keyed by their names.
    data : np.ndarray, optional
        Time series data array, typically representing the system's output.
    fs : float, optional
        Sampling frequency of the data.

    Methods
    -------
    add_algorithms(...)
        Adds algorithms to the setup and sets the data and sampling frequency for them.
    run_all(...)
        Runs all the algorithms added to the class.
    run_by_name(...)
        Executes a specific algorithm by its name.
    MPE(...)
        Extracts modal parameters from selected poles/peaks.
    MPE_fromPlot(...)
        Extracts modal parameters directly from plot selections.
    plot_geo1(...)
        Plots the first type of geometry setup for the structure.
    plot_geo2(...)
        Plots the second type of geometry setup for the structure.

    Warning
    -------
    The BaseSetup class is not intended for direct instantiation by users.
    It acts as a common interface for handling different types of setup configurations.
    Specific functionalities are provided through its subclasses.
    """

    algorithms: typing.Dict[str, BaseAlgorithm]
    data: typing.Optional[np.ndarray] = None  # TODO use generic typing
    fs: typing.Optional[float] = None  # sampling frequency
    Geo1: typing.Optional[Geometry1] = None
    Geo2: typing.Optional[Geometry2] = None

    # add algorithm (method) to the set.
    def add_algorithms(self, *algorithms: BaseAlgorithm):
        """
        Adds algorithms to the setup and configures them with data and sampling frequency.

        Parameters
        ----------
        algorithms : variable number of BaseAlgorithm
            One or more algorithm instances to be added to the setup.

        Notes
        -----
        The algorithms must be instantiated before adding them to the setup,
        and their names must be unique.
        """
        self.algorithms = {
            **getattr(self, "algorithms", {}),
            **{alg.name: alg._set_data(data=self.data, fs=self.fs) for alg in algorithms},
        }

    # run the whole set of algorithms (methods). METODO 1 di tutti
    def run_all(self):
        """
        Runs all the algorithms added to the setup.

        Iterates through each algorithm stored in the setup and executes it. The results are saved within
        each algorithm instance.

        Notes
        -----
        This method assumes that all algorithms are properly initialized and can be executed without
        additional parameters.
        """
        for alg_name in self.algorithms:
            self.run_by_name(name=alg_name)
        logger.info("all done")

    # run algorithm (method) by name. QUESTO è IL METODO 1 di un singolo
    def run_by_name(self, name: str):
        """
        Runs a specific algorithm by its name.

        Parameters
        ----------
        name : str
            The name of the algorithm to be executed.

        Raises
        ------
        KeyError
            If the specified algorithm name does not exist in the setup.

        Notes
        -----
        The result of the algorithm execution is saved within the algorithm instance.
        """
        logger.info("Running %s...", name)
        logger.debug("...with parameters: %s", self[name].run_params)
        self[name]._pre_run()
        result = self[name].run()
        logger.debug("...saving %s result", name)
        self[name]._set_result(result)

    # get the modal properties (all results).
    def MPE(self, name: str, *args, **kwargs):
        """
        Extracts modal parameters from selected poles/peaks of a specified algorithm.

        Parameters
        ----------
        name : str
            Name of the algorithm from which to extract modal parameters.
        args : tuple
            Positional arguments to be passed to the algorithm's MPE method.
        kwargs : dict
            Keyword arguments to be passed to the algorithm's MPE method.

        Raises
        ------
        KeyError
            If the specified algorithm name does not exist in the setup.
        """
        logger.info("Getting MPE modal parameters from %s", name)
        self[name].mpe(*args, **kwargs)

    # get the modal properties (all results) from the plots.
    def MPE_fromPlot(self, name: str, *args, **kwargs):
        """
        Extracts modal parameters directly from plot selections of a specified algorithm.

        Parameters
        ----------
        name : str
            Name of the algorithm from which to extract modal parameters.
        args : tuple
            Positional arguments to be passed to the algorithm's MPE method.
        kwargs : dict
            Keyword arguments to be passed to the algorithm's MPE method.

        Raises
        ------
        KeyError
            If the specified algorithm name does not exist in the setup.
        """
        logger.info("Getting MPE modal parameters from plot... %s", name)
        self[name].mpe_fromPlot(*args, **kwargs)

    def __getitem__(self, name: str) -> BaseAlgorithm:
        """
        Retrieves an algorithm from the setup by its name.

        Parameters
        ----------
        name : str
            The name of the algorithm to retrieve.

        Returns
        -------
        BaseAlgorithm
            The algorithm instance with the specified name.

        Raises
        ------
        KeyError
            If no algorithm with the given name exists in the setup.
        """
        if name in self.algorithms:
            return self.algorithms[name]
        else:
            raise KeyError(f"No algorithm named '{name}' exists.")

    def get(
        self, name: str, default: typing.Optional[BaseAlgorithm] = None
    ) -> typing.Optional[BaseAlgorithm]:
        """
        Retrieves an algorithm from the setup by its name, returning a default value if not found.

        Parameters
        ----------
        name : str
            The name of the algorithm to retrieve.
        default : BaseAlgorithm, optional
            The default value to return if the specified algorithm is not found.

        Returns
        -------
        BaseAlgorithm or None
            The algorithm instance with the specified name or the default value if not found.
        """
        return self.algorithms.get(name, default)

    # metodo per plottare geometria 1
    def plot_geo1(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        colsns: str = "red",
        colsns_lines: str = "red",
        colBG_nodes: str = "gray",
        colBG_lines: str = "gray",
        colBG_surf: str = "gray",
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
            If Geo1 is not defined in the setup.

        Returns
        -------
        tuple
            A tuple containing the figure and axis objects of the plot. This can be used for
            further customization or saving the plot externally.

        """
        if self.Geo1 is None:
            raise ValueError(
                f"Geo1 is not defined. cannot plot geometry on {self}. Call def_geo1 first."
            )
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Plot of the geometry and sensors' placement and direction")
        # plot sensors' nodes
        sens_coord = self.Geo1.sens_coord[["x", "y", "z"]].to_numpy()
        plt_nodes(ax, sens_coord, color=colsns)

        # plot sensors' directions
        plt_quiver(
            ax,
            sens_coord,
            self.Geo1.sens_dir,
            scaleF=scaleF,
            names=self.Geo1.sens_names,
            color=colsns,
            color_text=col_txt,
            method="2",
        )

        # Check that BG nodes are defined
        if self.Geo1.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, self.Geo1.bg_nodes, color=colBG_nodes, alpha=0.5)
            # Check that BG lines are defined
            if self.Geo1.bg_lines is not None:
                # if True plot
                plt_lines(
                    ax,
                    self.Geo1.bg_nodes,
                    self.Geo1.bg_lines,
                    color=colBG_lines,
                    alpha=0.5,
                )
            if self.Geo1.bg_surf is not None:
                # if True plot
                plt_surf(
                    ax, self.Geo1.bg_nodes, self.Geo1.bg_surf, alpha=0.1, color=colBG_surf
                )

        # check for sens_lines
        if self.Geo1.sens_lines is not None:
            # if True plot
            plt_lines(ax, sens_coord, self.Geo1.sens_lines, color=colsns_lines)

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
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        colsns: str = "red",
        colsns_lines: str = "black",
        colsns_surf: str = "lightcoral",
        colBG_nodes: str = "gray",
        colBG_lines: str = "gray",
        colBG_surf: str = "gray",
        col_txt: str = "red",
    ):
        """
        Plots the geometry (type 2) of tested structure.

        This method creates a 3D or 2D plot of a specific geometric configuration (Geo2) with
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
            If Geo2 is not defined in the setup.

        Returns
        -------
        tuple
            Returns a tuple containing the figure and axis objects of the matplotlib plot.
            This allows for further customization or saving outside the method.

        """
        if self.Geo2 is None:
            raise ValueError(
                f"Geo2 is not defined. Cannot plot geometry on {self}. Call def_geo2 first."
            )
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Plot of the geometry and sensors' placement and direction")
        # plot sensors'
        pts = self.Geo2.pts_coord.to_numpy()[:, :]
        plt_nodes(ax, pts, color="red")

        # plot sensors' directions
        ch_names = self.Geo2.sens_map.to_numpy()
        s_sign = self.Geo2.sens_sign.to_numpy().astype(float)  # array of signs

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
                color=colsns,
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
                color=colsns,
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
                color=colsns,
                color_text=col_txt,
                method="2",
            )

        # Check that BG nodes are defined
        if self.Geo2.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, self.Geo2.bg_nodes, color=colBG_nodes, alpha=0.5)
            # Check that BG lines are defined
            if self.Geo2.bg_lines is not None:
                # if True plot
                plt_lines(
                    ax,
                    self.Geo2.bg_nodes,
                    self.Geo2.bg_lines,
                    color=colBG_lines,
                    alpha=0.5,
                )
            if self.Geo2.bg_surf is not None:
                # if True plot
                plt_surf(
                    ax, self.Geo2.bg_nodes, self.Geo2.bg_surf, color=colBG_surf, alpha=0.1
                )

        # check for sens_lines
        if self.Geo2.sens_lines is not None:
            # if True plot
            plt_lines(ax, pts, self.Geo2.sens_lines, color=colsns_lines)

        if self.Geo2.sens_surf is not None:
            # if True plot
            plt_surf(
                ax,
                self.Geo2.pts_coord.values,
                self.Geo2.sens_surf,
                color=colsns_surf,
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
        Defines the first geometry setup (Geo1) for the instance.

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
        ref_ind = self.ref_ind if self.ref_ind is not None else None
        sens_names = flatten_sns_names(sens_names, ref_ind=ref_ind)
        # ---------------------------------------------------------------------
        # Find the indices that rearrange sens_coord to sens_names
        newIDX = find_map(sens_names, sens_coord.index.to_numpy())
        # reorder if necessary
        sens_coord = sens_coord.reindex(labels=newIDX)
        sens_dir = sens_dir[newIDX, :]

        self.Geo1 = Geometry1(
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
        ref_ind = self.ref_ind if self.ref_ind is not None else None

        data = import_excel_GEO1(path, ref_ind=ref_ind)

        self.Geo1 = Geometry1(
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
        cstrn: pd.DataFrame = None,
        sens_sign: pd.DataFrame = None,
        sens_lines: npt.NDArray[np.int64] = None,  # lines connecting sensors
        sens_surf: npt.NDArray[np.int64] = None,  # surf connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None,  # Background nodes
        bg_lines: npt.NDArray[np.float64] = None,  # Background lines
        bg_surf: npt.NDArray[np.float64] = None,  # Background lines
    ):
        """
        Defines the second geometry setup (Geo2) for the instance.

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
        # ---------------------------------------------------------------------
        # Check if sens_names is a DataFrame with more than one row or a list of lists
        # FOR MULTI-SETUP GEOMETRIES

        ref_ind = self.ref_ind if self.ref_ind is not None else None
        sens_names = flatten_sns_names(sens_names, ref_ind=ref_ind)

        # ---------------------------------------------------------------------
        if sens_sign is None:
            sens_sign = pd.DataFrame(
                np.ones(sens_map.to_numpy()[:, :].shape), columns=sens_map.columns
            )
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

        self.Geo2 = Geometry2(
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

    def def_geo2_byFILE(self, path: str):
        ref_ind = self.ref_ind if hasattr(self, "ref_ind") else None

        data = import_excel_GEO2(path, ref_ind=ref_ind)

        self.Geo2 = Geometry2(
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

    def save_to_file(self, file_name):
        """ """
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, file_name):
        """ """
        with open(file_name, "rb") as f:
            instance = pickle.load(f)  # noqa S301
        return instance

    # method to decimate data
    @staticmethod
    def _decimate_data(data: np.ndarray, fs: float, q: int, **kwargs) -> tuple:
        """
        Applies decimation to the data using the scipy.signal.decimate function.

        This method reduces the sampling rate of the data by a factor of 'q'.
        The decimation process includes low-pass filtering to reduce aliasing.
        The method updates the instance's data and sampling frequency attributes.

        Parameters
        ----------
        data : np.ndarray
            The input data to be decimated.
        q : int
            The decimation factor. Must be greater than 1.
        axis : int, optional
            The axis along which to decimate the data. Default is 0.
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
            A tuple containing the decimated data, updated sampling frequency, sampling interval,

        See Also
        --------
        For further information, see `scipy.signal.decimate
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html>`_.
        """
        newdata = decimate(data, q, **kwargs)
        fs = fs / q
        dt = 1 / fs
        Ndat = newdata.shape[0]
        T = 1 / fs / q * Ndat
        return newdata, fs, dt, Ndat, T

    # method to detrend data
    @staticmethod
    def _detrend_data(data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies detrending to the data using the scipy.signal.detrend function.

        This method removes a linear or constant trend from the data, commonly used to remove drifts
        or offsets in time series data. It's a preprocessing step, often necessary for methods that
        assume stationary data. The method updates the instance's data attribute.

        Parameters
        ----------
        data : np.ndarray
            The input data to be detrended.
        axis : int, optional
            The axis along which to detrend the data. Default is 0.
        **kwargs : dict, optional, will be passed to scipy.signal.detrend
            Additional keyword arguments for the scipy.signal.detrend function:
            type : {'linear', 'constant'}, optional
                The type of detrending: 'linear' for linear detrend, or 'constant' for just
                subtracting the mean. Default is 'linear'.
            bp : int or numpy.ndarray of int, optional
                Breakpoints where the data is split for piecewise detrending. Default is 0.

        Raises
        ------
        ValueError
            If invalid parameters are provided.

        Returns
        -------
        np.ndarray
            The detrended data.

        See Also
        --------
        For further information, see `scipy.signal.detrend
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html>`_.
        """
        axis = kwargs.pop("axis", 0)
        return detrend(data, axis=axis, **kwargs)

    # method to detrend data
    @staticmethod
    def _filter_data(
        data: np.ndarray,
        fs: float,
        Wn: typing.Union[float, typing.Tuple[float, float]],
        order: int = 8,
        btype: str = "lowpass",
    ) -> np.ndarray:
        """
        Apply a Butterworth filter to the input data and return the filtered signal.

        This function designs and applies a Butterworth filter with the specified parameters to the input
        data. It can be used to apply lowpass, highpass, bandpass, or bandstop filters.

        Parameters
        ----------
        data : ndarray
            The input signal data to be filtered. The filter is applied along the first axis
            (i.e., each column is filtered independently).
        fs : float
            The sampling frequency of the input data.
        Wn : array_like
            The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar;
            for bandpass and bandstop filters, Wn is a length-2 sequence.
        order : int, optional
            The order of the filter. A higher order leads to a sharper frequency cutoff but can also
            lead to instability and significant phase delay. Default is 8.
        btype : str, optional
            The type of filter to apply. Options are "lowpass", "highpass", "bandpass", or "bandstop".
            Default is "lowpass".

        Returns
        -------
        filt_data : ndarray
            The filtered signal, with the same shape as the input data.

        Notes
        -----
        For more information, see the scipy documentation for `signal.butter`
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
        and `signal.sosfiltfilt`
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
        """
        return filter_data(
            data=data,
            fs=fs,
            Wn=Wn,
            order=order,
            btype=btype,
        )


class SingleSetup(BaseSetup):
    """
    Class for managing and processing single setup data for Operational Modal Analysis.

    This class handles data from a single setup, offering functionalities like plotting,
    data processing, and interaction with various analysis algorithms. It inherits
    attributes and methods from the BaseSetup class.

    Parameters
    ----------
    data : Iterable[float]
        The data to be processed, expected as an iterable of floats.
    fs : float
        The sampling frequency of the data.

    Attributes
    ----------
    data : Iterable[float]
        Stores the input data.
    fs : float
        Stores the sampling frequency.
    dt : float
        The sampling interval, calculated as the inverse of the sampling frequency.
    algorithms : Dict[str, BaseAlgorithm]
        A dictionary to store algorithms associated with the setup.

    Methods
    -------
    plot_data(...)
        Plots the time histories of the data channels in a subplot format.
    plot_ch_info(...)
        Plots Time History (TH), Power Spectral Density (PSD),
        and Kernel Density Estimation (KDE) for each channel.
    plt_STFT(...)
        Plots the Short Time Fourier Transform (STFT) magnitude spectrogram for specified channels.
    decimate_data(...)
        Decimates the data using a wrapper for the scipy.signal.decimate function.
    detrend_data(...)
        Detrends the data using a wrapper for the scipy.signal.detrend function.
    filter_data(...)
        Applies a Butterworth filter to the input data based on specified parameters.
    def_geo1(...)
        Defines the first geometry setup (Geo1) for the instances.
    def_geo2(...)
        Defines the second geometry setup (Geo2) for the instance..

    Notes
    -----
    The ``algorithms`` dictionary is initialized empty and is meant to store various algorithms as needed.
    """

    Geo1: typing.Optional[Geometry1] = None
    Geo2: typing.Optional[Geometry2] = None

    def __init__(self, data: np.ndarray, fs: float):
        """
        Initialize a SingleSetup instance with data and sampling frequency.

        Parameters
        ----------
        data : np.ndarray
            The data to be processed, expected as a 2D array of shape (N, M)
        fs : float
            The sampling frequency of the data.
        """
        self.data = data  # data
        self.fs = fs  # sampling frequency [Hz]
        self.dt = 1 / fs  # sampling interval
        self.Nch = data.shape[1]  # number of channels
        self.Ndat = data.shape[0]  # number of data points
        self.T = self.dt * self.Ndat  # Period of acquisition [sec]

        self.algorithms: typing.Dict[str, BaseAlgorithm] = {}  # set of algo

    # method to plot the time histories of the data channels.
    def plot_data(
        self,
        nc: int = 1,
        names: typing.Optional[typing.List[str]] = None,
        unit: str = "unit",
        show_rms: bool = False,
    ):
        """
        Plots the time histories of the data channels in a subplot format.

        Parameters
        ----------
        nc : int, optional
            Number of columns for the subplot. Default is 1.
        names : List[str], optional
            List of names for the channels. If provided, these names are used as labels.
            Default is None.
        unit : str, optional
            String label for the y-axis representing the unit of measurement. Default is "unit".
        show_rms : bool, optional
            If True, the RMS acceleration is shown in the plot. Default is False.

        Returns
        -------
        tuple
            A tuple containing the figure and axis objects of the plot for further customization
            or saving externally.
        """
        data = self.data
        fs = self.fs
        nc = nc  # number of columns for subplot
        names = names  # list of names (str) of the channnels
        unit = unit  # str label for the y-axis (unit of measurement)
        show_rms = show_rms  # wheter to show or not the rms acc in the plot
        fig, ax = plt_data(data, fs, nc, names, unit, show_rms)
        return fig, ax

    # Method to plot info on channel (TH,auto-corr,PSD,PDF,dataVSgauss)
    def plot_ch_info(
        self,
        nxseg: float = 1024,
        ch_idx: typing.Union[str, typing.List[int]] = "all",
        freqlim: typing.Optional[tuple[float, float]] = None,
        logscale: bool = True,
        unit: str = "unit",
    ):
        """
        Plot channel information including time history, normalized auto-correlation,
        power spectral density (PSD), probability density function, and normal probability
        plot for each channel in the data.

        Parameters
        ----------
        data : ndarray
            The input signal data.
        fs : float
            The sampling frequency of the input data.
        nxseg : int, optional
            The number of points per segment for the PSD estimation. Default is 1024.
        freqlim : tuple of float, optional
            The frequency limits (min, max) for the PSD plot. Default is None.
        logscale : bool, optional
            If True, the PSD plot will use a logarithmic scale. Default is False.
        ch_idx : str or list of int, optional
            The indices of the channels to plot. If "all", information for all channels is plotted.
            Default is "all".
        unit : str, optional
            The unit of the input data for labeling the plots. Default is "unit".

        Returns
        -------
        figs : list of matplotlib.figure.Figure
            A list of figure objects, one for each channel plotted.
        axs : list of matplotlib.axes.Axes
            A list of Axes objects corresponding to the subplots for each channel's information.

        Notes
        -----
        This function provides a comprehensive overview of the signal characteristics for one or
        multiple channels of a dataset. It's useful for initial signal analysis and understanding
        signal properties.
        """
        data = self.data
        fs = self.fs
        fig, ax = plt_ch_info(
            data,
            fs,
            ch_idx=ch_idx,
            freqlim=freqlim,
            logscale=logscale,
            nxseg=nxseg,
            unit=unit,
        )
        return fig, ax

    # Method to plot Short Time Fourier Transform
    def plot_STFT(
        self,
        nxseg: float = 512,
        pov: float = 0.9,
        ch_idx: typing.Union[str, typing.List[int]] = "all",
        freqlim: typing.Optional[tuple[float, float]] = None,
        win: str = "hann",
    ):
        """
        Plot the Short-Time Fourier Transform (STFT) magnitude spectrogram for the specified channels.

        This function computes and plots the STFT magnitude spectrogram for each selected channel in the
        data. The spectrogram is plotted as a heatmap where the x-axis represents time, the y-axis
        represents frequency, and the color intensity represents the magnitude of the STFT.

        Parameters
        ----------
        data : ndarray
            The input signal data.
        fs : float
            The sampling frequency of the input data.
        nxseg : int, optional
            The number of data points used in each segment of the STFT. Default is 512.
        pov : float, optional
            The proportion of overlap between consecutive segments, expressed as a decimal between 0 and 1.
            Default is 0.9.
        win : str, optional
            The type of window function to apply to each segment. Default is "hann".
        freqlim : tuple of float, optional
            The frequency limits (min, max) for the spectrogram display. Default is None, which uses the
            full frequency range.
        ch_idx : str or list of int, optional
            The indices of the channels to plot. If "all", the STFT for all channels is plotted.
            Default is "all".

        Returns
        -------
        figs : list of matplotlib.figure.Figure
            A list of figure objects, one for each channel plotted.
        axs : list of matplotlib.axes.Axes
            A list of Axes objects corresponding to the figures.

        Notes
        -----
        This function is designed to visualize the frequency content of signals over time, which can be
        particularly useful for analyzing non-stationary signals.
        """

        data = self.data
        fs = self.fs
        fig, ax = STFT(
            data,
            fs,
            nxseg=nxseg,
            pov=pov,
            win=win,
            ch_idx=ch_idx,
            freqlim=freqlim,
        )
        return fig, ax

    def decimate_data(
        self, q: int, inplace: bool = False, **kwargs
    ) -> typing.Optional[tuple]:
        """
        Decimates the data using the scipy.signal.decimate function.

        This method reduces the sampling rate of the data by a factor of 'q'.
        The decimation process includes low-pass filtering to reduce aliasing.
        The method updates the instance's data and sampling frequency attributes.

        Parameters
        ----------
        q : int
            The decimation factor. Must be greater than 1.
        inplace : bool, optional
            If True, the data is decimated in place. If False, a copy of the decimated data is returned.
            Default is False.
        **kwargs : dict, optional
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
            A tuple containing the decimated data, updated sampling frequency, sampling interval,
            number of data points, and total time period.
            If 'inplace' is True, returns None.
        """
        axis = kwargs.pop("axis", 0)
        data = self.data
        if not inplace:
            data = copy.deepcopy(self.data)
        decimated_data, fs, dt, Ndat, T = super()._decimate_data(
            data=data, fs=self.fs, q=q, axis=axis, **kwargs
        )
        if inplace:
            self.data = decimated_data
            self.fs = fs
            self.dt = dt
            self.Ndat = Ndat
            self.T = T
            return None
        return decimated_data, fs, dt, Ndat, T

    def detrend_data(
        self, inplace: bool = False, **kwargs
    ) -> typing.Optional[np.ndarray]:
        """
        Detrends the data using the scipy.signal.detrend function.

        This method removes a linear or constant trend from the data, commonly used to remove drifts
        or offsets in time series data. It's a preprocessing step, often necessary for methods that
        assume stationary data. The method updates the instance's data attribute.

        Parameters
        ----------
        in_place : bool, optional
            If True, the data is detrended in place. If False, a copy of the detrended data is returned.
            Default is False.
        **kwargs : dict, optional
            Additional keyword arguments for the scipy.signal.detrend function:
            type : {'linear', 'constant'}, optional
                The type of detrending: 'linear' for linear detrend, or 'constant' for just
                subtracting the mean. Default is 'linear'.
            bp : int or numpy.ndarray of int, optional
                Breakpoints where the data is split for piecewise detrending. Default is 0.

        Raises
        ------
        ValueError
            If invalid parameters are provided.

        Returns
        -------
        detrended_data : np.ndarray
            The detrended data if 'inplace' is False, otherwise None.
        """
        data = self.data
        if not inplace:
            data = copy.deepcopy(self.data)
        detrended_data = super()._detrend_data(data=data, **kwargs)
        if inplace:
            self.data = detrended_data
            return None
        return detrended_data

    def filter_data(
        self,
        Wn: typing.Union[float, typing.Tuple[float, float]],
        order: int = 8,
        btype: str = "lowpass",
        inplace: bool = False,
    ) -> typing.Optional[np.ndarray]:
        """
        Apply a Butterworth filter to the input data and return the filtered signal.

        This function designs and applies a Butterworth filter with the specified parameters to the input
        data. It can be used to apply lowpass, highpass, bandpass, or bandstop filters.

        Parameters
        ----------
        Wn : float or tuple of float
            The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar;
            for bandpass and bandstop filters, Wn is a length-2 sequence.
        order : int, optional
            The order of the filter. A higher order leads to a sharper frequency cutoff but can also
            lead to instability and significant phase delay. Default is 8.
        btype : str, optional
            The type of filter to apply. Options are "lowpass", "highpass", "bandpass", or "bandstop".
            Default is "lowpass".
        inplace: bool, optional
            If True, the data is filtered in place. If False, a copy of the filtered data is returned.
            Default is False.

        Returns
        -------
        filt_data : ndarray
            The filtered signal, with the same shape as the input data. If 'inplace' is True, returns None.

        Notes
        -----
        For more information, see the scipy documentation for `signal.butter`
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
        and `signal.sosfiltfilt`
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
        """
        data = self.data
        if not inplace:
            data = copy.deepcopy(self.data)
        filt_data = super()._filter_data(
            data=data,
            fs=self.fs,
            Wn=Wn,
            order=order,
            btype=btype,
        )
        if inplace:
            self.data = filt_data
            return None
        return filt_data


# =============================================================================
# MULTISETUP
# =============================================================================
# FIXME add references!
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
        Plots mode shapes for a specified mode number using the first type of geometric setup (Geo1).
    plot_mode_g2(mode_number: int, scale_factor: int, view_type: str)
        Plots mode shapes for a specified mode number using the second type of geometric setup (Geo2).
    anim_mode_g2(mode_number: int, scale_factor: int, view_type: str, save_as_gif: bool)
        Creates an animation of the mode shapes for a specified mode number using the second type
        of geometric setup (Geo2). Option to save the animation as a GIF file.
    def_geo1(...)
        Defines the first type of geometric setup (Geo1) for the instance, based on sensor placements
        and structural characteristics.
    def_geo2(...)
        Defines the second type of geometric setup (Geo2) for the instance, typically involving more
        complex geometries or additional data.

    plot_geo1(...)
        Plots the geometric configuration of the structure based on the Geo1 setup, including sensor
        placements and structural details.
    plot_geo2(...)
        Plots the geometric configuration of the structure based on the Geo2 setup, highlighting
        more intricate details or alternative layouts.

    Warning
    -------
    The PoSER approach assumes that the setups used are compatible in terms of their experimental
    setup and data characteristics.
    """

    __result: typing.Optional[typing.Dict[str, MsPoserResult]] = None
    __alg_ref: typing.Optional[typing.Dict[type[BaseAlgorithm], str]] = None
    Geo1: typing.Optional[Geometry1] = None
    Geo2: typing.Optional[Geometry2] = None

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
        Defines the first geometry setup (Geo1) for the instance.

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
        ref_ind = self.ref_ind if self.ref_ind is not None else None
        sens_names = flatten_sns_names(sens_names, ref_ind=ref_ind)
        # ---------------------------------------------------------------------
        # Find the indices that rearrange sens_coord to sens_names
        newIDX = find_map(sens_names, sens_coord.index.to_numpy())
        # reorder if necessary
        sens_coord = sens_coord.reindex(labels=newIDX)
        sens_dir = sens_dir[newIDX, :]

        self.Geo1 = Geometry1(
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
        ref_ind = self.ref_ind if self.ref_ind is not None else None

        data = import_excel_GEO1(path, ref_ind=ref_ind)

        self.Geo1 = Geometry1(
            sens_names=data[0],
            sens_coord=data[1],
            sens_dir=data[2].values,
            sens_lines=data[3],
            bg_nodes=data[4],
            bg_lines=data[5],
            bg_surf=data[6],
        )

    # metodo per plottare geometria 1
    def plot_geo1(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        colsns: str = "red",
        colsns_lines: str = "red",
        colBG_nodes: str = "gray",
        colBG_lines: str = "gray",
        colBG_surf: str = "gray",
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
            If Geo1 is not defined in the setup.

        Returns
        -------
        tuple
            A tuple containing the figure and axis objects of the plot. This can be used for
            further customization or saving the plot externally.

        """
        if self.Geo1 is None:
            raise ValueError(
                f"Geo1 is not defined. cannot plot geometry on {self}. Call def_geo1 first."
            )
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Plot of the geometry and sensors' placement and direction")
        # plot sensors' nodes
        sens_coord = self.Geo1.sens_coord[["x", "y", "z"]].to_numpy()
        plt_nodes(ax, sens_coord, color=colsns)

        # plot sensors' directions
        plt_quiver(
            ax,
            sens_coord,
            self.Geo1.sens_dir,
            scaleF=scaleF,
            names=self.Geo1.sens_names,
            color=colsns,
            color_text=col_txt,
            method="2",
        )

        # Check that BG nodes are defined
        if self.Geo1.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, self.Geo1.bg_nodes, color=colBG_nodes, alpha=0.5)
            # Check that BG lines are defined
            if self.Geo1.bg_lines is not None:
                # if True plot
                plt_lines(
                    ax,
                    self.Geo1.bg_nodes,
                    self.Geo1.bg_lines,
                    color=colBG_lines,
                    alpha=0.5,
                )
            if self.Geo1.bg_surf is not None:
                # if True plot
                plt_surf(
                    ax, self.Geo1.bg_nodes, self.Geo1.bg_surf, alpha=0.1, color=colBG_surf
                )

        # check for sens_lines
        if self.Geo1.sens_lines is not None:
            # if True plot
            plt_lines(ax, sens_coord, self.Geo1.sens_lines, color=colsns_lines)

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
        Defines the second geometry setup (Geo2) for the instance, incorporating sensors' names,
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

        ref_ind = self.ref_ind if self.ref_ind is not None else None
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

        self.Geo2 = Geometry2(
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
        ref_ind = self.ref_ind if self.ref_ind is not None else None

        data = import_excel_GEO2(path, ref_ind=ref_ind)

        self.Geo2 = Geometry2(
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

    def plot_geo2(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        colsns: str = "red",
        colsns_lines: str = "black",
        colsns_surf: str = "lightcoral",
        colBG_nodes: str = "gray",
        colBG_lines: str = "gray",
        colBG_surf: str = "gray",
        col_txt: str = "red",
    ):
        """
        Plots the geometry (type 2) of tested structure.

        This method creates a 3D or 2D plot of a specific geometric configuration (Geo2) with
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
            If Geo2 is not defined in the setup.

        Returns
        -------
        tuple
            Returns a tuple containing the figure and axis objects of the matplotlib plot.
            This allows for further customization or saving outside the method.

        """
        if self.Geo2 is None:
            raise ValueError(
                f"Geo2 is not defined. Cannot plot geometry on {self}. Call def_geo2 first."
            )
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Plot of the geometry and sensors' placement and direction")
        # plot sensors'
        pts = self.Geo2.pts_coord.to_numpy()[:, :]
        plt_nodes(ax, pts, color="red")

        # plot sensors' directions
        ch_names = self.Geo2.sens_map.to_numpy()
        s_sign = self.Geo2.sens_sign.to_numpy().astype(float)  # array of signs

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
                color=colsns,
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
                color=colsns,
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
                color=colsns,
                color_text=col_txt,
                method="2",
            )
        # Check that BG nodes are defined
        if self.Geo2.bg_nodes is not None:
            # if True plot
            plt_nodes(ax, self.Geo2.bg_nodes, color=colBG_nodes, alpha=0.5)
            # Check that BG lines are defined
            if self.Geo2.bg_lines is not None:
                # if True plot
                plt_lines(
                    ax,
                    self.Geo2.bg_nodes,
                    self.Geo2.bg_lines,
                    color=colBG_lines,
                    alpha=0.5,
                )
            if self.Geo2.bg_surf is not None:
                # if True plot
                plt_surf(
                    ax, self.Geo2.bg_nodes, self.Geo2.bg_surf, color=colBG_surf, alpha=0.1
                )

        # check for sens_lines
        if self.Geo2.sens_lines is not None:
            # if True plot
            plt_lines(ax, pts, self.Geo2.sens_lines, color=colsns_lines)

        if self.Geo2.sens_surf is not None:
            # if True plot
            plt_surf(
                ax,
                self.Geo2.pts_coord.values,
                self.Geo2.sens_surf,
                color=colsns_surf,
                alpha=0.5,
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
        """
        Plots the mode shapes for a specified mode number from the results of an algorithm,
        using Geometry 1 setup.

        Parameters
        ----------
        Algo_Res : MsPoserResult
            The results from an algorithm, containing mode shapes and other modal properties.
        Geo1 : Geometry1
            The first geometry setup of the structure.
        mode_numb : int
            The mode number to be visualized.
        scaleF : int, optional
            Scaling factor for the mode shape visualization, by default 1.
        view : Literal["3D", "xy", "xz", "yz", "x", "y", "z"], optional
            The type of view for plotting the mode shape, by default "3D".
        remove_fill : bool, optional
            If True, removes the fill from the plot, by default True.
        remove_grid : bool, optional
            If True, removes the grid from the plot, by default True.
        remove_axis : bool, optional
            If True, removes the axis labels and ticks from the plot, by default True.

        Returns
        -------
        tuple
            A tuple containing the figure and axis objects of the plot.
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
            #            names=Geo1.sens_names,
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
            scaleF=scaleF,
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
        color: str = "cmap",
        *args,
        **kwargs,
    ) -> typing.Any:
        """
        Plots the mode shapes for the specified mode number from the results of an algorithm,
        using Geometry 2 setup.

        Parameters
        ----------
        Algo_Res : MsPoserResult
            The results from an algorithm, containing mode shapes and other modal properties.
        Geo2 : Geometry2
            The geometry (type2) of the structure.
        mode_numb : Optional[int]
            The mode number to be visualized.
        scaleF : int, optional
            Scaling factor for the mode shape visualization, by default 1.
        view : Literal["3D", "xy", "xz", "yz", "x", "y", "z"], optional
            The type of view for plotting the mode shape, by default "3D".
        remove_fill : bool, optional
            If True, removes the fill from the plot, by default True.
        remove_grid : bool, optional
            If True, removes the grid from the plot, by default True.
        remove_axis : bool, optional
            If True, removes the axis labels and ticks from the plot, by default True.

        Returns
        -------
        tuple
            A tuple containing the figure and axis objects of the plot.
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

        if Geo2.cstrn is not None:
            aa = Geo2.cstrn.to_numpy(na_value=0)[:, 1:]
            aa = np.nan_to_num(aa, copy=True, nan=0.0)
            val = aa @ phi
            ctn_df = pd.DataFrame(
                {"cName": Geo2.cstrn.to_numpy()[:, 0], "val": val},
            )

            mapping = dict(zip(df_phi["sName"], df_phi["Phi"]))
            mapping1 = dict(zip(ctn_df["cName"], ctn_df["val"]))
            mapp = dict(mapping, **mapping1)
        else:
            mapp = dict(zip(df_phi["sName"], df_phi["Phi"]))

        # reshape the mode shape dataframe to fit the pts coord
        df_phi_map = Geo2.sens_map.replace(mapp).astype(float)
        # add together coordinates and mode shape displacement
        # newpoints = Geo2.pts_coord.add(df_phi_map * Geo2.sens_sign, fill_value=0)
        newpoints = (
            Geo2.pts_coord.to_numpy() + df_phi_map.to_numpy() * Geo2.sens_sign.to_numpy()
        )
        # extract only the displacement array
        # newpoints = newpoints.to_numpy()[:, 1:]

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
        if color == "cmap":
            oldpoints = Geo2.pts_coord.to_numpy()[:, :]
            plt_nodes(ax, newpoints, color="cmap", initial_coord=oldpoints)

        else:
            plt_nodes(ax, newpoints, color=color)
        # check for sens_lines
        if Geo2.sens_lines is not None:
            if color == "cmap":
                plt_lines(
                    ax, newpoints, Geo2.sens_lines, color="cmap", initial_coord=oldpoints
                )
            else:
                plt_lines(ax, newpoints, Geo2.sens_lines, color=color)

        if Geo2.sens_surf is not None:
            if color == "cmap":
                plt_surf(
                    ax,
                    newpoints,
                    Geo2.sens_surf,
                    color="cmap",
                    initial_coord=oldpoints,
                    alpha=0.4,
                )
            else:
                plt_surf(ax, newpoints, Geo2.sens_surf, color=color, alpha=0.4)

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
        saveGIF: bool = False,
        *args,
        **kwargs,
    ) -> typing.Any:
        """
        Creates an animation of the mode shapes for the specified mode number from the results
        of an algorithm, using Geometry 2 setup.

        This method visualizes the animated mode shapes of a structure based on the results obtained
        from a specific algorithm, using geometry type 2 configuration.

        Parameters
        ----------
        Algo_Res : MsPoserResult
            The results from an algorithm, containing mode shapes and other modal properties.
        Geo2 : Geometry2
            The geometry (type2) of the structure.
        mode_numb : Optional[int]
            The mode number to be visualized.
        scaleF : int, optional
            Scaling factor for the mode shape visualization, by default 1.
        view : Literal["3D", "xy", "xz", "yz", "x", "y", "z"], optional
            The type of view for plotting the mode shape, by default "3D".
        remove_fill : bool, optional
            If True, removes the fill from the plot, by default True.
        remove_grid : bool, optional
            If True, removes the grid from the plot, by default True.
        remove_axis : bool, optional
            If True, removes the axis labels and ticks from the plot, by default True.
        saveGIF : bool, optional
            If True, saves the animation as a GIF file, by default False.

        Returns
        -------
        Any
            The generated animation object.
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
            saveGIF=saveGIF,
        )
        logger.info("...end AniMode FDD...")


# -----------------------------------------------------------------------------


# FIXME add reference!
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
        Defines the first type of geometric setup (Geo1) for the instance.
    def_geo2(...)
        Defines the second type of geometric setup (Geo2) for the instance.

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
