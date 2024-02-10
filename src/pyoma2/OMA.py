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

    Notes
    -----
    The BaseSetup class is not intended for direct instantiation by users.
    It acts as a common interface for handling different types of setup configurations.
    Specific functionalities are provided through its subclasses.
    """

    algorithms: typing.Dict[str, BaseAlgorithm]
    data: typing.Optional[np.ndarray] = None  # TODO use generic typing
    fs: typing.Optional[float] = None  # sampling frequency

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
            **self.algorithms,
            **{alg.name: alg.set_data(data=self.data, fs=self.fs) for alg in algorithms},
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
        result = self[name].run()
        logger.debug("...saving %s result", name)
        self[name].set_result(result)

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
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
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

        Returns
        -------
        tuple
            A tuple containing the figure and axis objects of the plot. This can be used for
            further customization or saving the plot externally.

        """
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Plot of the geometry and sensors' placement and direction")
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

        Returns
        -------
        tuple
            Returns a tuple containing the figure and axis objects of the matplotlib plot.
            This allows for further customization or saving outside the method.

        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Plot of the geometry and sensors' placement and direction")
        # plot sensors'
        pts = self.Geo2.pts_coord.to_numpy()[:, 1:]
        plt_nodes(ax, pts, color="red")

        # plot sensors' directions
        ch_names = self.Geo2.sens_map.to_numpy()[:, 1:]
        s_sign = self.Geo2.sens_sign.to_numpy()[:, 1:]  # array of signs
        # N.B. the size of s_sign will vary depending on the order_red
        # parameter. order_red="None" size(npts,3);
        # order_red="xy/xz/yz" size(npts,2);
        # order_red="x/y/z" size(npts,1)
        # (same for ch_names)
        ord_red = self.Geo2.order_red
        zero1 = np.zeros(s_sign.shape[0]).reshape(-1, 1)
        zero2 = np.zeros((s_sign.shape[0], 2))
        if ord_red is None:
            pass
        elif ord_red == "xy":
            s_sign = np.hstack((s_sign, zero1))
            ch_names = np.hstack((ch_names, zero1))
        elif ord_red == "xz":
            s_sign = np.insert(s_sign, 1, 0)
            ch_names = np.insert(ch_names, 1, 0)
        elif ord_red == "yz":
            s_sign = np.hstack((zero1, s_sign))
            ch_names = np.hstack((zero1, ch_names))
        elif ord_red == "x":
            s_sign = np.hstack((s_sign, zero2))
            ch_names = np.hstack((ch_names, zero2))
        elif ord_red == "y":
            s_sign = np.insert(zero2, 1, s_sign)
            ch_names = np.insert(zero2, 1, ch_names)
        elif ord_red == "z":
            s_sign = np.hstack((zero2, s_sign))
            ch_names = np.hstack((zero2, ch_names))

        s_sign[s_sign == 0] = np.nan
        ch_names[ch_names == 0] = np.nan
        for ii in range(3):
            s_sign1 = np.hstack((s_sign[:, 0].reshape(-1, 1), zero2))
            s_sign2 = np.insert(zero2, 1, s_sign[:, 1], axis=1)
            s_sign3 = np.hstack((zero2, s_sign[:, 2].reshape(-1, 1)))

            plt_quiver(
                ax,
                pts,
                s_sign1,
                scaleF=scaleF,
                names=ch_names[:, 0],
            )
            plt_quiver(
                ax,
                pts,
                s_sign2,
                scaleF=scaleF,
                names=ch_names[:, 1],
            )
            plt_quiver(
                ax,
                pts,
                s_sign3,
                scaleF=scaleF,
                names=ch_names[:, 2],
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
            plt_lines(ax, pts, self.Geo2.sens_lines, color="red")

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
    """
    Class for managing and processing single setup data for operational modal analysis.

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
    decimate_data(...)
        Decimates the data using a wrapper for the scipy.signal.decimate function.
    detrend_data(...)
        Detrends the data using a wrapper for the scipy.signal.detrend function.
    def_geo1(...)
        Defines the first geometry setup (Geo1) for the instance.
    def_geo2(...)
        Defines the second geometry setup (Geo2) for the instance.
    __getitem__(...)
        Retrieve an algorithm from the set by its name.
    get(...)
        Retrieve an algorithm from the set by its name, returning a default value if it does not exist.

    Notes
    -----
    - The sampling interval `dt` is calculated automatically from the provided sampling frequency.
    - `algorithms` dictionary is initialized empty and is meant to store various algorithms as needed.
    """

    def __init__(self, data: typing.Iterable[float], fs: float):
        """
        Initialize a SingleSetup instance with data and sampling frequency.

        Parameters
        ----------
        data : Iterable[float]
            The data to be processed, expected as an iterable of floats.
        fs : float
            The sampling frequency of the data.
        """
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
        freqlim: tuple[float, float] | None = None,
        logscale: bool = True,
        nxseg: float | None = None,
        pov: float = 0.0,
        window: str = "boxcar",
    ):
        """
        Plots Time History (TH), Power Spectral Density (PSD),
        and Kernel Density Estimation (KDE) for each channel.

        Parameters
        ----------
        ch_idx : str | List[int], optional
            Channel indices to be plotted. Can be a list of indices or 'all' for all channels.
            Default is 'all'.
        ch_names : List[str], optional
            List of channel names for labeling purposes. Default is None.
        freqlim : Tuple[float, float] | None, optional
            Frequency limit for the plots. Default is None.
        logscale : bool, optional
            If True, the PSD plot is in logarithmic scale. Default is True.
        nxseg : float | None, optional
            Number of segments for the Welch method in PSD calculation. Default is None.
        pov : float, optional
            Percentage of overlap for the segments in PSD calculation. Default is 0.
        window : str, optional
            Windowing function to be used in PSD calculation. Default is 'boxcar'.

        Returns
        -------
        tuple
            A tuple containing the figure and axis objects of the plots.

        Notes
        -----
        Utilizes the `plt_ch_info` function for plotting.
        """
        data = self.data
        fs = self.fs

        fig, ax = plt_ch_info(
            data,
            fs,
            ch_idx,
            ch_names=ch_names,
            freqlim=freqlim,
            logscale=logscale,
            nxseg=nxseg,
            pov=pov,
            window=window,
        )
        return fig, ax

    # method to decimate data
    def decimate_data(
        self,
        q: int,
        n: int | None = None,
        ftype: typing.Literal["iir", "fir"] = "iir",
        axis: int = 0,
        zero_phase: bool = True,
    ):
        """
        Applies decimation to the data using the scipy.signal.decimate function.

        This method reduces the sampling rate of the data by a factor of 'q'.
        Parameters
        ----------
        q : int
            The decimation factor.
        n : Optional[int], optional
            The order of the filter (if 'ftype' is 'fir') or the number of times to apply
            the filter (if 'ftype' is 'iir'). If None, a default value is used. Default is None.
        ftype : {'iir', 'fir'}, optional
            The type of filter to use for decimation: 'iir' for an IIR filter or 'fir'
            for an FIR filter. Default is 'iir'.
        axis : int, optional
            The axis along which to decimate the data. Default is 0.
        zero_phase : bool, optional
            If True, applies a zero-phase filter, which has no phase distortion.
            If False, uses a causal filter with some phase distortion. Default is True.

        Raises
        ------
        ValueError
            If the decimation factor 'q' is not greater than 1.

        Notes
        -----
        The decimation process includes low-pass filtering to reduce aliasing.
        This method updates the instance's data and sampling frequency attributes.
        """

        self.data = decimate(self.data, q, n, ftype, axis, zero_phase)
        self.fs = self.fs / q
        self.dt = 1 / self.fs

    # method to detrend data
    def detrend_data(
        self,
        axis: int = 0,
        type: typing.Literal["linear", "constant"] = "linear",
        bp: int | npt.NDArray[np.int64] = 0,
    ):
        """
        Applies detrending to the data using the scipy.signal.detrend function.

        This method removes a linear or constant trend from the data. It's commonly used
        to remove drifts or offsets in time series data.

        Parameters
        ----------
        axis : int, optional
            The axis along which to detrend the data. Default is 0.
        type : {'linear', 'constant'}, optional
            The type of detrending: 'linear' for linear detrend, or 'constant' for just
            subtracting the mean. Default is 'linear'.
        bp : int or numpy.ndarray of int, optional
            Breakpoints where the data is split for piecewise detrending. Default is 0.

        Raises
        ------
        ValueError
            If invalid parameters are provided.

        Notes
        -----
        Detrending is a preprocessing step, often necessary for methods that assume
        stationary data. This method updates the instance's data attribute.
        """
        self.data = detrend(self.data, axis, type, bp)

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

        Raises
        ------
        AssertionError
            If the number of sensors does not match between data, coordinates, and directions.
        """
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
        """
        Defines the second geometry setup (Geo2) for the instance.

        This method sets up an alternative geometry configuration, including sensors' names,
        points' coordinates, mapping, sign data, and optional elements like order reduction,
        sensor lines, background nodes, lines, and surfaces.

        Parameters
        ----------
        sens_names : Union[numpy.ndarray of string, List of string]
            An array or list containing the names of the sensors.
        pts_coord : pandas.DataFrame
            A DataFrame containing the coordinates of the points.
        sens_map : pandas.DataFrame
            A DataFrame containing the mapping data for sensors.
        sens_sign : pandas.DataFrame
            A DataFrame containing sign data for the sensors.
        order_red : {'xy', 'xz', 'yz', 'x', 'y', 'z'}, optional
            Specifies the order reduction if any. Default is None.
        sens_lines : numpy.ndarray of int64, optional
            An array defining lines connecting sensors. Default is None.
        bg_nodes : numpy.ndarray of float64, optional
            An array defining background nodes. Default is None.
        bg_lines : numpy.ndarray of float64, optional
            An array defining background lines. Default is None.
        bg_surf : numpy.ndarray of float64, optional
            An array defining background surfaces. Default is None.

        Raises
        ------
        AssertionError
            If the number of columns in mapping and sign data does not match the expected
            dimensions based on the order reduction.
        """
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
# FIXME add references!
class MultiSetup_PoSER:
    """
    A class for operational modal analysis (OMA) of multi-setup experiments using the
    Post Separate Estimation Re-scaling (PoSER) approach. It allows for merging results from
    multiple setups and analyzing the combined data to extract modal properties.

    Attributes
    ----------
    __result : Optional[Dict[str, MsPoserResult]]
        Private attribute storing the merged results from multiple setups.
    __alg_ref : Optional[Dict[type[BaseAlgorithm], str]]
        Private attribute storing references to the algorithms used in the setups.

    Methods
    -------
    merge_results()
        Merges the results from individual setups into a combined result.
    plot_mode_g1(...)
        Plots mode shapes for a specified mode number using Geometry 1 setup.
    plot_mode_g2(...)
        Plots mode shapes for a specified mode number using Geometry 2 setup.
    anim_mode_g2(...)
        Creates an animation for mode shapes for a specified mode number using Geometry 2 setup.
    def_geo1(...)
        Defines the first geometry setup (Geo1) for the instance.
    def_geo2(...)
        Defines the second geometry setup (Geo2) for the instance.
    plot_geo1(...)
        Plots the geometry (type 1) of the tested structure.
    plot_geo2(...)
        Plots the geometry (type 2) of the tested structure.
    """

    __result: typing.Optional[typing.Dict[str, MsPoserResult]] = None
    __alg_ref: typing.Optional[typing.Dict[type[BaseAlgorithm], str]] = None

    def __init__(
        self,
        ref_ind: typing.List[typing.List[int]],
        single_setups: typing.List[SingleSetup],  # | None = None,
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
        self._setups = (
            [el for el in self._init_setups(single_setups)] if single_setups else []
        )
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
        sens_names: typing.List[typing.List[str]],  # sensors' names MS
        sens_coord: pd.DataFrame,  # sensors' coordinates
        sens_dir: npt.NDArray[np.int64],  # sensors' directions
        # # OPTIONAL
        sens_lines: npt.NDArray[np.int64] = None,  # lines connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None,  # Background nodes
        bg_lines: npt.NDArray[np.int64] = None,  # Background lines
        bg_surf: npt.NDArray[np.float64] = None,  # Background surfaces
    ):
        """
        Defines the first geometry setup (Geo1) for the instance, integrating sensors' names,
        coordinates, and directions, along with optional elements like sensor lines, background
        nodes, lines, and surfaces.

        Parameters
        ----------
        sens_names : List[List[str]]
            A nested list containing the names of the sensors for each setup.
        sens_coord : pd.DataFrame
            A DataFrame containing the coordinates of the sensors. Columns should include 'x', 'y', and 'z'.
        sens_dir : np.ndarray
            A NumPy array defining the directions of the sensors. Shape: (number of sensors, 3).
        sens_lines : Optional[np.ndarray], optional
            An array defining lines connecting sensors, by default None.
        bg_nodes : Optional[np.ndarray], optional
            An array defining background nodes for additional context, by default None.
        bg_lines : Optional[np.ndarray], optional
            An array defining background lines to connect nodes, by default None.
        bg_surf : Optional[np.ndarray], optional
            An array defining background surfaces, useful for visual context, by default None.

        Raises
        ------
        AssertionError
            If the number of sensors does not match between data, coordinates, and directions.

        Notes
        -----
        The method performs various checks to ensure the integrity and consistency of the input data.
        Adapts to zero-indexing for background lines if provided.
        Reorders sensor coordinates and directions to match the provided sensor names.
        """

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
        """
        Plots the geometry of the tested structure based on the first geometry setup (Geo1).

        This method visualizes the geometry including sensor placements and directions, offering
        customizable plot parameters such as scaling factor, view type, and options to remove
        fill, grid, and axis from the plot.

        Parameters
        ----------
        scaleF : int, optional
            The scaling factor for the sensor direction quivers, by default 1.
        view : Literal["3D", "xy", "xz", "yz", "x", "y", "z"], optional
            The type of view for plotting the geometry (3D or 2D projections), by default "3D".
        remove_fill : bool, optional
            If True, removes the fill from the plot, by default True.
        remove_grid : bool, optional
            If True, removes the grid from the plot, by default True.
        remove_axis : bool, optional
            If True, removes the axis labels and ticks from the plot, by default True.

        Returns
        -------
        tuple
            A tuple containing the figure and axis objects of the plot for further customization.
        """
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Plot of the geometry and sensors' placement and direction")
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
        order_red : Optional[Literal["xy", "xz", "yz", "x", "y", "z"]], optional
            Specifies the order reduction if any, by default None.
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
        Performs checks to ensure consistency and correctness of input data based on the order reduction.
        Adapts to zero-indexing for sensor and background lines if provided."""
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

    # metodo per plottare geometria 2
    def plot_geo2(
        self,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
    ):
        """
        Plots the geometry of the tested structure based on the second geometry setup (Geo2).

        This method allows for visualization of a more complex geometric configuration of the
        structure, with customizable plot parameters.

        Parameters
        ----------
        scaleF : int, optional
            Scaling factor for quiver plots representing sensors' directions, by default 1.
        view : Literal["3D", "xy", "xz", "yz", "x", "y", "z"], optional
            Type of view for the plot (3D or 2D projections), by default "3D".
        remove_fill : bool, optional
            If True, removes the plot's fill, by default True.
        remove_grid : bool, optional
            If True, removes the plot's grid, by default True.
        remove_axis : bool, optional
            If True, removes the plot's axes, by default True.

        Returns
        -------
        tuple
            A tuple containing the figure and axis objects of the matplotlib plot.
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Plot of the geometry and sensors' placement and direction")
        # plot sensors'
        pts = self.Geo2.pts_coord.to_numpy()[:, 1:]
        plt_nodes(ax, pts, color="red")

        # plot sensors' directions
        ch_names = self.Geo2.sens_map.to_numpy()[:, 1:]
        s_sign = self.Geo2.sens_sign.to_numpy()[:, 1:]  # array of signs
        # N.B. the size of s_sign will vary depending on the order_red
        # parameter. order_red="None" size(npts,3);
        # order_red="xy/xz/yz" size(npts,2);
        # order_red="x/y/z" size(npts,1)
        # (same for ch_names)
        ord_red = self.Geo2.order_red
        zero1 = np.zeros(s_sign.shape[0]).reshape(-1, 1)
        zero2 = np.zeros((s_sign.shape[0], 2))
        if ord_red is None:
            pass
        elif ord_red == "xy":
            s_sign = np.hstack((s_sign, zero1))
            ch_names = np.hstack((ch_names, zero1))
        elif ord_red == "xz":
            s_sign = np.insert(s_sign, 1, 0)
            ch_names = np.insert(ch_names, 1, 0)
        elif ord_red == "yz":
            s_sign = np.hstack((zero1, s_sign))
            ch_names = np.hstack((zero1, ch_names))
        elif ord_red == "x":
            s_sign = np.hstack((s_sign, zero2))
            ch_names = np.hstack((ch_names, zero2))
        elif ord_red == "y":
            s_sign = np.insert(zero2, 1, s_sign)
            ch_names = np.insert(zero2, 1, ch_names)
        elif ord_red == "z":
            s_sign = np.hstack((zero2, s_sign))
            ch_names = np.hstack((zero2, ch_names))

        s_sign[s_sign == 0] = np.nan
        ch_names[ch_names == 0] = np.nan
        for ii in range(3):
            s_sign1 = np.hstack((s_sign[:, 0].reshape(-1, 1), zero2))
            s_sign2 = np.insert(zero2, 1, s_sign[:, 1], axis=1)
            s_sign3 = np.hstack((zero2, s_sign[:, 2].reshape(-1, 1)))

            plt_quiver(
                ax,
                pts,
                s_sign1,
                scaleF=scaleF,
                names=ch_names[:, 0],
            )
            plt_quiver(
                ax,
                pts,
                s_sign2,
                scaleF=scaleF,
                names=ch_names[:, 1],
            )
            plt_quiver(
                ax,
                pts,
                s_sign3,
                scaleF=scaleF,
                names=ch_names[:, 2],
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
            plt_lines(ax, pts, self.Geo2.sens_lines, color="red")

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

        Notes
        -----
        This method visualizes the mode shapes based on the provided geometry and algorithm results.
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

        Notes
        -----
        This method provides a visual representation of the mode shapes using geometry 2 setup.
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

        Notes
        -----
        The animation provides a dynamic representation of the mode shapes, enhancing the
        understanding of structural behavior under different modes.
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
    A class to conduct OMA of multi-setup experiments, with the
    "Pre Global Estimation Re-scaling" (PreGER) approach. (add reference)

    This class is designed to handle multiple datasets and apply the
    pre-global estimation re-scaling method. It allows plotting of data,
    channel information, and geometric configurations, and provides methods
    for data decimation and detrending.

    Attributes
    ----------
    fs : float
        The sampling frequency, assumed to be the same for all datasets.
    dt : float
        The sampling interval, calculated as the inverse of the sampling frequency.
    ref_ind : list[list[int]]
        A list of lists indicating reference indices for each dataset.
    datasets : list[npt.NDArray[np.float64]]
        A list of NumPy arrays, each representing a dataset.
    data : npt.NDArray[np.float64]
        The processed data after applying the PreGER method.
    algorithms : typing.Dict[str, BaseAlgorithm]
        A dictionary to store algorithms associated with the setup.

    Methods, inherited from BaseSetup
    -------
    add_algorithms(*algorithms)
        Adds algorithms to the setup and sets the data and sampling frequency for them.
    run_all()
        Runs all the algorithms added to the class.
    run_by_name(name)
        Executes a specific algorithm by its name.
    MPE(name, *args, **kwargs)
        Extracts modal parameters from selected poles/peaks.
    MPE_fromPlot(name, *args, **kwargs)
        Extracts modal parameters directly from plot selections.
    plot_geo1(scaleF, view, remove_fill, remove_grid, remove_axis)
        Plots the first type of geometry setup for the structure.
    plot_geo2(scaleF, view, remove_fill, remove_grid, remove_axis)
        Plots the second type of geometry setup for the structure.

    Methods
    -------
    plot_data(...)
        Plots the time histories of the data channels for selected datasets.
    plot_ch_info(...)
        Plots Time History (TH), Power Spectral Density (PSD), and Kernel Density Estimation (KDE)
        for each channel.
    decimate_data(...)
        Applies decimation to the data using a wrapper method for scipy.signal.decimate function.
    detrend_data(...)
        Applies detrending to the data using a wrapper method for scipy.signal.detrend function.
    def_geo1(...)
        Defines the first geometry setup (Geo1) for the instance.
    def_geo2(...)
        Defines the second geometry setup (Geo2) for the instance.

    Notes
    -----
    - The class inherits from `BaseSetup`, which provide foundational attributes and methods.
    - The `ref_ind` attribute determines how datasets are merged and scaled.
    - The `plot_data` and `plot_ch_info` methods allow visualization of the datasets'
      time history and channel information.
    - The `decimate_data` and `detrend_data` methods provide preprocessing capabilities.
    - The `def_geo1` and `def_geo2` methods allow setting up geometric configurations for the tested
        structure.

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
        self.fs = fs
        self.dt = 1 / fs
        self.ref_ind = ref_ind
        Y = PRE_MultiSetup(datasets, ref_ind)
        self.data = Y
        self.algorithms: typing.Dict[str, BaseAlgorithm] = {}  # set of algo
        self.datasets = datasets

    # method to plot the time histories of the data channels.
    def plot_data(
        self,
        data_idx: str | list[int] = "all",
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

        Notes
        -----
        - The method uses `plt_data` function for plotting.
        - The method can handle multiple datasets and plot them separately.
        """
        if data_idx != "all":
            datasets = [self.datasets[i] for i in data_idx]
        else:
            datasets = self.datasets

        dt = self.dt
        figs, axs = [], []
        for ii, data in enumerate(datasets):
            nc = nc  # number of columns for subplot
            if names is not None:
                nam = names[ii]  # list of names (str) of the channnels
            else:
                nam = None
            unit = unit  # str label for the y-axis (unit of measurement)
            show_rms = show_rms  # wheter to show or not the rms acc in the plot
            fig, ax = plt_data(data, dt, nc, nam, unit, show_rms)
            figs.append(fig)
            axs.append(ax)
        return figs, axs

    # method to plot TH, PSD and KDE for each channel
    def plot_ch_info(
        self,
        data_idx: str | list[int] = "all",
        ch_idx: str | list[int] = "all",
        ch_names: typing.Optional[typing.List[str]] = None,
        freqlim: tuple[float, float] | None = None,
        logscale: bool = True,
        nxseg: float | None = None,
        pov: float = 0.0,
        window: str = "boxcar",
    ):
        """
        Plots Time History (TH), Power Spectral Density (PSD), and
        Kernel Density Estimation (KDE) for each channel.

        This method generates plots for TH, PSD, and KDE for the specified channels across
        multiple datasets. Allows configuration of frequency limits, log scale, segments for
        PSD calculation, overlap percentage, and windowing function.

        Parameters
        ----------
        data_idx : str | list[int], optional
            Indices of datasets to be plotted. Can be 'all' or a list of indices. Default is 'all'.
        ch_idx : str | list[int], optional
            Channel indices to be plotted. Can be 'all' or a list of indices. Default is 'all'.
        ch_names : typing.Optional[typing.List[str]], optional
            Channel names for labeling purposes. Default is None.
        freqlim : float | None, optional
            Frequency limit for the plots. Default is None.
        logscale : bool, optional
            If True, plots PSD in logarithmic scale. Default is True.
        nxseg : float | None, optional
            Number of segments for Welch's method in PSD calculation. Default is None.
        pov : float, optional
            Percentage of overlap for segments in PSD calculation. Default is 0.
        window : str, optional
            Windowing function used in PSD calculation. Default is 'boxcar'.

        Returns
        -------
        list
            A list of tuples, each containing the figure and axes objects for the plots of each dataset.

        Notes
        -----
        - Utilizes `plt_ch_info` function for plotting.
        - Capable of handling and visualizing multiple datasets separately.
        """
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
                ch_idx,
                ch_names=ch_names,
                freqlim=freqlim,
                logscale=logscale,
                nxseg=nxseg,
                pov=pov,
                window=window,
            )
            figs.append(fig)
            axs.append(ax)
        return figs, axs

    # method to decimate data
    def decimate_data(
        self,
        q: int,
        n: int | None = None,
        ftype: typing.Literal["iir", "fir"] = "iir",
        axis: int = 0,
        zero_phase: bool = True,
    ):
        """
        wrapper method for scipy.signal.decimate function"""
        datasets = self.datasets
        newdatasets = []
        for data in datasets:
            newdata = decimate(data, q, n, ftype, axis, zero_phase)
            newdatasets.append(newdata)

        Y = PRE_MultiSetup(newdatasets, self.ref_ind)
        self.data = Y
        self.fs = self.fs / q
        self.dt = 1 / self.fs

    # method to detrend data
    def detrend_data(
        self,
        axis: int = 0,
        type: typing.Literal["linear", "constant"] = "linear",
        bp: int | npt.NDArray[np.int64] = 0,
    ):
        """
        wrapper method for scipy.signal.detrend function"""
        datasets = self.datasets
        newdatasets = []
        for data in datasets:
            newdata = detrend(data, axis, type, bp)
            newdatasets.append(newdata)

        Y = PRE_MultiSetup(newdatasets, self.ref_ind)
        self.data = Y

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
        """
        Defines the first geometry setup (Geo1) for the instance.

        This method sets up the geometry involving sensors' names, coordinates, directions,
        and optional elements like sensor lines, background nodes, lines, and surfaces.

        Parameters
        ----------
        sens_names : typing.Union[npt.NDArray[np.string], typing.List[str]]
            An array or list containing the names of the sensors.
        sens_coord : pd.DataFrame
            A pandas DataFrame containing the coordinates of the sensors.
        sens_dir : npt.NDArray[np.int64]
            An array defining the directions of the sensors.
        sens_lines : npt.NDArray[np.int64], optional
            An array defining lines connecting sensors. Default is None.
        bg_nodes : npt.NDArray[np.float64], optional
            An array defining background nodes. Default is None.
        bg_lines : npt.NDArray[np.int64], optional
            An array defining background lines. Default is None.
        bg_surf : npt.NDArray[np.float64], optional
            An array defining background surfaces. Default is None.

        Raises
        ------
        AssertionError
            If the number of sensors does not match between data, coordinates, and directions.

        Notes
        -----
        - The method performs various checks to ensure the integrity and consistency of the input data.
        - Adapts to zero-indexing for background lines if provided.
        - Reorders sensor coordinates and directions to match the provided sensor names."""

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
        order_red: typing.Literal["xy", "xz", "yz", "x", "y", "z"] = None,
        sens_lines: npt.NDArray[np.int64] = None,  # lines connecting sensors
        bg_nodes: npt.NDArray[np.float64] = None,  # Background nodes
        bg_lines: npt.NDArray[np.float64] = None,  # Background lines
        bg_surf: npt.NDArray[np.float64] = None,  # Background lines
    ):
        """
        Defines the second geometry setup (Geo2) for the instance.

        This method sets up an alternative geometry configuration, including sensors' names,
        points' coordinates, mapping, sign data, and optional elements like order reduction,
        sensor lines, background nodes, lines, and surfaces.

        Parameters
        ----------
        sens_names : typing.Union[npt.NDArray[np.string], typing.List[str]]
            An array or list containing the names of the sensors.
        pts_coord : pd.DataFrame
            A DataFrame containing the coordinates of the points.
        sens_map : pd.DataFrame
            A DataFrame containing the mapping data for sensors.
        sens_sign : pd.DataFrame
            A DataFrame containing sign data for the sensors.
        order_red : typing.Optional[typing.Literal["xy", "xz", "yz", "x", "y", "z"]], optional
            Specifies the order reduction if any. Default is None.
        sens_lines : npt.NDArray[np.int64], optional
            An array defining lines connecting sensors. Default is None.
        bg_nodes : npt.NDArray[np.float64], optional
            An array defining background nodes. Default is None.
        bg_lines : npt.NDArray[np.float64], optional
            An array defining background lines. Default is None.
        bg_surf : npt.NDArray[np.float64], optional
            An array defining background surfaces. Default is None.

        Raises
        ------
        AssertionError
            If the number of columns in mapping and sign data does not match the expected
            dimensions based on the order reduction.

        Notes
        -----
        - Performs checks to ensure consistency and correctness of input data based on the order reduction.
        - Adapts to zero-indexing for sensor and background lines if provided."""
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
