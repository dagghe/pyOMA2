# -*- coding: utf-8 -*-
"""
Operational Modal Analysis Module for Single and Multi-Setup Configurations.
Part of the pyOMA2 package.
Authors:
Dag Pasca
Diego Margoni
"""

from __future__ import annotations

import logging
import pickle
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
from pyoma2.functions.Gen_funct import (
    filter_data,
    find_map,
    flatten_sns_names,
    import_excel_GEO1,
    import_excel_GEO2,
)
from pyoma2.support.geometry import Geometry1, Geometry2
from pyoma2.support.MplPlotter import MplGeoPlotter
from pyoma2.support.PyVistaPlotter import PvGeoPlotter
from scipy.signal import decimate, detrend

if typing.TYPE_CHECKING:
    from pyoma2.algorithms import BaseAlgorithm


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
        # TODO
        # assert dimensions

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
        if self.Geo1 is None:
            raise ValueError("Geo1 is not defined. Call def_geo1 first.")

        Plotter = MplGeoPlotter(self.Geo1)

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
        if self.Geo2 is None:
            raise ValueError("Geo2 is not defined. Call def_geo2 first.")

        Plotter = MplGeoPlotter(self.Geo2)

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
        if self.Geo2 is None:
            raise ValueError("Geo2 is not defined. Call def_geo2 first.")

        Plotter = PvGeoPlotter(self.Geo2)

        pl = Plotter.plot_geo(
            plot_points,
            points_sett,
        )
        return pl

    # FIXME SAVE LOAD FILES NOT WORKING
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
