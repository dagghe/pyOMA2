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
import typing

import numpy as np
from scipy.signal import decimate, detrend

from pyoma2.functions.gen import (
    filter_data,
)

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

    Warning
    -------
    The BaseSetup class is not intended for direct instantiation by users.
    It acts as a common interface for handling different types of setup configurations.
    Specific functionalities are provided through its subclasses.
    """

    algorithms: typing.Dict[str, BaseAlgorithm]
    data: np.ndarray  # TODO use generic typing
    fs: float  # sampling frequency

    def rollback(self) -> None:
        """
        Rollback the data to the initial state.

        This method must be implemented by subclasses to provide a rollback mechanism for the data.
        Raises a `NotImplementedError` in the base class.
        """
        raise NotImplementedError("Rollback method must be implemented by subclasses.")

    # add algorithm (method) to the set.
    def add_algorithms(self, *algorithms: BaseAlgorithm) -> None:
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
    def run_all(self) -> None:
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
    def run_by_name(self, name: str) -> None:
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
    def mpe(self, name: str, *args, **kwargs) -> None:
        """
        Extracts modal parameters from selected poles/peaks of a specified algorithm.

        Parameters
        ----------
        name : str
            Name of the algorithm from which to extract modal parameters.
        args : tuple
            Positional arguments to be passed to the algorithm's mpe method.
        kwargs : dict
            Keyword arguments to be passed to the algorithm's mpe method.

        Raises
        ------
        KeyError
            If the specified algorithm name does not exist in the setup.
        """
        logger.info("Getting mpe modal parameters from %s", name)
        self[name].mpe(*args, **kwargs)

    # get the modal properties (all results) from the plots.
    def mpe_from_plot(self, name: str, *args, **kwargs) -> None:
        """
        Extracts modal parameters directly from plot selections of a specified algorithm.

        Parameters
        ----------
        name : str
            Name of the algorithm from which to extract modal parameters.
        args : tuple
            Positional arguments to be passed to the algorithm's mpe method.
        kwargs : dict
            Keyword arguments to be passed to the algorithm's mpe method.

        Raises
        ------
        KeyError
            If the specified algorithm name does not exist in the setup.
        """
        logger.info("Getting mpe modal parameters from plot... %s", name)
        self[name].mpe_from_plot(*args, **kwargs)

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

        Notes
        -----
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

        Notes
        -----
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
        Wn : float or tuple of float
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
