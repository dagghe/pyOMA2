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

import matplotlib.pyplot as plt
import numpy as np

from pyoma2.functions import plot
from pyoma2.setup.base import BaseSetup
from pyoma2.support.geometry import GeometryMixin

if typing.TYPE_CHECKING:
    from pyoma2.algorithms import BaseAlgorithm


logger = logging.getLogger(__name__)


class SingleSetup(BaseSetup, GeometryMixin):
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

    Notes
    -----
    The ``algorithms`` dictionary is initialized empty and is meant to store various algorithms as needed.
    """

    dt: float
    Nch: int
    Ndat: int
    T: float
    algorithms: typing.Dict[str, BaseAlgorithm]

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

        self._initialize_data(data=data, fs=fs)

    def _initialize_data(self, data: np.ndarray, fs: float) -> None:
        """
        Pre process the data and set the initial attributes after copying the data.

        This method is called during the initialization of the SingleSetup instance.
        """
        # Store a copy of the initial data
        self._initial_data = copy.deepcopy(data)
        self._initial_fs = fs

        self.dt = 1 / fs  # sampling interval
        self.Nch = data.shape[1]  # number of channels
        self.Ndat = data.shape[0]  # number of data points
        self.T = self.dt * self.Ndat  # Period of acquisition [sec]

        self.algorithms: typing.Dict[str, BaseAlgorithm] = {}  # set of algo

    def rollback(self) -> None:
        """
        Restores the data and sampling frequency to their initial state.

        This method reverts the `data` and `fs` attributes to their original values, effectively
        undoing any operations that modify the data, such as filtering, detrending, or decimation.
        It can be used to reset the setup to the state it was in after instantiation.
        """
        self.data = self._initial_data
        self.fs = self._initial_fs

        self._initialize_data(data=self._initial_data, fs=self._initial_fs)

    # method to plot the time histories of the data channels.
    def plot_data(
        self,
        nc: int = 1,
        names: typing.Optional[typing.List[str]] = None,
        unit: str = "unit",
        show_rms: bool = False,
    ) -> typing.Tuple[plt.Figure, np.ndarray]:
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
        fig, ax = plot.plt_data(data, fs, nc, names, unit, show_rms)
        return fig, ax

    # Method to plot info on channel (TH,auto-corr,PSD,PDF,dataVSgauss)
    def plot_ch_info(
        self,
        nxseg: float = 1024,
        ch_idx: typing.Union[str, typing.List[int]] = "all",
        freqlim: typing.Optional[tuple[float, float]] = None,
        logscale: bool = True,
        unit: str = "unit",
    ) -> typing.Tuple[plt.Figure, np.ndarray]:
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
        """
        data = self.data
        fs = self.fs
        fig, ax = plot.plt_ch_info(
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
    ) -> typing.Tuple[plt.Figure, np.ndarray]:
        """
        Plot the Short-Time Fourier Transform (STFT) magnitude spectrogram for the specified channels.

        This method computes and plots the STFT magnitude spectrogram for each selected channel in the
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
        """

        data = self.data
        fs = self.fs
        fig, ax = plot.STFT(
            data,
            fs,
            nxseg=nxseg,
            pov=pov,
            win=win,
            ch_idx=ch_idx,
            freqlim=freqlim,
        )
        return fig, ax

    def decimate_data(self, q: int, **kwargs) -> None:
        """
        Decimates the data using the scipy.signal.decimate function.

        This method reduces the sampling rate of the data by a factor of 'q'.
        The decimation process includes low-pass filtering to reduce aliasing.
        The method updates the instance's data and sampling frequency attributes.

        Parameters
        ----------
        q : int
            The decimation factor. Must be greater than 1.
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

        Notes
        -----
        For further information, see `scipy.signal.decimate
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html>`_.
        """
        axis = kwargs.pop("axis", 0)
        decimated_data, fs, dt, Ndat, T = super()._decimate_data(
            data=self.data, fs=self.fs, q=q, axis=axis, **kwargs
        )
        self.data = decimated_data
        self.fs = fs
        self.dt = dt
        self.Ndat = Ndat
        self.T = T

    def detrend_data(self, **kwargs) -> None:
        """
        Detrends the data using the scipy.signal.detrend function.

        This method removes a linear or constant trend from the data, commonly used to remove drifts
        or offsets in time series data. It's a preprocessing step, often necessary for methods that
        assume stationary data. The method updates the instance's data attribute.

        Parameters
        ----------
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

        Notes
        -----
        For further information, see `scipy.signal.detrend
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html>`_.
        """
        detrended_data = super()._detrend_data(data=self.data, **kwargs)
        self.data = detrended_data

    def filter_data(
        self,
        Wn: typing.Union[float, typing.Tuple[float, float]],
        order: int = 8,
        btype: str = "lowpass",
    ) -> None:
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

        Notes
        -----
        For more information, see the scipy documentation for `signal.butter`
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
        and `signal.sosfiltfilt`
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
        """
        filt_data = super()._filter_data(
            data=self.data,
            fs=self.fs,
            Wn=Wn,
            order=order,
            btype=btype,
        )
        self.data = filt_data
