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
import numpy.typing as npt

from pyoma2.algorithms.data.result import MsPoserResult
from pyoma2.functions.gen import (
    merge_mode_shapes,
    pre_multisetup,
)
from pyoma2.functions.plot import (
    STFT,
    plt_ch_info,
    plt_data,
)
from pyoma2.setup.base import BaseSetup
from pyoma2.setup.single import SingleSetup
from pyoma2.support.geometry import (
    GeometryMixin,
)

if typing.TYPE_CHECKING:
    from pyoma2.algorithms import BaseAlgorithm


logger = logging.getLogger(__name__)

# =============================================================================
# POSER
# =============================================================================


class MultiSetup_PoSER(GeometryMixin):
    """
    Class for conducting Operational Modal Analysis (OMA) on multi-setup experiments using
    the Post Separate Estimation Re-scaling (PoSER) approach. This approach is designed to
    merge and analyze data from multiple experimental setups for operational modal analysis.

    Attributes
    ----------
    ref_ind : List[List[int]]
        Indices of reference sensors for each dataset, as a list of lists.
    setups : List[SingleSetup]
        A list of SingleSetup instances representing individual measurement setups.
    names : List[str]
        A list of names for the algorithms used in each setup. Used to retrieve results.
    __result : Optional[Dict[str, MsPoserResult]]
        Private attribute to store the merged results from multiple setups. Each entry in the
        dictionary corresponds to a specific algorithm used across setups, with its results.
    Warning
    -------
    The PoSER approach assumes that the setups used are compatible in terms of their experimental
    setup and data characteristics.
    """

    __result: typing.Optional[typing.Dict[str, MsPoserResult]] = None

    def __init__(
        self,
        ref_ind: typing.List[typing.List[int]],
        single_setups: typing.List[SingleSetup],
        names: typing.List[str],
    ):
        """
        Initializes the MultiSetup_PoSER instance with reference indices and a list of SingleSetup instances.

        Parameters
        ----------
        ref_ind : List[List[int]]
            Reference indices for merging results from different setups.
        single_setups : List[SingleSetup]
            A list of SingleSetup instances to be merged using the PoSER approach.
        names : List[str]
            A list of names for the algorithms used in each setup. Used to retrieve results.
            Te list must be len as the number of algorithms in each setup.

        Raises
        ------
        ValueError
            If any of the provided setups are invalid or incompatible.
        """
        self.names = names
        self._setups = [
            el for el in self._init_setups(setups=single_setups if single_setups else [])
        ]
        self.ref_ind = ref_ind
        self.__result = None

    @property
    def setups(self) -> typing.List[SingleSetup]:
        """
        Returns the list of SingleSetup instances representing individual measurement setups.
        """
        return self._setups

    @setups.setter
    def setups(self, setups: typing.List[SingleSetup]) -> None:
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
        if len(setups) <= 1:
            raise ValueError("You must pass at least two setup")
        if any(not setup.algorithms for setup in setups):
            raise ValueError("You must pass setups with at least one algorithm")

        algo_instances = [setup.algorithms.values() for setup in setups]

        # The following validation ensures that each list of algorithms has the same set of algorithm classes.
        # This means that the order and presence of each class must be consistent across all lists.
        # For example:
        # [[fdd, ssi], [fdd, ssi], [fdd, ssi]] is valid (same order and presence)
        # [[fdd, fdd], [fdd, fdd], [fdd, fdd]] is valid (same order and presence)
        # [[fdd, ssi], [fdd, ssi], [fdd, fdd]] is not valid (different presence in the last list)
        # [[fdd, ssi], [fdd, ssi], [ssi, fdd]] is not valid (different order in the last list)
        # [[fdd, ssi], [fdd, ], [ssi, fdd]] is not valid (different presence in the second list)
        if not all(
            [type(algo) for algo in algos] == [type(algo) for algo in algo_instances[0]]
            for algos in algo_instances
        ):
            raise ValueError("The algorithms must be consistent between setups")

        if len(self.names) != len(setups[0].algorithms):
            raise ValueError("The number of names must match the number of algorithms")

        for i, setup in enumerate(setups):
            logger.debug("Initializing %s/%s setups", i + 1, len(setups))
            for alg in setup.algorithms.values():
                if not alg.result or alg.result.Fn is None:
                    raise ValueError(
                        "You must pass Single setups that have already been run"
                        " and the Modal Parameters have to be extracted (call mpe method on SingleSetup)"
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
        alg_groups: typing.Dict[str, typing.List[BaseAlgorithm]] = {}
        for setup in self.setups:
            for ii, alg in enumerate(setup.algorithms.values()):
                alg_groups.setdefault(self.names[ii], []).append(alg)

        for alg_name, algs in alg_groups.items():
            alg_cl = algs[0].__class__
            logger.info("Merging %s results for %s group", alg_cl.__name__, alg_name)
            # get the reference algorithm
            all_fn = []
            all_xi = []
            all_phi = []
            for alg in algs:
                logger.info("Merging %s results", alg.name)
                all_fn.append(alg.result.Fn)
                all_xi.append(alg.result.Xi)
                all_phi.append(alg.result.Phi)

            # Convert lists to numpy arrays
            all_fn = np.array(all_fn)
            all_xi = np.array(all_xi)

            # Calculate mean and covariance
            fn_mean = np.mean(all_fn, axis=0)
            xi_mean = np.mean(all_xi, axis=0)

            fn_std = np.std(all_fn, axis=0)
            xi_std = np.std(all_xi, axis=0)
            Phi = merge_mode_shapes(MSarr_list=all_phi, reflist=self.ref_ind)

            if self.__result is None:
                self.__result = {}

            self.__result[alg_name] = MsPoserResult(
                Phi=Phi,
                Fn=fn_mean,
                Fn_std=fn_std,
                Xi=xi_mean,
                Xi_std=xi_std,
            )
        return self.__result


# =============================================================================
#
# =============================================================================


class MultiSetup_PreGER(BaseSetup, GeometryMixin):
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

    Warning
    -------
    The PreGER approach assumes that the setups used are compatible in terms of their experimental
    setup and data characteristics.
    """

    dt: float
    Nsetup: int
    data: typing.List[typing.Dict[str, np.ndarray]]
    algorithms: typing.Dict[str, BaseAlgorithm]
    Nchs: typing.List[int]
    Ndats: typing.List[int]
    Ts: typing.List[float]

    def __init__(
        self,
        fs: float,
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
        self.ref_ind = ref_ind  # list of (list of) reference indices
        self.datasets = datasets

        self._initialize_data(fs=fs, ref_ind=ref_ind, datasets=datasets)

    def _initialize_data(
        self,
        fs: float,
        ref_ind: typing.List[typing.List[int]],
        datasets: typing.List[npt.NDArray[np.float64]],
    ) -> None:
        """
        Pre process the data and set the initial attributes after copying the data.

        This method is called internally to pre-process the data and set the initial attributes
        """
        # Store a copy of the initial data
        self._initial_fs = fs
        self._initial_ref_ind = copy.deepcopy(ref_ind)
        self._initial_datasets = copy.deepcopy(datasets)

        self.dt = 1 / fs  # sampling interval
        self.Nsetup = len(ref_ind)

        # Pre-process the data so to be multi-setup compatible
        Y = pre_multisetup(datasets, ref_ind)

        self.data = Y
        self.algorithms: typing.Dict[str, BaseAlgorithm] = {}  # set of algo
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

    def rollback(self) -> None:
        """
        Restores the data and sampling frequency to their initial state.

        This method reverts the `data` and `fs` attributes to their original values, effectively
        undoing any operations that modify the data, such as filtering, detrending, or decimation.
        It can be used to reset the setup to the state it was in after instantiation.
        """
        self.fs = self._initial_fs
        self.ref_ind = self._initial_ref_ind
        self.datasets = self._initial_datasets

        self._initialize_data(
            fs=self._initial_fs,
            ref_ind=self._initial_ref_ind,
            datasets=self._initial_datasets,
        )

    # method to plot the time histories of the data channels.
    def plot_data(
        self,
        data_idx: typing.Union[str, typing.List[int]] = "all",
        nc: int = 1,
        names: typing.Optional[typing.List[str]] = None,
        # names: list[list[str]] = None,
        unit: str = "unit",
        show_rms: bool = False,
    ) -> typing.Tuple[plt.Figure, plt.Axes]:
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
    ) -> typing.Tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """
        Plot channel information including time history, normalized auto-correlation,
        power spectral density (PSD), probability density function, and normal probability
        plot for each channel in the selected datasets.

        Parameters
        ----------
        data_idx : str or list[int], optional
            Indices of the datasets to plot. Use 'all' to plot data for all datasets. Default is 'all'.
        nxseg : float, optional
            The number of data points per segment for the PSD estimation. Default is 1024.
        ch_idx : str or list[int], optional
            Indices of the channels to plot. Use 'all' to plot information for all channels. Default is 'all'.
        freqlim : tuple of float, optional
            Frequency limits (min, max) for the PSD plot. Default is None, using the full range.
        logscale : bool, optional
            Whether to use a logarithmic scale for the PSD plot. Default is True.
        unit : str, optional
            Unit of measurement for the data, used in labeling the plots. Default is 'unit'.

        Returns
        -------
        tuple
            A tuple containing lists of figure and axes objects for further customization or saving.
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
    ) -> typing.Tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """
        Plot the Short-Time Fourier Transform (STFT) magnitude spectrogram for the specified channels.

        This method computes and plots the STFT magnitude spectrogram for each selected channel in the
        specified datasets. The spectrogram is plotted as a heatmap where the x-axis represents time, the y-axis
        represents frequency, and the color intensity represents the magnitude of the STFT.

        Parameters
        ----------
        data_idx : str or list[int], optional
            Indices of the datasets to plot. Use 'all' to plot data for all datasets. Default is 'all'.
        nxseg : float, optional
            The number of data points per segment for the STFT calculation. Default is 256.
        pov : float, optional
            Proportion of overlap between consecutive segments, expressed as a decimal between 0 and 1.
            Default is 0.9 (90% overlap).
        ch_idx : str or list[int], optional
            Indices of the channels to plot. Use 'all' to plot information for all channels. Default is 'all'.
        freqlim : tuple of float, optional
            Frequency limits (min, max) for the STFT plot. Default is None, using the full range.
        win : str, optional
            The windowing function to apply to each segment. Default is 'hann'.

        Returns
        -------
        tuple
            A tuple containing lists of figure and axes objects for further customization or saving.
        """
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
        **kwargs: typing.Any,
    ) -> None:
        """
        Applies decimation to the data using the scipy.signal.decimate function.

        This method reduces the sampling rate of the data by a factor of 'q'.
        The decimation process includes low-pass filtering to reduce aliasing.
        The method updates the instance's data and sampling frequency attributes.

        Parameters
        ----------
        q : int
            The decimation factor. Must be greater than 1.
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

        Notes
        -----
        For further information, see `scipy.signal.decimate
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html>`_.
        """
        n = kwargs.get("n")
        ftype = kwargs.get("ftype", "iir")
        axis = kwargs.get("axis", 0)
        zero_phase = kwargs.get("zero_phase", True)

        newdatasets = []
        Ndats = []
        Ts = []
        for data in self.datasets:
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

        Y = pre_multisetup(newdatasets, self.ref_ind)
        fs = self.fs / q
        dt = 1 / self.fs

        self.datasets = newdatasets
        self.data = Y
        self.fs = fs
        self.dt = dt
        self.Ndats = Ndats
        self.Ts = Ts

    # method to filter data
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
        newdatasets = []
        for data in self.datasets:
            newdata = super()._filter_data(
                data=data,
                fs=self.fs,
                Wn=Wn,
                order=order,
                btype=btype,
            )
            newdatasets.append(newdata)

        Y = pre_multisetup(newdatasets, self.ref_ind)
        self.data = Y

    # method to detrend data
    def detrend_data(
        self,
        **kwargs: typing.Any,
    ) -> None:
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
        newdatasets = []
        for data in self.datasets:
            newdata = super()._detrend_data(data=data, **kwargs)
            newdatasets.append(newdata)

        Y = pre_multisetup(newdatasets, self.ref_ind)
        self.data = Y
