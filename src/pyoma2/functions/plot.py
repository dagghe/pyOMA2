"""
Plotting Utility Functions module.
Part of the pyOMA2 package.
Author:
Dag Pasca
"""

import logging
import typing

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy import signal, stats
from scipy.interpolate import interp1d

from .gen import MAC

logger = logging.getLogger(__name__)

# =============================================================================
# PLOT ALGORITMI
# =============================================================================


def plot_dtot_hist(dtot, bins="auto", sugg_co=True):
    """
    Plot a histogram of the total distance matrix with optional suggested cut-off distances.

    This function plots a histogram of the values in the input distance matrix or vector `dtot`.
    It overlays a kernel density estimate (KDE) and optionally indicates suggested cut-off distances
    for clustering using single-linkage and average-linkage methods.

    Parameters
    ----------
    dtot : ndarray
        The input distance data. If a 2D array is provided, the upper triangular elements
        (excluding the diagonal) are extracted. If a 1D array is provided, it is used as is.
    bins : int or str, optional
        The number of bins for the histogram. Defaults to "auto".
    sugg_co : bool, optional
        Whether to compute and display suggested cut-off distances for clustering.
        Defaults to True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object for the histogram plot.
    """
    if dtot.ndim == 2:
        # Extract upper triangular indices and values
        upper_tri_indices = np.triu_indices_from(dtot, k=0)
        x = dtot[upper_tri_indices]
    elif dtot.ndim == 1:
        x = dtot

    xs = np.linspace(dtot.min(), dtot.max(), 500)

    kde = stats.gaussian_kde(x)
    # Evaluate the KDE to get the PDF
    pdf = kde(xs)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the histogram on the axis
    ax.hist(x, bins=bins, color="skyblue", edgecolor="black")
    ax1 = ax.twinx()
    ax1.plot(xs, kde(xs), "k-", label="Kernel density estimate")

    if sugg_co:
        minima_in = signal.argrelmin(pdf)[0]
        minima = pdf[minima_in]
        min_abs = minima.argmin()
        min_abs_ind = minima_in[min_abs]
        maxima_in = signal.argrelmax(pdf)
        dc2_ind = maxima_in[0][0]
        dc2 = xs[dc2_ind]
        dc1 = xs[min_abs_ind]

        ax1.axvline(
            dc2,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Suggested cut-off distance single-linkage: {dc2:.4f}",
        )
        ax1.axvline(
            dc1,
            color="red",
            linestyle="dotted",
            linewidth=2,
            label=f"Suggested cut-off distance average-linkage: {dc1:.4f}",
        )

    # Customize plot
    ax.set_xlabel("dtot")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of distances")
    ax.xaxis.set_major_locator(MultipleLocator(0.1))  # Major ticks every 0.1
    ax.xaxis.set_minor_locator(
        MultipleLocator(0.02)
    )  # Minor ticks every 1/5 of major (0.02)

    # Add grid only for the x-axis
    ax.grid(which="major", axis="x", color="gray", linestyle="-", linewidth=0.5)
    ax.grid(
        which="minor", axis="x", color="gray", linestyle=":", linewidth=0.5, alpha=0.7
    )
    ax1.legend(framealpha=1)
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------


# Helper function to adjust alpha of colors
def adjust_alpha(color, alpha):
    """
    Adjust the alpha (opacity) of a given color.

    Parameters
    ----------
    color : str or tuple
        The input color in any valid Matplotlib format (e.g., string name, hex code, or RGB tuple).
    alpha : float
        The desired alpha value, between 0 (completely transparent) and 1 (completely opaque).

    Returns
    -------
    tuple
        The RGBA representation of the input color with the specified alpha value.
    """
    rgba = to_rgba(color)
    return rgba[:3] + (alpha,)


# Rearrange legend elements for column-wise ordering
def rearrange_legend_elements(legend_elements, ncols):
    """
    Rearrange legend elements into a column-wise ordering.

    Parameters
    ----------
    legend_elements : list of matplotlib.lines.Line2D
        A list of legend elements to be rearranged.
    ncols : int
        The number of columns to arrange the legend elements into.

    Returns
    -------
    list
        A reordered list of legend elements arranged column-wise.
    """
    n = len(legend_elements)
    nrows = int(np.ceil(n / ncols))
    total_entries = nrows * ncols
    legend_elements_padded = legend_elements + [None] * (total_entries - n)
    legend_elements_array = np.array(legend_elements_padded).reshape(nrows, ncols)
    rearranged_elements = legend_elements_array.flatten(order="F")
    rearranged_elements = [elem for elem in rearranged_elements if elem is not None]
    return rearranged_elements


# -----------------------------------------------------------------------------


def freq_vs_damp_plot(
    Fn_fl: np.ndarray,
    Xi_fl: np.ndarray,
    labels: np.ndarray,
    freqlim: typing.Optional[typing.Tuple] = None,
    plot_noise: bool = False,
    name: str = None,
    fig: typing.Optional[plt.Figure] = None,
    ax: typing.Optional[plt.Axes] = None,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plot frequency versus damping, with points grouped by clusters.

    Parameters
    ----------
    Fn_fl : np.ndarray
        Array of natural frequencies (flattened, 1d).
    Xi_fl : np.ndarray
        Array of damping ratios (flattened, 1d).
    labels : np.ndarray
        Cluster labels for each data point. Use `-1` for noise.
    freqlim : tuple of float, optional
        Tuple specifying the (min, max) limits for the frequency axis, by default None.
    plot_noise : bool, optional
        Whether to include points labeled as noise (`-1`) in the plot, by default False.
    fig : plt.Figure, optional
        Existing Matplotlib figure to plot on, by default None.
    ax : plt.Axes, optional
        Existing Matplotlib axes to plot on, by default None.

    Returns
    -------
    fig : plt.Figure
        The Matplotlib figure object containing the plot.
    ax : plt.Axes
        The Matplotlib axes object containing the plot.
    """
    # Initialize figure and axes if not provided
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(
            right=0.75
        )  # Adjust the right margin to make room for the legend
    elif fig is not None and ax is None:
        ax = fig.add_subplot(1, 1, 1)
    elif fig is None and ax is not None:
        fig = ax.figure

    # Assign x and y
    x = Fn_fl
    y = Xi_fl * 100  # N.B. Transform to percent

    # Filter out noise points if plot_noise is False
    if not plot_noise:
        mask = labels != -1
        x = x[mask]
        y = y[mask]
        labels_filtered = labels[mask]

    else:
        labels_filtered = labels

    # Identify unique labels (after filtering)
    unique_labels = np.unique(labels_filtered)

    # Separate noise label
    labels_without_noise = unique_labels[unique_labels != -1]
    n_labels = len(labels_without_noise)

    # Choose a colormap with enough distinct colors, excluding greys
    if n_labels <= 9:  # Exclude grey from tab10
        cmap = plt.get_cmap("tab10")
        colors = [cmap.colors[i] for i in range(len(cmap.colors)) if i != 7]
    elif n_labels <= 18:  # Exclude greys from tab20
        cmap = plt.get_cmap("tab20")
        colors = [cmap.colors[i] for i in range(len(cmap.colors)) if i not in [14, 15]]
    else:
        # Generate a colormap with n_labels distinct colors
        cmap = plt.cm.get_cmap("hsv", n_labels)
        colors = cmap(np.linspace(0, 1, n_labels))

    # Create a mapping from label to color for clusters (excluding noise)
    color_map = {label: colors[i] for i, label in enumerate(labels_without_noise)}

    # Assign grey color to noise label
    if -1 in unique_labels:
        color_map[-1] = "grey"

    point_colors = [color_map[label] for label in labels_filtered]

    # Create masks for noise and cluster data
    noise_mask = labels_filtered == -1
    cluster_mask = labels_filtered != -1

    # Plot the scatter points for clusters
    ax.scatter(
        x[cluster_mask],
        y[cluster_mask],
        c=[point_colors[i] for i in range(len(point_colors)) if cluster_mask[i]],
        s=70,
        alpha=0.5,
        edgecolors="k",
        linewidth=0.9,
    )

    # Plot the scatter points for noise cluster
    if plot_noise and noise_mask.any():
        ax.scatter(
            x[noise_mask],
            y[noise_mask],
            c=[point_colors[i] for i in range(len(point_colors)) if noise_mask[i]],
            s=70,
            alpha=0.2,
            edgecolors="k",
            linewidth=0.5,
        )

    # Prepare legend labels
    legend_labels = {}
    cluster_counter = 1
    for label in unique_labels:
        if label == -1 and plot_noise:
            legend_labels[label] = "Noise"
        elif label == -1 and not plot_noise:
            continue  # Skip noise label
        else:
            legend_labels[label] = f"Cluster {cluster_counter}"
            cluster_counter += 1

    # Create custom legend handles with formatted labels
    legend_elements = []
    for label in unique_labels:
        if label == -1 and not plot_noise:
            continue  # Skip adding Noise to legend if plot_noise is False
        if label == -1:
            facecolor = adjust_alpha(color_map[label], 0.5)
        else:
            facecolor = adjust_alpha(color_map[label], 0.9)
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=facecolor,
                markeredgecolor="k",
                markeredgewidth=0.5,
                markersize=10,
                label=legend_labels[label],
            )
        )

    # Determine the number of columns for the legend
    ncols = 1 if len(legend_elements) <= 20 else 2

    legend_elements = rearrange_legend_elements(legend_elements, ncols)

    # Add the legend to the plot
    ax.legend(
        handles=legend_elements,
        title="Clusters",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        ncol=ncols,
        borderaxespad=0.0,
    )

    # Add cross for each cluster (excluding Noise)
    for label in labels_without_noise:
        # Extract x-values for the current cluster
        cluster_x = x[labels_filtered == label]
        cluster_y = y[labels_filtered == label]
        if len(cluster_x) == 0:
            continue  # Skip if no points in cluster
        median_x = np.median(cluster_x)
        median_y = np.median(cluster_y)

        ax.plot(
            median_x,
            median_y,
            marker="x",  # 'x' marker for cross
            markersize=12,  # Larger size than cluster circles
            markeredgewidth=2,  # Thin lines for the cross
            markeredgecolor="red",  # Red color for visibility
            linestyle="None",  # No connecting lines
        )

    # Set plot titles and labels
    ax.set_title(f"Frequency vs Damping - Clusters {name}")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Damping [%]")

    # Set x-axis limits if freqlim is provided
    if freqlim is not None:
        ax.set_xlim(freqlim[0], freqlim[1])

    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    return fig, ax


# -----------------------------------------------------------------------------


def stab_clus_plot(
    Fn_fl: np.ndarray,
    order_fl: np.ndarray,
    labels: np.ndarray,
    step: int,
    ordmax: int,
    ordmin: int = 0,
    freqlim: typing.Optional[typing.Tuple[float, float]] = None,
    Fn_std: np.ndarray = None,
    plot_noise: bool = False,
    name: str = None,
    fig: typing.Optional[plt.Figure] = None,
    ax: typing.Optional[plt.Axes] = None,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plots a stabilization chart of the poles of a system with clusters indicated by colors.
    The legend labels clusters as "Cluster 1", "Cluster 2", ..., "Cluster N", and "-1" as "Noise".
    Additionally, adds a vertical line at the median frequency of each cluster.
    Optionally, the noise cluster can be excluded from the plot.

    Parameters
    ----------
    Fn_fl : np.ndarray
        Frequency values.
    order_fl : np.ndarray
        Model order values.
    labels : np.ndarray
        Cluster labels for each point.
    step : int
        Step parameter (usage not shown in the plot).
    ordmax : int
        Maximum order for y-axis limit.
    ordmin : int, optional
        Minimum order for y-axis limit, by default 0.
    freqlim : tuple of float, optional
        Frequency limits for x-axis, by default None.
    Fn_std : np.ndarray, optional
        Standard deviation for frequency, by default None.
    fig : plt.Figure, optional
        Existing figure to plot on, by default None.
    ax : plt.Axes, optional
        Existing axes to plot on, by default None.
    plot_noise : bool, optional
        Whether to include the noise cluster in the plot, by default False.

    Returns
    -------
    fig : plt.Figure
        The matplotlib Figure object containing the plot.
    ax : plt.Axes
        The matplotlib Axes object containing the plot.
    """

    # Initialize figure and axes if not provided
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(
            right=0.75
        )  # Adjust the right margin to make room for the legend
    elif fig is not None and ax is None:
        ax = fig.add_subplot(1, 1, 1)
    elif fig is None and ax is not None:
        fig = ax.figure

    # Assign x and y
    x = Fn_fl
    y = order_fl

    # Filter out noise points if plot_noise is False
    if not plot_noise:
        mask = labels != -1
        x = x[mask]
        y = y[mask]
        labels_filtered = labels[mask]
        if Fn_std is not None:
            Fn_std = Fn_std[mask]
    else:
        labels_filtered = labels

    # Identify unique labels (after filtering)
    unique_labels = np.unique(labels_filtered)

    # Separate noise label
    labels_without_noise = unique_labels[unique_labels != -1]
    n_labels = len(labels_without_noise)

    # Choose a colormap with enough distinct colors, excluding greys
    if n_labels <= 9:  # Exclude grey from tab10
        cmap = plt.get_cmap("tab10")
        colors = [cmap.colors[i] for i in range(len(cmap.colors)) if i != 7]
    elif n_labels <= 18:  # Exclude greys from tab20
        cmap = plt.get_cmap("tab20")
        colors = [cmap.colors[i] for i in range(len(cmap.colors)) if i not in [14, 15]]
    else:
        # Generate a colormap with n_labels distinct colors
        cmap = plt.cm.get_cmap("hsv", n_labels)
        colors = cmap(np.linspace(0, 1, n_labels))

    # Create a mapping from label to color for clusters (excluding noise)
    color_map = {label: colors[i] for i, label in enumerate(labels_without_noise)}

    # Assign grey color to noise label
    if -1 in unique_labels:
        color_map[-1] = "grey"

    point_colors = [color_map[label] for label in labels_filtered]

    # Create masks for noise and cluster data
    noise_mask = labels_filtered == -1
    cluster_mask = labels_filtered != -1

    # Plot the scatter points for clusters
    ax.scatter(
        x[cluster_mask],
        y[cluster_mask],
        c=[point_colors[i] for i in range(len(point_colors)) if cluster_mask[i]],
        s=70,
        alpha=0.9,
        edgecolors="k",
        linewidth=0.9,
    )

    # Plot the scatter points for noise cluster
    if plot_noise and noise_mask.any():
        ax.scatter(
            x[noise_mask],
            y[noise_mask],
            c=[point_colors[i] for i in range(len(point_colors)) if noise_mask[i]],
            s=70,
            alpha=0.2,
            edgecolors="k",
            linewidth=0.5,
        )

    # If Fn_std is provided, add error bars
    if Fn_std is not None:
        # For cluster data
        if cluster_mask.any():
            ax.errorbar(
                x[cluster_mask].squeeze(),
                y[cluster_mask].squeeze(),
                xerr=Fn_std[cluster_mask].squeeze(),
                fmt="none",
                ecolor="gray",
                alpha=0.7,
                capsize=5,
            )
        # For noise data
        if plot_noise and noise_mask.any():
            ax.errorbar(
                x[noise_mask].squeeze(),
                y[noise_mask].squeeze(),
                xerr=Fn_std[noise_mask].squeeze(),
                fmt="none",
                ecolor="gray",
                alpha=0.5,
                capsize=5,
            )

    # Prepare legend labels
    legend_labels = {}
    cluster_counter = 1
    for label in unique_labels:
        if label == -1 and plot_noise:
            legend_labels[label] = "Noise"
        elif label == -1 and not plot_noise:
            continue  # Skip noise label
        else:
            legend_labels[label] = f"Cluster {cluster_counter}"
            cluster_counter += 1

    # Create custom legend handles with formatted labels
    legend_elements = []
    for label in unique_labels:
        if label == -1 and not plot_noise:
            continue  # Skip adding Noise to legend if plot_noise is False
        if label == -1:
            facecolor = adjust_alpha(color_map[label], 0.5)
        else:
            facecolor = adjust_alpha(color_map[label], 0.9)
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=facecolor,
                markeredgecolor="k",
                markeredgewidth=0.5,
                markersize=10,
                label=legend_labels[label],
            )
        )

    # Determine the number of columns for the legend
    ncols = 1 if len(legend_elements) <= 20 else 2

    legend_elements = rearrange_legend_elements(legend_elements, ncols)

    # Add the legend to the plot
    ax.legend(
        handles=legend_elements,
        title="Clusters",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        ncol=ncols,
        borderaxespad=0.0,
    )

    # Add vertical lines for each cluster (excluding Noise)
    for label in labels_without_noise:
        # Extract x-values for the current cluster
        cluster_x = x[labels_filtered == label]
        if len(cluster_x) == 0:
            continue  # Skip if no points in cluster
        median_x = np.median(cluster_x)
        ax.axvline(
            x=median_x, color=color_map[label], alpha=0.8, linestyle="--", linewidth=2
        )

    # Set plot titles and labels
    ax.set_title(f"Stabilization Chart with Clusters {name}")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Model Order")

    # Set y-axis limits
    ax.set_ylim(ordmin, ordmax + 1)

    # Set x-axis limits if freqlim is provided
    if freqlim is not None:
        ax.set_xlim(freqlim[0], freqlim[1])

    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    return fig, ax


# -----------------------------------------------------------------------------


def CMIF_plot(
    S_val: np.ndarray,
    freq: np.ndarray,
    freqlim: typing.Optional[typing.Tuple] = None,
    nSv: str = "all",
    fig: typing.Optional[plt.Figure] = None,
    ax: typing.Optional[plt.Axes] = None,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plots the Complex Mode Indicator Function (CMIF) based on given singular values and frequencies.

    Parameters
    ----------
    S_val : ndarray
        A 3D array representing the singular values, with shape [nChannel, nChannel, nFrequencies].
    freq : ndarray
        An array representing the frequency values corresponding to the singular values.
    freqlim : tuple of float, optional
        The frequency range (lower, upper) for the plot. If None, includes all frequencies. Default is None.
    nSv : int or str, optional
        The number of singular values to plot. If "all", plots all singular values.
        Otherwise, should be an integer specifying the number of singular values. Default is "all".
    fig : matplotlib.figure.Figure, optional
        An existing matplotlib figure object to plot on. If None, a new figure is created. Default is None.
    ax : matplotlib.axes.Axes, optional
        An existing axes object to plot on. If None, new axes are created on the provided or new figure.
        Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The axes object with the CMIF plot.

    Raises
    ------
    ValueError
        If `nSv` is not "all" and is not less than the number of singular values in `S_val`.
    """
    # COMPLEX MODE INDICATOR FUNCTION
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    if nSv == "all":
        nSv = S_val.shape[1]
    # Check that the number of singular value to plot is lower thant the total
    # number of singular values
    else:
        try:
            assert int(nSv) < S_val.shape[1]
        except Exception as e:
            raise ValueError(
                f"ERROR: nSV must be less or equal to {S_val.shape[1]}. nSV={int(nSv)}"
            ) from e

    for k in range(nSv):
        if k == 0:
            ax.plot(
                freq,
                10 * np.log10(S_val[k, k, :] / S_val[k, k, :][np.argmax(S_val[k, k, :])]),
                "k",
                linewidth=2,
            )
        else:
            ax.plot(
                freq,
                10 * np.log10(S_val[k, k, :] / S_val[0, 0, :][np.argmax(S_val[0, 0, :])]),
                "grey",
            )

    ax.set_title("Singular values of spectral matrix")
    ax.set_ylabel("dB rel. to unit")
    ax.set_xlabel("Frequency [Hz]")
    if freqlim is not None:
        ax.set_xlim(freqlim[0], freqlim[1])
    ax.grid()
    # plt.show()
    return fig, ax


# -----------------------------------------------------------------------------


def EFDD_FIT_plot(
    Fn: np.ndarray,
    Xi: np.ndarray,
    PerPlot: typing.List[typing.Tuple],
    freqlim: typing.Optional[typing.Tuple] = None,
) -> typing.Tuple[plt.Figure, typing.List[plt.Axes]]:
    """
    Plot detailed results for the Enhanced Frequency Domain Decomposition (EFDD) and
    the Frequency Spatial Domain Decomposition (FSDD) algorithms.

    Parameters
    ----------
    Fn : ndarray
        An array containing the natural frequencies identified for each mode.
    Xi : ndarray
        An array containing the damping ratios identified for each mode.
    PerPlot : list of tuples
        A list where each tuple contains data for one mode. Each tuple should have
        the structure (freq, time, SDOFbell, Sval, idSV, normSDOFcorr, minmax_fit_idx, lam, delta).
    freqlim : tuple of float, optional
        The frequency range (lower, upper) for the plots. If None, includes all frequencies. Default is None.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        A list of matplotlib figure objects.
    axs : list of lists of matplotlib.axes.Axes
        A list of lists containing axes objects for each figure.

    Note
    -----
    The function plots several aspects of the EFDD method for each mode, including the SDOF Bell function,
    auto-correlation function, and the selected portion for fit and the actual fit. Each mode's plot
    includes four subplots, showing the details of the EFDD fit process, including identified frequency
    and damping ratio.
    """

    figs = []
    axs = []
    for numb_mode in range(len(PerPlot)):
        freq = PerPlot[numb_mode][0]
        time = PerPlot[numb_mode][1]
        SDOFbell = PerPlot[numb_mode][2]
        Sval = PerPlot[numb_mode][3]
        idSV = PerPlot[numb_mode][4]
        fsval = freq[idSV]

        normSDOFcorr = PerPlot[numb_mode][5]
        minmax_fit_idx = PerPlot[numb_mode][6]
        lam = PerPlot[numb_mode][7]
        delta = PerPlot[numb_mode][8]

        xi_EFDD = Xi[numb_mode]
        fn_EFDD = Fn[numb_mode]

        # If the plot option is activated we return the following plots
        # build a rectangle in axes coords
        left, _ = 0.25, 0.5
        bottom, height = 0.25, 0.5
        # right = left + width
        top = bottom + height
        # axes coordinates are 0,0 is bottom left and 1,1 is upper right

        # PLOT 1 - Plotting the SDOF bell function extracted
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
        ax1.plot(
            freq, 10 * np.log10(Sval[0, 0] / Sval[0, 0][np.argmax(Sval[0, 0])]), c="k"
        )
        ax1.plot(
            fsval,
            10 * np.log10(SDOFbell[idSV].real / SDOFbell[np.argmax(SDOFbell)].real),
            c="r",
            label="SDOF bell",
        )
        ax1.set_title("SDOF Bell function")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel(r"dB rel to unit.$")
        ax1.grid()

        if freqlim is not None:
            ax1.set_xlim(freqlim[0], freqlim[1])

        ax1.legend()

        # Plot 2
        ax2.plot(time[:], normSDOFcorr, c="k")
        ax2.set_title("Auto-correlation Function")
        ax2.set_xlabel("Time lag[s]")
        ax2.set_ylabel("Normalized correlation")
        ax2.grid()

        # PLOT 3 (PORTION for FIT)
        ax3.plot(time[: minmax_fit_idx[-1]], normSDOFcorr[: minmax_fit_idx[-1]], c="k")
        ax3.scatter(time[minmax_fit_idx], normSDOFcorr[minmax_fit_idx], c="r", marker="x")
        ax3.set_title("Portion for fit")
        ax3.set_xlabel("Time lag[s]")
        ax3.set_ylabel("Normalized correlation")
        ax3.grid()

        # PLOT 4 (FIT)
        ax4.scatter(np.arange(len(minmax_fit_idx)), delta, c="k", marker="x")
        ax4.plot(
            np.arange(len(minmax_fit_idx)),
            lam / 2 * np.arange(len(minmax_fit_idx)),
            c="r",
        )

        ax4.text(
            left,
            top,
            r"""$f_n$ = %.3f
        $\xi$ = %.2f%s"""
            % (fn_EFDD, float(xi_EFDD) * 100, "%"),
            transform=ax4.transAxes,
        )

        ax4.set_title("Fit - Frequency and Damping")
        ax4.set_xlabel(r"counter $k^{th}$ extreme")
        ax4.set_ylabel(r"$2ln\left(r_0/|r_k|\right)$")
        ax4.grid()

        plt.tight_layout()

        figs.append(fig)
        axs.append([ax1, ax2, ax3, ax4])

    return figs, axs


# -----------------------------------------------------------------------------


def stab_plot(
    Fn: np.ndarray,
    Lab: np.ndarray,
    step: int,
    ordmax: int,
    ordmin: int = 0,
    freqlim: typing.Optional[typing.Tuple] = None,
    hide_poles: bool = True,
    Fn_std: np.array = None,
    fig: typing.Optional[plt.Figure] = None,
    ax: typing.Optional[plt.Axes] = None,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plots a stabilization chart of the poles of a system.

    Parameters
    ----------
    Fn : np.ndarray
        The frequencies of the poles.
    Lab : np.ndarray
        Labels indicating whether each pole is stable (1) or unstable (0).
    step : int
        The step size between model orders.
    ordmax : int
        The maximum model order.
    ordmin : int, optional
        The minimum model order, by default 0.
    freqlim : tuple, optional
        The frequency limits for the x-axis as a tuple (min_freq, max_freq), by default None.
    hide_poles : bool, optional
        Whether to hide the unstable poles, by default True.
    fig : plt.Figure, optional
        A matplotlib Figure object, by default None.
    ax : plt.Axes, optional
        A matplotlib Axes object, by default None.
    Fn_std : np.ndarray, optional
        The covariance of the frequencies, used for error bars, by default None.

    Returns
    -------
    fig : plt.Figure
        The matplotlib Figure object containing the plot.
    ax : plt.Axes
        The matplotlib Axes object containing the plot.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)

    # Stable pole
    Fns_stab = np.where(Lab == 1, Fn, np.nan)

    # new or unstable
    Fns_unstab = np.where(Lab == 0, Fn, np.nan)

    ax.set_title("Stabilisation Chart")
    ax.set_ylabel("Model Order")
    ax.set_xlabel("Frequency [Hz]")

    if hide_poles:
        x = Fns_stab.flatten(order="F")
        y = np.array([i // len(Fns_stab) for i in range(len(x))]) * step
        ax.plot(x, y, "go", markersize=7)

        if Fn_std is not None:
            xerr = Fn_std.flatten(order="f")

            ax.errorbar(x, y, xerr=xerr, fmt="None", capsize=5, ecolor="gray")

    else:
        x = Fns_stab.flatten(order="f")
        y = np.array([i // len(Fns_stab) for i in range(len(x))]) * step

        x1 = Fns_unstab.flatten(order="f")
        y1 = np.array([i // len(Fns_unstab) for i in range(len(x))]) * step

        ax.plot(x, y, "go", markersize=7, label="Stable pole")
        ax.scatter(x1, y1, marker="o", s=4, c="r", label="Unstable pole")

        if Fn_std is not None:
            xerr = abs(Fn_std).flatten(order="f")

            ax.errorbar(
                x, y, xerr=xerr.flatten(order="f"), fmt="None", capsize=5, ecolor="gray"
            )

            ax.errorbar(
                x1, y1, xerr=xerr.flatten(order="f"), fmt="None", capsize=5, ecolor="gray"
            )

        ax.legend(loc="lower center", ncol=2)
        ax.set_ylim(ordmin, ordmax + 1)

    ax.grid()
    if freqlim is not None:
        ax.set_xlim(freqlim[0], freqlim[1])
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------


def cluster_plot(
    Fn: np.ndarray,
    Xi: np.ndarray,
    Lab: np.ndarray,
    ordmin: int = 0,
    freqlim: typing.Optional[typing.Tuple] = None,
    hide_poles: bool = True,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plots the frequency-damping clusters of the identified poles.

    Parameters
    ----------
    Fn : ndarray
        An array containing the frequencies of poles for each model order and identification step.
    Xi : ndarray
        An array containing the damping ratios associated with the poles in `Fn`.
    Lab : ndarray
        Labels indicating whether each pole is stable (1) or unstable (0).
    ordmin : int, optional
        The minimum model order to be displayed on the plot. Default is 0.
    freqlim : tuple of float, optional
        The frequency limits for the x-axis as a tuple (min_freq, max_freq), by default None.
    hide_poles : bool, optional
        Whether to hide the unstable poles, by default True.

    Returns
    -------
    tuple
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        ax : matplotlib.axes.Axes
            The axes object with the stabilization chart.
    """
    # Stable pole
    a = np.where(Lab == 1, Fn, np.nan)
    aa = np.where(Lab == 1, Xi, np.nan)

    # new or unstable
    b = np.where(Lab == 0, Fn, np.nan)
    bb = np.where(Lab == 0, Xi, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.set_title("Frequency-damping clustering")
    ax.set_ylabel("Damping")
    ax.set_xlabel("Frequency [Hz]")
    if hide_poles:
        x = a.flatten(order="f")
        y = aa.flatten(order="f")
        ax.plot(x, y, "go", markersize=7, label="Stable pole")

    else:
        x = a.flatten(order="f")
        y = aa.flatten(order="f")

        x1 = b.flatten(order="f")
        y1 = bb.flatten(order="f")

        ax.plot(x, y, "go", markersize=7, label="Stable pole")

        ax.scatter(x1, y1, marker="o", s=4, c="r", label="Unstable pole")

        ax.legend(loc="lower center", ncol=2)

    ax.grid()
    if freqlim is not None:
        ax.set_xlim(freqlim[0], freqlim[1])
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------


def svalH_plot(
    H: np.ndarray,
    br: int,
    iter_n: int = None,
    fig: typing.Optional[plt.Figure] = None,
    ax: typing.Optional[plt.Axes] = None,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plot the singular values of the Hankel matrix.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)

    # SINGULAR VALUE DECOMPOSITION
    U1, S1, V1_t = np.linalg.svd(H)
    S1rad = np.sqrt(S1)

    ax.stem(S1rad, linefmt="k-")

    ax.set_title(f"Singular values plot, for block-rows = {br}")
    ax.set_ylabel("Singular values")
    ax.set_xlabel("Index number")
    if iter_n is not None:
        ax.set_xlim(-1, iter_n)

    ax.grid()

    return fig, ax


# -----------------------------------------------------------------------------


# =============================================================================
# PLOT GEO
# =============================================================================


def plt_nodes(
    ax: plt.Axes,
    nodes_coord: np.ndarray,
    alpha: float = 1.0,
    color: str = "k",
    initial_coord: np.ndarray = None,
) -> plt.Axes:
    """
    Plots nodes coordinates in a 3D scatter plot on the provided axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object where the nodes will be plotted. This should be a 3D axes.
    nodes_coord : ndarray
        A 2D array with dimensions [number of nodes, 3], representing the coordinates (x, y, z) of each node.
    alpha : float, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque). Default is 1.
    color : str or list of str, optional
        Color or list of colors for the nodes. Default is "k" (black). If 'cmap' is provided, initial_coord must be specified.
    initial_coord : ndarray, optional
        A 2D array with dimensions [number of nodes, 3], representing the initial coordinates (x, y, z) of each node.
        Required if color is 'cmap'.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object with the nodes plotted.

    Note
    -----
    This function is designed to work with 3D plots and adds node representations to an existing 3D plot.
    """
    if color == "cmap":
        if initial_coord is None:
            raise ValueError("initial_coord must be specified when color is 'cmap'")

        # Calculate distances from initial positions
        distances = np.linalg.norm(nodes_coord - initial_coord, axis=1)

        # Normalize distances to the range [0, 1]
        norm = plt.Normalize(vmin=np.min(distances), vmax=np.max(distances))
        cmap = plt.cm.plasma

        # Map distances to colors
        colors = cmap(norm(distances))
    else:
        colors = color
    if isinstance(colors, np.ndarray) and colors.ndim > 1:
        for iter in range(nodes_coord.shape[0]):
            ax.scatter(
                nodes_coord[iter, 0],
                nodes_coord[iter, 1],
                nodes_coord[iter, 2],
                alpha=0.5,
                color=matplotlib.colors.to_rgba(colors[iter, :]),
            )
    else:
        ax.scatter(
            nodes_coord[:, 0],
            nodes_coord[:, 1],
            nodes_coord[:, 2],
            alpha=alpha,
            color=colors,
        )
    return ax


def plt_lines(
    ax: plt.Axes,
    nodes_coord: np.ndarray,
    lines: np.ndarray,
    alpha: float = 1.0,
    color: str = "k",
    initial_coord: np.ndarray = None,
) -> plt.Axes:
    """
    Plots lines between specified nodes in a 3D plot on the provided axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object where the lines will be plotted. This should be a 3D axes.
    nodes_coord : ndarray
        A 2D array with dimensions [number of nodes, 3], representing the coordinates
        (x, y, z) of each node.
    lines : ndarray
        A 2D array with dimensions [number of lines, 2]. Each row represents a line,
        with the two elements being indices into `nodes_coord` indicating the start
        and end nodes of the line.
    alpha : float, optional
        The alpha blending value for the lines, between 0 (transparent) and 1 (opaque).
        Default is 1.
    color : str or list of str, optional
        Color or list of colors for the lines. Default is "k" (black). If 'cmap' is provided, initial_coord must be specified.
    initial_coord : ndarray, optional
        A 2D array with dimensions [number of nodes, 3], representing the initial coordinates (x, y, z) of each node.
        Required if color is 'cmap'.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object with the lines plotted.

    Note
    -----
    This function is designed to work with 3D plots and adds line representations between
    nodes in an existing 3D plot.
    """
    if color == "cmap":
        if initial_coord is None:
            raise ValueError("initial_coord must be specified when color is 'cmap'")

        # Calculate distances from initial positions
        distances_start = np.linalg.norm(
            nodes_coord[lines[:, 0]] - initial_coord[lines[:, 0]], axis=1
        )
        distances_end = np.linalg.norm(
            nodes_coord[lines[:, 1]] - initial_coord[lines[:, 1]], axis=1
        )

        # Calculate average distances
        avg_distances = (distances_start + distances_end) / 2

        # Normalize distances to the range [0, 1]
        norm = plt.Normalize(vmin=np.min(avg_distances), vmax=np.max(avg_distances))
        cmap = plt.cm.plasma

        # Map average distances to colors
        line_colors = cmap(norm(avg_distances))
    else:
        line_colors = [color] * lines.shape[0]

    for ii in range(lines.shape[0]):
        StartX, EndX = nodes_coord[lines[ii, 0]][0], nodes_coord[lines[ii, 1]][0]
        StartY, EndY = nodes_coord[lines[ii, 0]][1], nodes_coord[lines[ii, 1]][1]
        StartZ, EndZ = nodes_coord[lines[ii, 0]][2], nodes_coord[lines[ii, 1]][2]
        ax.plot(
            [StartX, EndX],
            [StartY, EndY],
            [StartZ, EndZ],
            alpha=alpha,
            color=line_colors[ii],
        )

    return ax


# -----------------------------------------------------------------------------


def plt_surf(
    ax: plt.Axes,
    nodes_coord: np.ndarray,
    surf: np.ndarray,
    alpha: float = 0.5,
    color: str = "cyan",
    initial_coord: np.ndarray = None,
) -> plt.Axes:
    """
    Plots a 3D surface defined by nodes and surface triangulation on the provided axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object where the surface will be plotted. This should be a 3D axes.
    nodes_coord : ndarray
        A 2D array with dimensions [number of nodes, 3], representing the coordinates (x, y, z) of each node.
    surf : ndarray
        A 2D array specifying the triangles that make up the surface. Each row represents a triangle,
        with the three elements being indices into `nodes_coord` indicating the vertices of the triangle.
    alpha : float, optional
        The alpha blending value for the surface, between 0 (transparent) and 1 (opaque). Default is 0.5.
    color : str, optional
        Color for the surface. Default is "cyan".

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object with the 3D surface plotted.

    Note
    -----
    This function is designed for plotting 3D surfaces in a 3D plot. It uses `matplotlib.tri.Triangulation`
    for creating a triangulated surface. Ideal for visualizing complex surfaces or meshes in a 3D space.
    """
    xy = nodes_coord[:, :2]
    z = nodes_coord[:, 2]
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=surf)

    if color == "cmap":
        if initial_coord is None:
            raise ValueError("initial_coord must be specified when color is 'cmap'")

        # Calculate distances from initial positions
        distances = np.linalg.norm(nodes_coord - initial_coord, axis=1)

        # Normalize distances to the range [0, 1]
        norm = plt.Normalize(vmin=np.min(distances), vmax=np.max(distances))
        cmap = plt.cm.plasma

        # Map distances to colors
        colors = cmap(norm(distances))

        # Loop through each triangle and plot
        for triangle_indices in surf:
            triangle_xy = xy[triangle_indices]
            triangle_z = z[triangle_indices]
            triangle_colors = colors[triangle_indices].mean(axis=0)
            triangle = mtri.Triangulation(
                triangle_xy[:, 0], triangle_xy[:, 1], triangles=[[0, 1, 2]]
            )
            ax.plot_trisurf(triangle, triangle_z, facecolor=triangle_colors, alpha=0.4)
    else:
        ax.plot_trisurf(triang, z, alpha=alpha, color=color)

    return ax


# -----------------------------------------------------------------------------


def plt_quiver(
    ax: plt.Axes,
    nodes_coord: np.ndarray,
    directions: np.ndarray,
    scaleF: float = 2,
    color: str = "red",
    names: typing.Optional[typing.List[str]] = None,
    color_text: str = "red",
    method: str = "1",
) -> plt.Axes:
    """
    Plots vectors (arrows) on a 3D plot to represent directions and magnitudes at given node coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object where the vectors will be plotted. This should be a 3D axes.
    nodes_coord : ndarray
        A 2D array with dimensions [number of nodes, 3], representing the coordinates (x, y, z)
        of each node.
    directions : ndarray
        A 2D array with the same shape as `nodes_coord`, representing the direction and magnitude vectors
        originating from the node coordinates.
    scaleF : float, optional
        Scaling factor for the magnitude of the vectors. Default is 2.
    color : str, optional
        Color of the vectors. Default is "red".
    names : list of str, optional
        Names or labels for each vector. If provided, labels are placed at the end of each vector.
        Default is None.
    color_text : str, optional
        Color of the text labels. Default is "red".

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object with the vectors plotted.

    Note
    -----
    Designed to work with 3D plots, allowing for the visualization of vector fields or directional data.
    `directions` array determines the direction and magnitude of the arrows, while `nodes_coord` specifies
    their starting points.
    """
    Points_f = nodes_coord + directions * scaleF
    xs0, ys0, zs0 = nodes_coord[:, 0], nodes_coord[:, 1], nodes_coord[:, 2]
    xs1, ys1, zs1 = Points_f[:, 0], Points_f[:, 1], Points_f[:, 2]
    if method == "1":
        ax.quiver(
            xs0,
            ys0,
            zs0,
            (xs1 - xs0),
            (ys1 - ys0),
            (zs1 - zs0),
            length=scaleF,
            color=color,
        )
    elif method == "2":
        for ii in range(len(xs0)):
            ax.plot(
                [xs0[ii], xs1[ii]],
                [ys0[ii], ys1[ii]],
                [zs0[ii], zs1[ii]],
                c=color,
                linewidth=2,
            )
    else:
        raise AttributeError("method must be either '1' or '2'!")

    if names is not None:
        for ii, nam in enumerate(names):
            ax.text(
                Points_f[ii, 0],
                Points_f[ii, 1],
                Points_f[ii, 2],
                f"{nam}",
                color=color_text,
            )
    return ax


# -----------------------------------------------------------------------------


def set_ax_options(
    ax: plt.Axes,
    bg_color: str = "w",
    remove_fill: bool = True,
    remove_grid: bool = True,
    remove_axis: bool = True,
    add_orig: bool = True,
    scaleF: float = 1,
) -> plt.Axes:
    """
    Configures various display options for a given matplotlib 3D axes object.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes object to be configured.
    bg_color : str, optional
        Background color for the axes. Default is "w" (white).
    remove_fill : bool, optional
        If True, removes the fill from the axes panes. Default is True.
    remove_grid : bool, optional
        If True, removes the grid from the axes. Default is True.
    remove_axis : bool, optional
        If True, turns off the axis lines, labels, and ticks. Default is True.
    add_orig : bool, optional
        If True, adds origin lines for the x, y, and z axes in red, green, and blue, respectively.
        Default is True.

    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        The modified 3D axes object with the applied configurations.

    Note
    -----
    Customizes the appearance of 3D plots. Controls background color, fill, grid, and axis visibility.
    """
    # avoid auto scaling of axis
    ax.set_aspect("equal")
    # Set backgroung color to white
    ax.xaxis.pane.set_edgecolor(bg_color)
    ax.yaxis.pane.set_edgecolor(bg_color)
    ax.zaxis.pane.set_edgecolor(bg_color)
    if remove_fill:
        # Remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    if remove_grid:
        # Get rid of the grid
        ax.grid(False)
    if remove_axis:
        # Turn off axis
        ax.set_axis_off()
    if add_orig:
        Or = (0, 0, 0)
        CS = np.asarray(
            [[0.4 * scaleF, 0, 0], [0, 0.4 * scaleF, 0], [0, 0, 0.4 * scaleF]]
        )
        _colours = ["r", "g", "b"]
        _axname = ["x", "y", "z"]
        for ii, _col in enumerate(_colours):
            ax.plot([Or[0], CS[ii, 0]], [Or[1], CS[ii, 1]], [Or[2], CS[ii, 2]], c=_col)
            # Q = ax.quiver(
            #     Or[0], Or[1], Or[2], (CS[ii,0] - Or[0]), (CS[ii,1] - Or[1]), (CS[ii,2] - Or[2]),
            #     color=_col,)
            ax.text(
                (CS[ii, 0] - Or[0]),
                (CS[ii, 1] - Or[1]),
                (CS[ii, 2] - Or[2]),
                f"{_axname[ii]}",
                color=_col,
            )

    return ax


# -----------------------------------------------------------------------------


def set_view(ax: plt.Axes, view: str) -> plt.Axes:
    """
    Sets the viewing angle of a 3D matplotlib axes object based on a predefined view option.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The 3D axes object whose view angle is to be set.
    view : str
        A string specifying the desired view. Options include "3D", "xy", "xz", and "yz".

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object with the new view angle set.

    Raises
    ------
    ValueError
        If the 'view' parameter is not one of the specified options.

    Note
    -----
    Useful for quickly setting the axes to a standard viewing angle, especially in 3D visualizations.
    View options: "3D" (azimuth -60, elevation 30), "xy" (top-down), "xz" (side, along y-axis),
    "yz" (side, along x-axis).
    """
    if view == "3D":
        azim = -60
        elev = 30
        ax.view_init(azim=azim, elev=elev)
    elif view == "xy":
        azim = 0
        elev = 90
        ax.view_init(azim=azim, elev=elev)
    elif view == "xz":
        azim = 90
        elev = 0
        ax.view_init(azim=azim, elev=elev)
    elif view == "yz":
        azim = 0
        elev = 0
        ax.view_init(azim=azim, elev=elev)
    else:
        raise ValueError(f"view must be one of (3D, xy, xz, yz), your input was '{view}'")
    return ax


# =============================================================================
# PLOT DATA
# =============================================================================


def plt_data(
    data: np.ndarray,
    fs: float,
    nc: int = 1,
    names: typing.Optional[typing.List[str]] = None,
    unit: str = "unit",
    show_rms: bool = False,
) -> typing.Tuple[plt.Figure, np.ndarray]:
    """
    Plots time series data for multiple channels, with an option to include the Root Mean Square (RMS)
    of each signal.

    Parameters
    ----------
    data : ndarray
        A 2D array with dimensions [number of data points, number of channels], representing signal data
        for multiple channels.
    fs : float
        Sampling frequency.
    nc : int, optional
        Number of columns for subplots, indicating how many subplots per row to display. Default is 1.
    names : list of str, optional
        Names of the channels, used for titling each subplot if provided. Default is None.
    unit : str, optional
        The unit to display on the y-axis label. Default is "unit".
    show_rms : bool, optional
        If True, includes the RMS of each signal on the corresponding subplot. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    axs : array of matplotlib.axes.Axes
        An array of axes objects for the generated subplots.

    Note
    -----
    Plots each channel in its own subplot for comparison. Supports multiple columns and adjusts the number of
    rows based on channels and columns.
    If `show_rms` is True, the RMS value of each signal is plotted as a constant line.
    """
    # show RMS of signal
    if show_rms is True:
        a_rmss = np.array(
            [
                np.sqrt(1 / len(data[:, _kk]) * np.sum(data[:, _kk] ** 2))
                for _kk in range(data.shape[1])
            ]
        )

    Ndat = data.shape[0]  # number of data points
    Nch = data.shape[1]  # number of channels
    timef = Ndat / fs  # final time value
    time = np.linspace(0, timef - 1 / fs, Ndat)  # time array

    nr = round(Nch / nc)  # number of rows in the subplot
    fig, axs = plt.subplots(
        figsize=(8, 6),
        nrows=nr,
        ncols=nc,
        sharex=True,
        sharey=True,
        tight_layout=True,
        squeeze=False,
    )
    fig.suptitle("Time Histories of all channels")

    kk = 0  # iterator for the dataset
    for ii in range(nr):
        # if there are more than one columns
        if nc != 1:
            # loop over the columns
            for jj in range(nc):
                ax = axs[ii, jj]
                ax.grid()
                try:
                    # while kk < data.shape[1]
                    ax.plot(time, data[:, kk], c="k")
                    if names is not None:
                        ax.set_title(f"{names[kk]}")
                    if ii == nr - 1:
                        ax.set_xlabel("time [s]")
                    if jj == 0:
                        ax.set_ylabel(f"{unit}")
                    if show_rms is True:
                        ax.plot(
                            time,
                            np.repeat(a_rmss[kk], len(time)),
                            label=f"arms={a_rmss[kk][0]:.3f}",
                            c="r",
                        )
                        ax.legend()
                except Exception as e:
                    logger.exception(e)

                kk += 1
        # if there is only 1 column
        else:
            ax = axs[ii, 0]
            ax.plot(time, data[:, kk], c="k")
            if names is not None:
                ax.set_title(f"{names[kk]}")
            if ii == nr - 1:
                ax.set_xlabel("time [s]")
            if show_rms is True:
                ax.plot(
                    time,
                    np.repeat(a_rmss[kk], len(time)),
                    label=f"arms={a_rmss[kk]:.3f}",
                    c="r",
                )
                ax.legend()
            ax.set_ylabel(f"{unit}")
            kk += 1
        # ax.grid()
    plt.tight_layout()

    return fig, axs


# -----------------------------------------------------------------------------


def plt_ch_info(
    data: np.ndarray,
    fs: float,
    nxseg: int = 1024,
    freqlim: typing.Optional[typing.Tuple] = None,
    logscale: bool = False,
    ch_idx: typing.Union[int, typing.List[int], str] = "all",
    unit: str = "unit",
) -> typing.Tuple[plt.Figure, np.ndarray]:
    """
    Plot channel information including time history, normalised auto-correlation,
    power spectral density (PSD), probability density function, and a normal
    probability plot for each channel in the data.

    Parameters
    ----------
    data : ndarray
        The input signal data.
    fs : float
        The sampling frequency of the input data in Hz.
    nxseg : int, optional
        The number of points per segment.
    freqlim : tuple of float, optional
        The frequency limits (min, max) for the PSD plot. If None, the full frequency range is used.
        Default is None.
    logscale : bool, optional
        If True, the PSD plot will be in log scale (decibel). Otherwise, it will be in linear scale.
        Default is False.
    ch_idx : int, list of int, or "all", optional
        The index (indices) of the channel(s) to plot. If "all", information for all channels is plotted.
        Default is "all".
    unit : str, optional
        The unit of the input data for labelling the PSD plot. Default is "unit".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing all the plots.
    axes : list of matplotlib.axes.Axes
        The list of axes objects corresponding to each subplot.

    Note
    -----
    This function is designed to provide a comprehensive overview of the signal characteristics for one or
    multiple channels of a dataset. It plots the time history, normalised auto-correlation, PSD, probability
    density function, and a normal probability plot for each specified channel.
    """
    if ch_idx != "all":
        data = data[:, ch_idx]

    ndat, nch = data.shape

    figs = []
    axs = []
    for ii in range(nch):
        fig = plt.figure(figsize=(8, 6), layout="constrained")
        spec = fig.add_gridspec(3, 2)

        # select channel
        x = data[:, ii]

        # Normalize data
        x = x - np.mean(x)
        x = x / np.std(x)
        sorted_x = np.sort(x)
        n = len(x)
        y = np.arange(1, n + 1) / n

        # Adjusted axis limits
        xlim = max(abs(sorted_x.min()), abs(sorted_x.max()))
        ylim = max(
            abs(np.sort(np.random.randn(n)).min()), abs(np.sort(np.random.randn(n)).max())
        )
        maxlim = np.max((xlim, ylim))

        # Plot 1: Time History
        ax0 = fig.add_subplot(spec[0, :])
        ax0.plot(np.linspace(0, len(x) / fs, len(x)), x, c="k")
        ax0.set_xlabel("Time [s]")
        ax0.set_title("Time History")
        ax0.set_ylabel("Unit")
        ax0.grid()

        # Plot 2: Normalised auto-correlation
        ax10 = fig.add_subplot(spec[1, 0])
        # R_i = np.array([ 1/(ndat - kk) * np.dot(x[:ndat-kk], x[kk:].T)  for kk in range(nxseg)])
        R_i = signal.correlate(x, x, mode="full")[len(x) - 1 : len(x) + nxseg - 1]
        R_i /= np.max(R_i)
        ax10.plot(np.linspace(0, len(R_i) / fs, len(R_i)), R_i, c="k")
        ax10.set_xlabel("Time [s]")
        ax10.set_ylabel("Norm. auto-corr.")
        ax10.set_title("Normalised auto-correlation")
        ax10.grid()

        # Plot 3: PSD
        freq, psd = signal.welch(
            x,
            fs,
            nperseg=nxseg,
            noverlap=nxseg * 0.5,
            window="hann",
        )
        ax20 = fig.add_subplot(spec[2, 0])
        if logscale is True:
            ax20.plot(freq, 10 * np.log10(psd), c="k")
            ax20.set_ylabel(f"dB rel. to {unit}")
        elif logscale is False:
            ax20.plot(freq, np.sqrt(psd), c="k")
            ax20.set_ylabel(rf"${unit}^2 / Hz$")
        if freqlim is not None:
            ax20.set_xlim(freqlim[0], freqlim[1])
        ax20.set_xlabel("Frequency [Hz]")
        ax20.set_title("PSD")
        ax20.grid()

        # Plot 4: Density function
        ax11 = fig.add_subplot(spec[1, 1])
        xm = min(abs(min(sorted_x)), abs(max(sorted_x)))
        dx = nxseg * xm / n
        xi = np.arange(-xm, xm + dx, dx)
        Fi = interp1d(sorted_x, y, kind="linear", fill_value="extrapolate")(xi)
        F2 = Fi[1:]
        F1 = Fi[:-1]
        f = (F2 - F1) / dx
        xf = (xi[1:] + xi[:-1]) / 2
        ax11.plot(
            xf,
            f,
            "k",
        )
        ax11.set_title("Probability Density Function")
        ax11.set_xlabel(
            "Normalised data",
        )
        ax11.set_ylabel("Probability")
        ax11.set_xlim(-xlim, xlim)
        ax11.grid()

        # Plot 5: Normal probability plot
        ax21 = fig.add_subplot(spec[2, 1])
        np.random.seed(0)
        xn = np.random.randn(n)
        sxn = np.sort(xn)
        ax21.plot(sorted_x, sxn, "k+", markersize=5)
        ax21.set_title(
            "Normal probability plot",
        )
        ax21.set_xlabel(
            "Normalised data",
        )
        ax21.set_ylabel(
            "Gaussian axis",
        )
        ax21.grid()
        ax21.set_xlim(-maxlim, maxlim)
        ax21.set_ylim(-maxlim, maxlim)

        if ch_idx != "all":
            fig.suptitle(f"Info plot channel nr.{ch_idx[ii]}")
        else:
            fig.suptitle(f"Info plot channel nr.{ii}")
        figs.append(fig)
        axs.append([ax0, ax10, ax20, ax11, ax21])
    # return fig, [ax0, ax10, ax20, ax11, ax21]
    return figs, axs


# -----------------------------------------------------------------------------


# Short time Fourier transform - SPECTROGRAM
def STFT(
    data: np.ndarray,
    fs: float,
    nxseg: int = 512,
    pov: float = 0.9,
    win: str = "hann",
    freqlim: typing.Optional[typing.Tuple] = None,
    ch_idx: typing.Union[int, typing.List[int], str] = "all",
) -> typing.Tuple[plt.Figure, np.ndarray]:
    """
    Perform the Short Time Fourier Transform (STFT) to generate spectrograms for given signal data.

    This function computes the STFT for each channel in the signal data, visualising the frequency content
    of the signal over time. It allows for the selection of specific channels and customisation of the
    STFT computation parameters.

    Parameters
    ----------
    data : ndarray
        The input signal data.
    fs : float
        The sampling frequency of the input data in Hz.
    nxseg : int, optional
        The number of points per segment for the STFT. Default is 512.
    pov : float, optional
        The proportion of overlap between segments, expressed as a value between 0 and 1. Default is 0.9.
    win : str, optional
        The type of window function to apply. Default is "hann".
    freqlim : tuple of float, optional
        The frequency limits (minimum, maximum) for the frequency axis of the spectrogram. If None,
        the full frequency range is used. Default is None.
    ch_idx : int, list of int, or "all", optional
        The index (indices) of the channel(s) to compute the STFT for. If "all", the STFT for all
        channels is computed. Default is "all".

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        The list of figure objects created, each corresponding to a channel in the input data.
    axs : list of matplotlib.axes.Axes
        The list of Axes objects corresponding to each figure, used for plotting the spectrograms.

    Notes
    -----
    The function visualises the magnitude of the STFT, showing how the frequency content of the signal
    changes over time. This is useful for analysing non-stationary signals. The function returns the
    figures and axes for further customisation or display.
    """
    if ch_idx != "all":
        data = data[:, ch_idx]

    ndat, nch = data.shape
    figs = []
    axs = []
    for ii in range(nch):
        # select channel
        ch = data[:, ii]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()
        ax.set_title(f"STFT Magnitude for channel nr.{ii+1}")
        ax.set_xlabel("Time [sec]")
        ax.set_ylabel("Frequency [Hz]")
        # nxseg = w_T*fs
        noverlap = nxseg * pov
        freq, time, Sxx = signal.stft(
            ch, fs, window=win, nperseg=nxseg, noverlap=noverlap
        )
        if freqlim is not None:
            idx1 = np.argmin(abs(freq - freqlim[0]))
            idx2 = np.argmin(abs(freq - freqlim[1]))

            freq = freq[idx1:idx2]
            Sxx = Sxx[idx1:idx2]
        ax.pcolormesh(time, freq, np.abs(Sxx))
        plt.tight_layout()
        figs.append(fig)
        axs.append(ax)
    return figs, axs


# -----------------------------------------------------------------------------


def plot_mac_matrix(
    array1, array2, colormap="plasma", ax=None
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Compute and plot the MAC matrix between the columns of two 2D arrays.

    Parameters
    ----------
    array1 : np.ndarray
        The first 2D array with shape (n_modes, n_dofs).
    array2 : np.ndarray
        The second 2D array with shape (n_modes, n_dofs).
    colormap : str, optional
        The colormap to use for the plot. Default is 'plasma'.
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on. If None, a new figure and axes will be created. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    # Check if there are more than 1 column vector in the input arrays
    if array1.shape[1] < 2 or array2.shape[1] < 2:
        raise ValueError("Each input array must have more than one column vector.")

    mac_matr = MAC(array1, array2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    else:
        fig = ax.figure

    cax = ax.imshow(mac_matr, cmap=colormap, aspect="auto")
    fig.colorbar(cax, ax=ax, label="MAC value")

    n_cols1 = array1.shape[1]
    n_cols2 = array2.shape[1]
    x_labels = [f"mode nr. {i+1}" for i in range(n_cols1)]
    y_labels = [f"mode nr. {i+1}" for i in range(n_cols2)]

    ax.set_xticks(np.arange(n_cols1))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_yticks(np.arange(n_cols2))
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Array 1")
    ax.set_ylabel("Array 2")
    ax.set_title("MAC Matrix")

    return fig, ax


# -----------------------------------------------------------------------------


def plot_mode_complexity(mode_shape):
    """
    Plot the complexity of a mode shape on a polar plot.

    This function visualizes the mode shape's complexity by representing its
    magnitudes and angles in a polar coordinate system. The magnitudes of
    the mode shape are plotted as arrows radiating outward, with their angles
    representing the phase of the corresponding components. Principal directions
    (0° and 180°) are highlighted for clarity.

    Parameters
    ----------
    mode_shape : array_like
        A complex-valued array representing the mode shape. Each element's
        magnitude and angle are used to generate the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.

    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """

    # Get angles (in radians) and magnitudes
    angles = np.angle(mode_shape)
    magnitudes = np.abs(mode_shape)

    # Create a polar plot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.set_theta_zero_location("E")  # Set 0 degrees to East
    ax.set_theta_direction(1)  # Counterclockwise
    ax.set_rmax(1.1)  # Set maximum radius slightly above 1 for clarity
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot arrows using annotate with fixed head size
    for angle, magnitude in zip(angles, magnitudes):
        ax.annotate(
            "",
            xy=(angle, magnitude),
            xytext=(angle, 0),
            arrowprops=dict(
                facecolor="blue",
                edgecolor="blue",
                arrowstyle="-|>",
                linewidth=1.5,
                mutation_scale=20,  # Controls the size of the arrowhead
            ),
        )
    # Highlight directions (0° and 180°)
    principal_angles = [0, np.pi]
    for pa in principal_angles:
        ax.plot([pa, pa], [0, 1.1], color="red", linestyle="--", linewidth=1)

    ax.set_yticklabels([])
    # Add title
    ax.set_title(
        "Mode Shape Complexity Plot", va="bottom", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
    return fig, ax
