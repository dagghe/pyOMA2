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
from sklearn.metrics import silhouette_samples, silhouette_score

from .gen import MAC

logger = logging.getLogger(__name__)

# =============================================================================
# COLOR CONFIGURATION
# =============================================================================

DEFAULT_COLORS = {
    "stable": "tab:green",
    "unstable": "tab:red",
}

# Alternative color sets for different preferences
ALTERNATIVE_COLORS = {
    # Good for color blind
    "classic": {"stable": "blue", "unstable": "orange"},
    # High contrast for B&W printing
    "high_contrast": {"stable": "black", "unstable": "gray"},
    # Viridis colormap extremes
    "viridis": {"stable": "#440154", "unstable": "#FDE725"},
}


def get_pole_colors(
    color_scheme: typing.Literal[
        "default", "classic", "high_contrast", "viridis"
    ] = "default",
) -> dict:
    """
    Get color scheme for stable and unstable poles.

    Parameters
    ----------
    color_scheme : str, optional
        Color scheme to use. Options: 'default', 'classic', 'high_contrast', 'viridis'.
        Default is 'default' which uses colorblind-friendly colors.

    Returns
    -------
    dict
        Dictionary with 'stable' and 'unstable' color keys.
    """
    if color_scheme == "default":
        return DEFAULT_COLORS.copy()
    elif color_scheme in ALTERNATIVE_COLORS:
        return ALTERNATIVE_COLORS[color_scheme].copy()
    else:
        logger.warning(f"Unknown color scheme '{color_scheme}', using default")
        return DEFAULT_COLORS.copy()


# =============================================================================
# PLOT for CLUSTER
# =============================================================================


def plot_silhouette(
    distance_matrix: np.ndarray, labels: np.ndarray, name: str
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plot a silhouette plot for clustering results given a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : array-like, shape (n_samples, n_samples)
        Pairwise distance matrix between samples.
    labels : array-like, shape (n_samples,)
        Cluster labels for each sample.
    name : str
        A label or title identifier used in the plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the silhouette plot.
    ax : matplotlib.axes.Axes
        The axes object containing the silhouette plot.

    Notes
    -----
    - Computes the average silhouette score for the provided labels.
    - Colors each cluster's silhouette values, and draws a vertical line at the average score.
    """
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))

    silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
    sample_silhouette_values = silhouette_samples(
        distance_matrix, labels, metric="precomputed"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    # Choose colormap based on number of clusters, excluding greys where specified
    if n_clusters <= 9:
        cmap = plt.get_cmap("tab10")
        colors = [cmap.colors[i] for i in range(len(cmap.colors)) if i != 7]
    elif n_clusters <= 18:
        cmap = plt.get_cmap("tab20")
        colors = [cmap.colors[i] for i in range(len(cmap.colors)) if i not in [14, 15]]
    else:
        cmap = plt.cm.get_cmap("hsv", n_clusters)
        colors = cmap(np.linspace(0, 1, n_clusters))

    for i in range(1, n_clusters + 1):
        cluster_idx = i - 1
        ith_vals = sample_silhouette_values[labels == cluster_idx]
        ith_vals.sort()
        size_i = ith_vals.shape[0]
        y_upper = y_lower + size_i

        color = colors[cluster_idx]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10  # 10px gap between clusters

    avg_label = f"Avg = {silhouette_avg:.3f}"
    ax.axvline(
        x=silhouette_avg, color="red", linestyle="--", linewidth=2, label=avg_label
    )
    ax.legend(loc="upper right")

    ax.set_title(f"Silhouette Plot - {name}")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1.0])

    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------


def plot_dtot_hist(
    dtot: np.ndarray, bins: typing.Union[int, str] = "auto", sugg_co: bool = True
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plot a histogram of the total distance data with optional suggested cut-off distances.

    This function plots a histogram of the values in the input distance matrix or vector `dtot`.
    It overlays a kernel density estimate (KDE) and optionally indicates suggested cut-off distances
    for clustering using single-linkage and average-linkage methods.

    Parameters
    ----------
    dtot : ndarray
        The input distance data. If a 2D array is provided, the upper triangular elements
        (including the diagonal) are extracted and flattened. If a 1D array is provided, it is used directly.
    bins : int or str, optional
        The number of bins or binning strategy for the histogram. Defaults to "auto".
    sugg_co : bool, optional
        Whether to compute and display suggested cut-off distances for clustering.
        Defaults to True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the histogram and KDE plot.
    ax : matplotlib.axes.Axes
        The axes object for the histogram (primary y-axis). The KDE is plotted on a secondary y-axis.

    Notes
    -----
    - If `dtot` is 2D, all entries from the upper triangular part (k=0) are used.
    - Uses SciPy's `gaussian_kde` to overlay a continuous density estimate.
    - For suggested cut-offs:
        * Single-linkage: the first local maximum of the KDE.
        * Average-linkage: the first local minimum of the KDE.
    """
    if dtot.ndim == 2:
        upper_tri_indices = np.triu_indices_from(dtot, k=0)
        x = dtot[upper_tri_indices]
    elif dtot.ndim == 1:
        x = dtot
    else:
        raise ValueError("dtot must be a 1D or 2D array.")

    xs = np.linspace(x.min(), x.max(), 500)
    kde = stats.gaussian_kde(x)
    pdf = kde(xs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(x, bins=bins, color="skyblue", edgecolor="black")
    ax1 = ax.twinx()
    ax1.plot(xs, pdf, "k-", label="Kernel density estimate")

    if sugg_co:
        minima_idxs = signal.argrelmin(pdf)[0]
        maxima_idxs = signal.argrelmax(pdf)[0]
        if minima_idxs.size > 0:
            min_vals = pdf[minima_idxs]
            min_abs_index = minima_idxs[np.argmin(min_vals)]
            dc1 = xs[min_abs_index]
            ax1.axvline(
                dc1,
                color="red",
                linestyle="dotted",
                linewidth=2,
                label=f"Suggested avg-linkage cut-off: {dc1:.4f}",
            )
        if maxima_idxs.size > 0:
            dc2 = xs[maxima_idxs[0]]
            ax1.axvline(
                dc2,
                color="red",
                linestyle="dashed",
                linewidth=2,
                label=f"Suggested single-linkage cut-off: {dc2:.4f}",
            )

    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of distances with KDE")
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax.grid(which="major", axis="x", color="gray", linestyle="-", linewidth=0.5)
    ax.grid(
        which="minor", axis="x", color="gray", linestyle=":", linewidth=0.5, alpha=0.7
    )
    ax1.legend(framealpha=1)
    plt.tight_layout()

    return fig, ax


# -----------------------------------------------------------------------------


def adjust_alpha(color: typing.Union[str, tuple], alpha: float) -> tuple:
    """
    Adjust the alpha (opacity) of a given color.

    Parameters
    ----------
    color : str or tuple
        The input color in any valid Matplotlib format (e.g., color name, hexadecimal code, or RGB(A) tuple).
    alpha : float
        The desired alpha value, between 0 (completely transparent) and 1 (completely opaque).

    Returns
    -------
    rgba: tuple
        The RGBA representation of the input color with the specified alpha applied.
    """
    rgba = to_rgba(color)
    return rgba[:3] + (alpha,)


# -----------------------------------------------------------------------------


def rearrange_legend_elements(
    legend_elements: typing.List[Line2D], ncols: int
) -> typing.List[Line2D]:
    """
    Rearrange legend elements into a column-wise ordering.

    For example, if there are 6 elements and 2 columns, ordering by columns yields:
    [e1, e3, e5, e2, e4, e6]

    Parameters
    ----------
    legend_elements : list of matplotlib.lines.Line2D
        A list of legend handles (e.g., Line2D instances) to be rearranged.
    ncols : int
        The desired number of columns in the legend.

    Returns
    -------
    rearranged_elements : list of matplotlib.lines.Line2D
        A reordered list of legend handles arranged column-wise.
    """
    n = len(legend_elements)
    nrows = int(np.ceil(n / ncols))
    total_entries = nrows * ncols
    padded = legend_elements + [None] * (total_entries - n)
    array = np.array(padded, dtype=object).reshape(nrows, ncols)
    flattened = array.flatten(order="F")
    return [elem for elem in flattened if elem is not None]


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
    ax.set_title(f"Frequency vs Damping - Clusters, '{name}'")
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
    ordmax: int,
    ordmin: int = 0,
    freqlim: typing.Optional[typing.Tuple[float, float]] = None,
    Fn_std: typing.Optional[np.ndarray] = None,
    plot_noise: bool = False,
    name: typing.Optional[str] = None,
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
    Fn_fl : ndarray, shape (n_points,)
        Array of frequency values for each identified pole, flattened.
    order_fl : ndarray, shape (n_points,)
        Array of model order values corresponding to each identified pole.
    labels : ndarray, shape (n_points,)
        Cluster labels for each data point. Use -1 for noise points.
    ordmax : int
        Maximum model order; used to set y-axis limit.
    ordmin : int, optional
        Minimum model order; used to set y-axis limit. Default is 0.
    freqlim : tuple(float, float), optional
        Frequency axis limits as (min_freq, max_freq). If None, auto-scale is used.
    Fn_std : ndarray, optional
        Standard deviation of frequency values, used for horizontal error bars. Should match `Fn_fl` shape.
        Default is None.
    plot_noise : bool, optional
        Whether to include and plot noise-labeled points (-1) in grey. Defaults to False.
    name : str, optional
        An identifier used in the plot title. Defaults to None.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, new axes are created on the provided or new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object containing the stabilization chart.
    ax : matplotlib.axes.Axes
        The Matplotlib axes object with the plotted data.

    Notes
    -----
    - Error bars for frequency (horizontal) are drawn if `Fn_std` is provided.
    - Vertical dashed lines at median frequency for each cluster (excluding noise).
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
    ax.set_title(f"Stabilization Chart with Clusters, '{name}'")
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


# =============================================================================
# PLOT for FDD
# =============================================================================


def CMIF_plot(
    S_val: np.ndarray,
    freq: np.ndarray,
    freqlim: typing.Optional[typing.Tuple[float, float]] = None,
    nSv: typing.Union[int, str] = "all",
    fig: typing.Optional[plt.Figure] = None,
    ax: typing.Optional[plt.Axes] = None,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plot the Complex Mode Indicator Function (CMIF) based on given singular values and frequencies.

    Parameters
    ----------
    S_val : ndarray, shape (n_channel, n_channel, n_frequencies)
        A 3D array representing the singular values of the spectral matrix at each frequency.
    freq : ndarray, shape (n_frequencies,)
        Frequency vector corresponding to the third axis of S_val.
    freqlim : tuple(float, float), optional
        Frequency limits for the x-axis as (min_freq, max_freq). If None, the full frequency range is used.
    nSv : int or "all", optional
        The number of singular values to plot per mode. If "all", all singular values are plotted.
        Otherwise, must be an integer less than or equal to the number of modes. Defaults to "all".
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, new axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object containing the CMIF plot.
    ax : matplotlib.axes.Axes
        The Matplotlib axes object containing the CMIF curves.

    Raises
    ------
    ValueError
        If `nSv` is not "all" and is greater than the number of modes in S_val.
    """
    # COMPLEX MODE INDICATOR FUNCTION

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
        title = "Singular values of spectral matrix"
        ls = "-"
        alpha = 1.0
    else:
        title = ax.get_title()
        ls = "--"
        alpha = 0.5

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
                alpha=alpha,
            )
        else:
            ax.plot(
                freq,
                10 * np.log10(S_val[k, k, :] / S_val[0, 0, :][np.argmax(S_val[0, 0, :])]),
                "grey",
                alpha=alpha,
            )

    ax.set_title(title)
    ax.set_ylabel("dB rel. to unit")
    ax.set_xlabel("Frequency [Hz]")
    if freqlim is not None:
        ax.set_xlim(freqlim[0], freqlim[1])
    ax.grid(ls=ls, alpha=alpha)
    # plt.show()
    return fig, ax


# -----------------------------------------------------------------------------


def EFDD_FIT_plot(
    Fn: np.ndarray,
    Xi: np.ndarray,
    PerPlot: typing.List[typing.Tuple],
    freqlim: typing.Optional[typing.Tuple[float, float]] = None,
) -> typing.Tuple[typing.List[plt.Figure], typing.List[typing.List[plt.Axes]]]:
    """
    Plot detailed results for the Enhanced Frequency Domain Decomposition (EFDD) and
    the Frequency Spatial Domain Decomposition (FSDD) algorithms.

    For each mode in `PerPlot`, this function creates a 2x2 subplot figure showing:
    1. The Spectral Density Function (SDOF bell) compared to the spectrum peak.
    2. The normalized autocorrelation function over time.
    3. The selected portion of the autocorrelation used for the damping fit.
    4. The linear fit to ln(r0/|rk|) vs. index of extrema, from which frequency and damping are derived.

    Parameters
    ----------
    Fn : ndarray, shape (n_modes,)
        Array of natural frequencies identified for each mode.
    Xi : ndarray, shape (n_modes,)
        Array of damping ratios identified for each mode.
    PerPlot : list of tuples, length = n_modes
        Each tuple corresponds to one mode and contains:
        (freq, time, SDOFbell, Sval, idSV, normSDOFcorr, minmax_fit_idx, lam, delta)
        - freq : ndarray of frequency values
        - time : ndarray of time lags for autocorrelation
        - SDOFbell : ndarray of SDOF bell function values
        - Sval : ndarray of spectral values used to compute SDOFbell
        - idSV : indices of frequency bins at which SDOFbell is evaluated
        - normSDOFcorr : normalized autocorrelation values
        - minmax_fit_idx : indices of minima/maxima in autocorrelation used for fit
        - lam : slope parameter used in damping fit
        - delta : values of 2 * ln(r0 / |rk|) used for linear regression
    freqlim : tuple(float, float), optional
        Frequency axis limits for the first subplot (SDOF bell) as (min_freq, max_freq).
        If None, the full frequency range is shown.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        List of figure objects, one per mode.
    axs : list of list of matplotlib.axes.Axes
        Nested list of axes for each figure: [[ax1, ax2, ax3, ax4], ...] for each mode.

    Notes
    -----
    - Subplot arrangement:
        (ax1) SDOF Bell vs. normalized spectral peak
        (ax2) Normalized autocorrelation function
        (ax3) Portion of autocorrelation selected for fit (highlighted extrema)
        (ax4) Linear fit for 2 ln(r0/|rk|) vs. extrema index, annotated with Fn and Xi
    """
    figs = []
    axs = []
    n_modes = len(PerPlot)

    for mode_idx in range(n_modes):
        (
            freq_vals,
            time_vals,
            SDOFbell,
            Sval,
            idSV,
            normSDOFcorr,
            minmax_fit_idx,
            lam,
            delta,
        ) = PerPlot[mode_idx]
        xi_EFDD = Xi[mode_idx]
        fn_EFDD = Fn[mode_idx]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        # Plot 1: SDOF bell function vs normalized Sval
        ref_peak = np.argmax(Sval[0, 0])
        norm_Sval = 10 * np.log10(Sval[0, 0] / Sval[0, 0][ref_peak])
        ax1.plot(freq_vals, norm_Sval, c="k", label="Normalized Sval (peak)")
        SDOFbell_norm = 10 * np.log10(
            SDOFbell[idSV].real / SDOFbell[np.argmax(SDOFbell)].real
        )
        ax1.plot(freq_vals[idSV], SDOFbell_norm, c="r", linestyle="-", label="SDOF bell")
        ax1.set_title("SDOF Bell Function")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("dB rel to peak")
        ax1.grid(True)
        if freqlim is not None:
            ax1.set_xlim(freqlim[0], freqlim[1])
        ax1.legend()

        # Plot 2: Normalized autocorrelation
        ax2.plot(time_vals, normSDOFcorr, c="k")
        ax2.set_title("Normalized Autocorrelation")
        ax2.set_xlabel("Time lag [s]")
        ax2.set_ylabel("Correlation")
        ax2.grid(True)

        # Plot 3: Portion for fit (highlighting minima/maxima)
        ax3.plot(
            time_vals[: minmax_fit_idx[-1]], normSDOFcorr[: minmax_fit_idx[-1]], c="k"
        )
        ax3.scatter(
            time_vals[minmax_fit_idx],
            normSDOFcorr[minmax_fit_idx],
            c="r",
            marker="x",
            label="Extrema",
        )
        ax3.set_title("Selected Portion for Fit")
        ax3.set_xlabel("Time lag [s]")
        ax3.set_ylabel("Correlation")
        ax3.grid(True)
        ax3.legend()

        # Plot 4: Linear fit of 2 ln(r0/|rk|) vs extrema index
        ax4.scatter(
            np.arange(len(minmax_fit_idx)), delta, c="k", marker="x", label="Data points"
        )
        ax4.plot(
            np.arange(len(minmax_fit_idx)),
            (lam / 2) * np.arange(len(minmax_fit_idx)),
            c="r",
            label="Fit: slope = λ/2",
        )
        annotation_text = (r"$f_n$ = {fn:.3f} Hz" "\n" r"$\xi$ = {xi:.2f}%").format(
            fn=fn_EFDD, xi=float(xi_EFDD) * 100
        )
        ax4.text(
            0.05,
            0.95,
            annotation_text,
            transform=ax4.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax4.set_title("Damping Fit: 2 ln(r0/|rk|) vs Index")
        ax4.set_xlabel("Index of extrema (k)")
        ax4.set_ylabel(r"$2 \ln(r_0 / |r_k|)$")
        ax4.grid(True)
        ax4.legend()

        plt.tight_layout()
        figs.append(fig)
        axs.append([ax1, ax2, ax3, ax4])

    return figs, axs


# =============================================================================
# PLOT for SSI
# =============================================================================


def stab_plot(
    Fn: np.ndarray,
    Lab: np.ndarray,
    step: int,
    ordmax: int,
    ordmin: int = 0,
    freqlim: typing.Optional[typing.Tuple[float, float]] = None,
    hide_poles: bool = True,
    Fn_std: typing.Optional[np.ndarray] = None,
    fig: typing.Optional[plt.Figure] = None,
    ax: typing.Optional[plt.Axes] = None,
    color_scheme: typing.Literal[
        "default", "classic", "high_contrast", "viridis"
    ] = "default",
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plot a stabilization chart of poles (frequency vs. model order) for stable vs unstable labeling.

    Parameters
    ----------
    Fn : ndarray, shape (n_modes, n_orders)
        2D array of frequencies for each pole, arranged by mode and order.
    Lab : ndarray, shape (n_modes, n_orders)
        2D array of labels indicating whether each pole is stable (1) or unstable (0).
    step : int
        Incremental step for model order plotting (vertical axis spacing).
    ordmax : int
        Maximum model order to display (upper y-limit).
    ordmin : int, optional
        Minimum model order to display (lower y-limit). Default is 0.
    freqlim : tuple(float, float), optional
        Frequency axis limits as (min_freq, max_freq). If None, auto-scale is used.
    hide_poles : bool, optional
        If True, only stable poles (Lab == 1) are shown. If False, both stable and unstable are shown.
    Fn_std : ndarray, optional
        Covariance (standard deviation) of frequencies, same shape as `Fn`. Used for horizontal error bars.
        Defaults to None.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, new axes are created on the provided or new figure.
    color_scheme : str, optional
        Color scheme to use for stable/unstable poles. Options: 'default', 'classic',
        'high_contrast', 'viridis'. Default is 'default' (colorblind-friendly).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the stabilization chart.
    ax : matplotlib.axes.Axes
        The axes object containing the plotted poles.

    Notes
    -----
    - By default uses colorblind-friendly blue/orange colors for stable/unstable poles.
    - Error bars represent frequency uncertainty if `Fn_std` is provided.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)

    # Get colors based on scheme
    colors = get_pole_colors(color_scheme)

    # Extract stable and unstable frequencies
    Fns_stab = np.where(Lab == 1, Fn, np.nan)
    Fns_unstab = np.where(Lab == 0, Fn, np.nan)

    ax.set_title("Stabilisation Chart")
    ax.set_ylabel("Model Order")
    ax.set_xlabel("Frequency [Hz]")

    # Indices for plotting
    if hide_poles:
        x = Fns_stab.flatten(order="F")
        y = np.arange(x.size) // Fn.shape[0] * step
        ax.plot(x, y, "o", color=colors["stable"], markersize=7)

        if Fn_std is not None:
            xerr = Fn_std.flatten(order="F")
            ax.errorbar(x, y, xerr=xerr, fmt="None", capsize=5, ecolor="gray")
    else:
        x_stab = Fns_stab.flatten(order="F")
        y_stab = np.arange(x_stab.size) // Fn.shape[0] * step
        x_unstab = Fns_unstab.flatten(order="F")
        y_unstab = np.arange(x_unstab.size) // Fn.shape[0] * step

        ax.plot(
            x_stab, y_stab, "o", color=colors["stable"], markersize=7, label="Stable pole"
        )
        ax.scatter(x_unstab, y_unstab, c=colors["unstable"], s=30, label="Unstable pole")

        if Fn_std is not None:
            xerr = np.abs(Fn_std.flatten(order="F"))
            ax.errorbar(x_stab, y_stab, xerr=xerr, fmt="None", capsize=5, ecolor="gray")
            ax.errorbar(
                x_unstab, y_unstab, xerr=xerr, fmt="None", capsize=5, ecolor="gray"
            )

        ax.legend(loc="lower center", ncol=2)
        ax.set_ylim(ordmin, ordmax + 1)

    ax.grid(True)
    if freqlim is not None:
        ax.set_xlim(freqlim[0], freqlim[1])
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------


def cluster_plot(
    Fn: np.ndarray,
    Xi: np.ndarray,
    Lab: np.ndarray,
    freqlim: typing.Optional[typing.Tuple[float, float]] = None,
    hide_poles: bool = True,
    color_scheme: typing.Literal[
        "default", "classic", "high_contrast", "viridis"
    ] = "default",
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plot a frequency-damping clustering chart, marking stable vs unstable poles.

    Parameters
    ----------
    Fn : ndarray, shape (n_modes, n_orders)
        2D array of frequencies for each pole, arranged by mode and order.
    Xi : ndarray, shape (n_modes, n_orders)
        2D array of damping ratios corresponding to each pole. Same shape as Fn.
    Lab : ndarray, shape (n_modes, n_orders)
        2D array of labels indicating whether each pole is stable (1) or unstable (0).
    freqlim : tuple(float, float), optional
        Frequency axis limits as (min_freq, max_freq). If None, full range is used.
    hide_poles : bool, optional
        If True, only stable poles are plotted. If False, both stable and unstable are plotted.
    color_scheme : Literal["default", "classic", "high_contrast", "viridis"], optional
        Color scheme to use for stable/unstable poles. Options: 'default', 'classic',
        'high_contrast', 'viridis'. Default is 'default' (colorblind-friendly).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the cluster chart.
    ax : matplotlib.axes.Axes
        The axes object containing the plotted clusters.

    Notes
    -----
    - By default uses colorblind-friendly blue/orange colors for stable/unstable poles.
    - Sets equal aspect by default.
    """
    # Get colors based on scheme
    colors = get_pole_colors(color_scheme=color_scheme)

    # Extract stable (a, aa) and unstable (b, bb) data
    a = np.where(Lab == 1, Fn, np.nan)
    aa = np.where(Lab == 1, Xi, np.nan)
    b = np.where(Lab == 0, Fn, np.nan)
    bb = np.where(Lab == 0, Xi, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.set_title("Frequency-Damping Clustering")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Damping")

    if hide_poles:
        x = a.flatten(order="F")
        y = aa.flatten(order="F")
        ax.plot(x, y, "o", color=colors["stable"], markersize=7, label="Stable pole")
    else:
        x_stab = a.flatten(order="F")
        y_stab = aa.flatten(order="F")
        x_unstab = b.flatten(order="F")
        y_unstab = bb.flatten(order="F")

        ax.plot(
            x_stab, y_stab, "o", color=colors["stable"], markersize=7, label="Stable pole"
        )
        ax.scatter(x_unstab, y_unstab, c=colors["unstable"], s=30, label="Unstable pole")
        ax.legend(loc="lower center", ncol=2)

    ax.grid(True)
    if freqlim is not None:
        ax.set_xlim(freqlim[0], freqlim[1])
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------


def svalH_plot(
    H: np.ndarray,
    br: int,
    iter_n: typing.Optional[int] = None,
    fig: typing.Optional[plt.Figure] = None,
    ax: typing.Optional[plt.Axes] = None,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plot the singular values of a Hankel matrix as a stem plot.

    Parameters
    ----------
    H : ndarray, shape (m, n)
        Hankel matrix for which singular values will be computed.
    br : int
        Number of block-rows used to construct the Hankel matrix; shown in the plot title.
    iter_n : int, optional
        Maximum iteration or model order, used to set x-axis limit. If provided, x-axis spans [-1, iter_n].
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, new axes are created on the provided or new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the singular values plot.
    ax : matplotlib.axes.Axes
        The axes object containing the stem plot of singular values.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)

    # Compute singular values
    U, S, Vt = np.linalg.svd(H)
    S_sqrt = np.sqrt(S)

    ax.stem(S_sqrt, linefmt="k-", markerfmt="ko", basefmt="k-")
    ax.set_title(f"Singular values plot for block-rows = {br}")
    ax.set_ylabel("Singular values")
    ax.set_xlabel("Model order index")
    if iter_n is not None:
        ax.set_xlim(-1, iter_n)
    ax.grid(True)
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------


def spectra_comparison(
    S_val: np.ndarray,
    S_val1: np.ndarray,
    freq: np.ndarray,
    freqlim: typing.Optional[typing.Tuple[float, float]] = None,
    nSv: typing.Union[int, str] = "all",
    fig: typing.Optional[plt.Figure] = None,
    ax: typing.Optional[plt.Axes] = None,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plots the Complex Mode Indicator Function (CMIF) based on given singular values and frequencies.

    Parameters
    ----------
    S_val : ndarray, shape (n_channel, n_channel, n_frequencies)
        A 3D array of singular values for the measured spectrum.
    S_val1 : ndarray, shape (n_channel, n_channel, n_frequencies)
        A 3D array of singular values for the synthesized (reference) spectrum.
    freq : ndarray, shape (n_frequencies,)
        Frequency vector corresponding to the third axis of S_val and S_val1.
    freqlim : tuple(float, float), optional
        Frequency limits for the x-axis as (min_freq, max_freq). If None, full range is used.
    nSv : int or "all", optional
        Number of singular values (modes) to plot. If "all", all are plotted. Defaults to "all".
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, new axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object containing the comparison plot.
    ax : matplotlib.axes.Axes
        The Matplotlib axes object containing the plotted curves.

    Raises
    ------
    ValueError
        If `nSv` is not "all" and is greater than the number of modes in S_val or S_val1.
    """
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
                "b",
                linewidth=2,
                label="measured spectrum",
            )
            ax.plot(
                freq,
                10
                * np.log10(S_val1[k, k, :] / S_val1[k, k, :][np.argmax(S_val1[k, k, :])]),
                "g",
                linewidth=2,
                label="synthesized spectrum",
            )
        else:
            ax.plot(
                freq,
                10 * np.log10(S_val[k, k, :] / S_val[0, 0, :][np.argmax(S_val[0, 0, :])]),
                "b",
                alpha=0.2,
            )
            ax.plot(
                freq,
                10
                * np.log10(S_val1[k, k, :] / S_val1[0, 0, :][np.argmax(S_val1[0, 0, :])]),
                "g",
                alpha=0.2,
            )
    ax.set_title("Singular values of spectral matrix")
    ax.set_ylabel("dB rel. to unit")
    ax.set_xlabel("Frequency [Hz]")
    if freqlim is not None:
        ax.set_xlim(freqlim[0], freqlim[1])
    ax.grid()
    ax.legend()
    # plt.show()
    return fig, ax


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
    method : str, optional
        Method for arrow plotting: '1' for quiver, '2' for line plots. Default is '1'.

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
    Plot time-series data for multiple channels, optionally showing RMS value lines.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_channels)
        Time-domain signal data for multiple channels.
    fs : float
        Sampling frequency in Hz.
    nc : int, optional
        Number of columns in the subplot grid. Determines how many subplots per row. Default is 1.
    names : list of str, optional
        Channel names for titling each subplot. If None, no titles are set. Default is None.
    unit : str, optional
        Unit label for the y-axis. Default is "unit".
    show_rms : bool, optional
        If True, plot the root mean square (RMS) value of each channel as a horizontal red line. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the subplots for each channel.
    axs : ndarray of matplotlib.axes.Axes
        2D array of axes objects for each subplot with shape (n_rows, nc).

    Notes
    -----
    - Arranges subplots in a grid with `nc` columns and as many rows as needed.
    - Shares x- and y-axes across subplots for consistent scaling.
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
    freqlim: typing.Optional[typing.Tuple[float, float]] = None,
    logscale: bool = False,
    ch_idx: typing.Union[int, typing.List[int], str] = "all",
    unit: str = "unit",
) -> typing.Tuple[typing.List[plt.Figure], typing.List[typing.List[plt.Axes]]]:
    """
    Plot channel diagnostic information: time history, autocorrelation, PSD, PDF, and normal probability plot.

    For each specified channel, creates a 3x2 grid of subplots showing:
    - Time history
    - Normalized autocorrelation
    - Power spectral density (PSD)
    - Probability density function (PDF) estimate
    - Normal probability plot

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_channels)
        Input signal data.
    fs : float
        Sampling frequency in Hz.
    nxseg : int, optional
        Number of points per segment for PSD and autocorrelation. Default is 1024.
    freqlim : tuple(float, float), optional
        Frequency axis limits for PSD as (min_freq, max_freq). If None, full range is used.
    logscale : bool, optional
        If True, PSD is plotted in dB. Otherwise, amplitude spectral density is plotted. Default is False.
    ch_idx : int, list of int, or "all", optional
        Index or list of indices of channels to plot. If "all", all channels are plotted. Default is "all".
    unit : str, optional
        Unit label for PSD y-axis (linear scale) or dB reference. Default is "unit".

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        List of figure objects, one per channel plotted.
    axs : list of list of matplotlib.axes.Axes
        Nested list of axes for each figure: [[ax0, ax1, ax2, ax3, ax4], ...]

    Notes
    -----
    - Autocorrelation is normalized by its maximum (lag 0).
    - PSD computed via SciPy's `signal.welch` with 50% overlap.
    - PDF estimated via numerical differentiation of the sorted CDF.
    - Normal probability plot compares normalized data to a standard normal distribution.
    """
    data = (
        data
        if ch_idx == "all"
        else data[:, [ch_idx]]
        if isinstance(ch_idx, int)
        else data[:, ch_idx]
    )

    ndat, nch = data.shape
    figs = []
    axs = []

    for i in range(nch):
        fig = plt.figure(figsize=(10, 8), layout="constrained")
        spec = fig.add_gridspec(3, 2)

        x = data[:, i]
        x = x - np.mean(x)
        x = x / np.std(x)
        sorted_x = np.sort(x)
        n = len(x)
        y_cdf = np.arange(1, n + 1) / n

        # Time History
        ax0 = fig.add_subplot(spec[0, :])
        ax0.plot(np.linspace(0, (n - 1) / fs, n), x, c="k")
        ax0.set_title("Time History")
        ax0.set_xlabel("Time [s]")
        ax0.set_ylabel(unit)
        ax0.grid(True)

        # Normalized Autocorrelation
        ax1 = fig.add_subplot(spec[1, 0])
        R = signal.correlate(x, x, mode="full")
        R = R[n - 1 : n - 1 + nxseg] / np.max(np.abs(R[n - 1 : n - 1 + nxseg]))
        ax1.plot(np.linspace(0, (nxseg - 1) / fs, nxseg), R, c="k")
        ax1.set_title("Normalized Autocorrelation")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Corr.")
        ax1.grid(True)

        # PSD
        ax2 = fig.add_subplot(spec[2, 0])
        freq, psd = signal.welch(
            x, fs, nperseg=nxseg, noverlap=int(nxseg * 0.5), window="hann"
        )
        if logscale:
            ax2.plot(freq, 10 * np.log10(psd), c="k")
            ax2.set_ylabel(f"dB rel. to {unit}")
        else:
            ax2.plot(freq, np.sqrt(psd), c="k")
            ax2.set_ylabel(rf"{unit}$^2$/Hz")
        if freqlim is not None:
            ax2.set_xlim(freqlim[0], freqlim[1])
        ax2.set_title("Power Spectral Density")
        ax2.set_xlabel("Frequency [Hz]")
        ax2.grid(True)

        # PDF
        ax3 = fig.add_subplot(spec[1, 1])
        xm = max(abs(sorted_x.min()), abs(sorted_x.max()))
        dx = 2 * xm / nxseg
        xi = np.linspace(-xm, xm, nxseg + 1)
        Fi = interp1d(sorted_x, y_cdf, kind="linear", fill_value="extrapolate")(xi)
        f_est = np.diff(Fi) / dx
        xf = (xi[:-1] + xi[1:]) / 2
        ax3.plot(xf, f_est, c="k")
        ax3.set_title("Probability Density Function")
        ax3.set_xlabel("Normalized data")
        ax3.set_ylabel("Probability")
        ax3.set_xlim(-xm, xm)
        ax3.grid(True)

        # Normal Probability Plot
        ax4 = fig.add_subplot(spec[2, 1])
        np.random.seed(0)
        normal_samples = np.random.randn(n)
        sorted_normal = np.sort(normal_samples)
        ax4.plot(sorted_x, sorted_normal, "k+", markersize=5)
        ax4.set_title("Normal Probability Plot")
        ax4.set_xlabel("Normalized data")
        ax4.set_ylabel("Gaussian quantiles")
        maxlim = max(
            abs(sorted_x.min()),
            abs(sorted_x.max()),
            abs(sorted_normal.min()),
            abs(sorted_normal.max()),
        )
        ax4.set_xlim(-maxlim, maxlim)
        ax4.set_ylim(-maxlim, maxlim)
        ax4.grid(True)

        fig.suptitle(f"Channel Info Plot: Channel {i}")
        figs.append(fig)
        axs.append([ax0, ax1, ax2, ax3, ax4])

    return figs, axs


# -----------------------------------------------------------------------------


def STFT(
    data: np.ndarray,
    fs: float,
    nxseg: int = 512,
    pov: float = 0.9,
    win: str = "hann",
    freqlim: typing.Optional[typing.Tuple[float, float]] = None,
    ch_idx: typing.Union[int, typing.List[int], str] = "all",
) -> typing.Tuple[typing.List[plt.Figure], typing.List[plt.Axes]]:
    """
    Compute and plot the Short-Time Fourier Transform (STFT) spectrogram for each channel.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_channels)
        Time-domain input signal data.
    fs : float
        Sampling frequency in Hz.
    nxseg : int, optional
        Window (segment) length for STFT in samples. Default is 512.
    pov : float, optional
        Proportion of overlap between segments (0 < pov < 1). Default is 0.9.
    win : str, optional
        Window function name for STFT (e.g., "hann"). Default is "hann".
    freqlim : tuple(float, float), optional
        Frequency limits for plotting as (min_freq, max_freq). If None, full range is used.
    ch_idx : int, list of int, or "all", optional
        Index or indices of channels to process. If "all", all channels are used. Default is "all".

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        List of figures created, one per channel.
    axs : list of matplotlib.axes.Axes
        List of axes objects corresponding to each figure's spectrogram plot.

    Notes
    -----
    - Uses SciPy's `signal.stft` function to compute the complex STFT.
    - Plot uses `pcolormesh` of the magnitude of the STFT.
    - If `freqlim` is provided, the frequency axis is cropped accordingly.
    """
    data = (
        data
        if ch_idx == "all"
        else data[:, [ch_idx]]
        if isinstance(ch_idx, int)
        else data[:, ch_idx]
    )

    ndat, nch = data.shape
    figs = []
    axs = []

    for i in range(nch):
        ch = data[:, i]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f"STFT Magnitude: Channel {i}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")

        noverlap = int(nxseg * pov)
        freq, t, Sxx = signal.stft(ch, fs, window=win, nperseg=nxseg, noverlap=noverlap)

        if freqlim is not None:
            idx_min = np.argmin(np.abs(freq - freqlim[0]))
            idx_max = np.argmin(np.abs(freq - freqlim[1])) + 1
            freq = freq[idx_min:idx_max]
            Sxx = Sxx[idx_min:idx_max, :]

        pcm = ax.pcolormesh(t, freq, np.abs(Sxx), shading="gouraud")
        fig.colorbar(pcm, ax=ax, label="Magnitude")
        plt.tight_layout()
        figs.append(fig)
        axs.append(ax)

    return figs, axs


# -----------------------------------------------------------------------------


def plot_mac_matrix(
    array1: np.ndarray,
    array2: np.ndarray,
    colormap: str = "plasma",
    ax: typing.Optional[plt.Axes] = None,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Compute and plot the Modal Assurance Criterion (MAC) matrix between two sets of mode shapes.

    Parameters
    ----------
    array1 : ndarray, shape (n_dofs, n_modes1)
        First set of mode shape vectors as columns.
    array2 : ndarray, shape (n_dofs, n_modes2)
        Second set of mode shape vectors as columns.
    colormap : str, optional
        Colormap name for the heatmap. Default is "plasma".
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates a new figure and axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the MAC matrix heatmap.
    ax : matplotlib.axes.Axes
        The axes object containing the heatmap.

    Raises
    ------
    ValueError
        If either `array1` or `array2` has fewer than 2 modes (columns).

    Notes
    -----
    - The MAC value between mode i of `array1` and mode j of `array2` is defined as:
      |(phi1_i^H phi2_j)|^2 / [(phi1_i^H phi1_i)(phi2_j^H phi2_j)].
    - Calls the `MAC` function from `.gen` to compute the matrix.
    """
    if array1.ndim != 2 or array2.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays with modes as columns.")
    if array1.shape[1] < 2 or array2.shape[1] < 2:
        raise ValueError("Each input array must have at least two mode columns.")

    mac_matr = MAC(array1, array2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    else:
        fig = ax.figure

    cax = ax.imshow(mac_matr, cmap=colormap, aspect="auto")
    fig.colorbar(cax, ax=ax, label="MAC value")

    n1 = array1.shape[1]
    n2 = array2.shape[1]
    x_labels = [f"Mode {i+1}" for i in range(n1)]
    y_labels = [f"Mode {j+1}" for j in range(n2)]

    ax.set_xticks(np.arange(n1))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_yticks(np.arange(n2))
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Array 1 Modes")
    ax.set_ylabel("Array 2 Modes")
    ax.set_title("MAC Matrix")
    return fig, ax


# -----------------------------------------------------------------------------


def plot_mode_complexity(
    mode_shape: typing.Union[np.ndarray, typing.List[complex]],
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plot the complexity of a mode shape on a polar coordinate plot.

    Each element of `mode_shape` is a complex number; its magnitude and phase are represented
    as arrows radiating from the origin. Principal directions at 0° and 180° are highlighted.

    Parameters
    ----------
    mode_shape : array_like, shape (n_dofs,)
        Complex-valued mode shape vector. Each element's magnitude and angle (phase) are plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the polar plot.
    ax : matplotlib.axes.Axes
        The polar axes object containing the mode shape complexity visualization.

    Notes
    -----
    - 0 degrees is set to East (positive x-axis) and direction is CCW.
    - Radial axis is scaled to accommodate maximum magnitude plus a small margin (1.1).
    - Arrows represent each complex component with an arrow from origin to (magnitude, angle).
    """
    mode_shape = np.asarray(mode_shape, dtype=complex)
    angles = np.angle(mode_shape)
    magnitudes = np.abs(mode_shape)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)  # Counterclockwise
    ax.set_rmax(magnitudes.max() * 1.1 if magnitudes.max() > 0 else 1.1)
    ax.grid(True, linestyle="--", alpha=0.5)

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
                mutation_scale=20,
            ),
        )

    # Highlight principal directions (0° and 180°)
    for pa in [0, np.pi]:
        ax.plot([pa, pa], [0, ax.get_rmax()], color="red", linestyle="--", linewidth=1)

    ax.set_yticklabels([])
    ax.set_title(
        "Mode Shape Complexity Plot", va="bottom", fontsize=14, fontweight="bold", pad=25
    )
    plt.tight_layout()
    return fig, ax
