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
from scipy import signal
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# =============================================================================
# PLOT ALGORITMI
# =============================================================================


def CMIF_plot(
    S_val: np.ndarray,
    freq: np.ndarray,
    freqlim: typing.Optional[typing.Tuple] = None,
    nSv: str = "all",
    fig: typing.Optional[plt.Figure] = None,
    ax: typing.Optional[plt.Axes] = None,
):
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
    tuple
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
        fig, ax = plt.subplots()
    if nSv == "all":
        nSv = S_val.shape[1]
    # Check that the number of singular value to plot is lower thant the total
    # number of singular values
    else:
        try:
            assert int(nSv) < S_val.shape[1]
        except Exception as e:
            # DA SISTEMARE!!!
            raise ValueError("ERROR") from e

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
):
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
    tuple
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


# COMMENT
def Stab_plot(
    Fn: np.ndarray,
    Lab: np.ndarray,
    step: int,
    ordmax: int,
    ordmin: int = 0,
    freqlim: typing.Optional[typing.Tuple] = None,
    hide_poles: bool = True,
    fig: typing.Optional[plt.Figure] = None,
    ax: typing.Optional[plt.Axes] = None,
):
    """
    Plot the stabilization chart for modal analysis.

    This function creates a stabilization chart, which is a graphical representation used in
    system identification to assess the stability of identified modes across different model
    orders.

    Parameters
    ----------
    Fn : ndarray
        An array containing the frequencies for each model order and identification step.
    Lab : ndarray
        An array of labels indicating the stability status of each pole. Different numbers represent
        different stability statuses such as stable pole, stable frequency, stable mode shape, etc.
    step : int
        The step size between model orders in the identification process.
    ordmax : int
        The maximum model order to be displayed on the plot.
    ordmin : int, optional
        The minimum model order to be displayed on the plot, by default 0.
    freqlim : tuple of float, optional
        A tuple defining the frequency limits for the plot. If None, includes all frequencies.
        Default is None.
    hide_poles : bool, optional
        If True, only stable poles are plotted; if False, all types of poles are plotted.
        Default is True.
    fig : matplotlib.figure.Figure, optional
        An existing matplotlib figure object to plot on. If None, a new figure is created.
        Default is None.
    ax : matplotlib.axes.Axes, optional
        An existing axes object to plot on. If None, new axes are created on the provided
        or new figure. Default is None.

    Returns
    -------
    tuple
        - fig : matplotlib.figure.Figure
            The matplotlib figure object containing the plot.
        - ax : matplotlib.axes.Axes
            The axes object with the stabilization chart.

    Notes
    -----
    The stabilization chart helps in identifying the number of physical modes and their
    stability by observing how poles behave across different model orders. Stable poles are
    typically considered as indicators of physical modes.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # Stable pole
    a = np.where(Lab == 7, Fn, np.nan)

    # Stable frequency, stable mode shape
    b = np.where(Lab == 6, Fn, np.nan)
    # Stable frequency, stable damping
    c = np.where(Lab == 5, Fn, np.nan)
    # Stable damping, stable mode shape
    d = np.where(Lab == 4, Fn, np.nan)
    # Stable damping
    e = np.where(Lab == 3, Fn, np.nan)
    # Stable mode shape
    f = np.where(Lab == 2, Fn, np.nan)
    # Stable frequency
    g = np.where(Lab == 1, Fn, np.nan)
    # new or unstable
    h = np.where(Lab == 0, Fn, np.nan)

    ax.set_title("Stabilisation Chart")
    ax.set_ylabel("Model Order")
    ax.set_xlabel("Frequency [Hz]")
    if hide_poles:
        x = a.flatten(order="f")
        y = np.array([i // len(a) for i in range(len(x))]) * step
        ax.plot(x, y, "go", markersize=7, label="Stable pole")

    else:
        x = a.flatten(order="f")
        y = np.array([i // len(a) for i in range(len(x))]) * step

        x1 = b.flatten(order="f")
        y1 = np.array([i // len(a) for i in range(len(x))]) * step

        x2 = c.flatten(order="f")
        x3 = d.flatten(order="f")
        x4 = e.flatten(order="f")
        x5 = f.flatten(order="f")
        x6 = g.flatten(order="f")
        x7 = h.flatten(order="f")

        ax.plot(x, y, "go", markersize=7, label="Stable pole")

        ax.scatter(
            x1,
            y1,
            marker="o",
            s=4,
            c="#FFFF00",
            label="Stable frequency, stable mode shape",
        )
        ax.scatter(
            x2, y1, marker="o", s=4, c="#FFFF00", label="Stable frequency, stable damping"
        )
        ax.scatter(
            x3,
            y1,
            marker="o",
            s=4,
            c="#FFFF00",
            label="Stable damping, stable mode shape",
        )
        ax.scatter(x4, y1, marker="o", s=4, c="#FFA500", label="Stable damping")
        ax.scatter(x5, y1, marker="o", s=4, c="#FFA500", label="Stable mode shape")
        ax.scatter(x6, y1, marker="o", s=4, c="#FFA500", label="Stable frequency")
        ax.scatter(x7, y1, marker="o", s=4, c="r", label="Unstable pole")

        ax.legend(loc="lower center", ncol=2)
        ax.set_ylim(ordmin, ordmax + 1)

    ax.grid()
    if freqlim is not None:
        ax.set_xlim(freqlim[0], freqlim[1])
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------


def Cluster_plot(
    Fn: np.ndarray,
    Sm: np.ndarray,
    Lab: np.ndarray,
    ordmin: int = 0,
    freqlim: typing.Optional[typing.Tuple] = None,
    hide_poles: bool = True,
):
    """
    Plots the frequency-damping clusters of the identified poles using the Stochastic Subspace Identification
    (SSI) method.

    Parameters
    ----------
    Fn : ndarray
        An array containing the frequencies of poles for each model order and identification step.
    Sm : ndarray
        An array containing the damping ratios associated with the poles in `Fn`.
    Lab : ndarray
        An array of labels indicating the stability status of each pole, where different numbers represent
        different stability statuses.
    ordmin : int, optional
        The minimum model order to be displayed on the plot. Default is 0.
    freqlim : tuple of float, optional
        The upper frequency limit for the plot. If None, includes all frequencies. Default is None.
    hide_poles : bool, optional
        If True, only stable poles are plotted. If False, all types of poles are plotted. Default is True.

    Returns
    -------
    tuple
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        ax1 : matplotlib.axes.Axes
            The axes object with the stabilization chart.
    """
    # Stable pole
    a = np.where(Lab == 7, Fn, np.nan)
    aa = np.where(Lab == 7, Sm, np.nan)

    # Stable frequency, stable mode shape
    b = np.where(Lab == 6, Fn, np.nan)
    bb = np.where(Lab == 6, Sm, np.nan)
    # Stable frequency, stable damping
    c = np.where(Lab == 5, Fn, np.nan)
    cc = np.where(Lab == 5, Sm, np.nan)
    # Stable damping, stable mode shape
    d = np.where(Lab == 4, Fn, np.nan)
    dd = np.where(Lab == 4, Sm, np.nan)
    # Stable damping
    e = np.where(Lab == 3, Fn, np.nan)
    ee = np.where(Lab == 3, Sm, np.nan)
    # Stable mode shape
    f = np.where(Lab == 2, Fn, np.nan)
    ff = np.where(Lab == 2, Sm, np.nan)
    # Stable frequency
    g = np.where(Lab == 1, Fn, np.nan)
    gg = np.where(Lab == 1, Sm, np.nan)
    # new or unstable
    h = np.where(Lab == 0, Fn, np.nan)
    hh = np.where(Lab == 0, Sm, np.nan)

    fig, ax1 = plt.subplots()
    ax1.set_title("Frequency-damping clustering")
    ax1.set_ylabel("Damping")
    ax1.set_xlabel("Frequency [Hz]")
    if hide_poles:
        x = a.flatten(order="f")
        y = aa.flatten(order="f")
        ax1.plot(x, y, "go", markersize=7, label="Stable pole")

    else:
        x = a.flatten(order="f")
        y = aa.flatten(order="f")

        x1 = b.flatten(order="f")
        y1 = bb.flatten(order="f")

        x2 = c.flatten(order="f")
        y2 = cc.flatten(order="f")

        x3 = d.flatten(order="f")
        y3 = dd.flatten(order="f")
        x4 = e.flatten(order="f")
        y4 = ee.flatten(order="f")
        x5 = f.flatten(order="f")
        y5 = ff.flatten(order="f")
        x6 = g.flatten(order="f")
        y6 = gg.flatten(order="f")
        x7 = h.flatten(order="f")
        y7 = hh.flatten(order="f")

        ax1.plot(x, y, "go", markersize=7, label="Stable pole")

        ax1.scatter(
            x1,
            y1,
            marker="o",
            s=4,
            c="#FFFF00",
            label="Stable frequency, stable mode shape",
        )
        ax1.scatter(
            x2, y2, marker="o", s=4, c="#FFFF00", label="Stable frequency, stable damping"
        )
        ax1.scatter(
            x3,
            y3,
            marker="o",
            s=4,
            c="#FFFF00",
            label="Stable damping, stable mode shape",
        )
        ax1.scatter(x4, y4, marker="o", s=4, c="#FFA500", label="Stable damping")
        ax1.scatter(x5, y5, marker="o", s=4, c="#FFA500", label="Stable mode shape")
        ax1.scatter(x6, y6, marker="o", s=4, c="#FFA500", label="Stable frequency")
        ax1.scatter(x7, y7, marker="o", s=4, c="r", label="Unstable pole")

        ax1.legend(loc="lower center", ncol=2)

    ax1.grid()
    if freqlim is not None:
        ax1.set_xlim(freqlim[0], freqlim[1])
    plt.tight_layout()
    return fig, ax1


# -----------------------------------------------------------------------------


def Sval_plot(
    H: np.ndarray,
    br: int,
    iter_n: int = None,
    fig: typing.Optional[plt.Figure] = None,
    ax: typing.Optional[plt.Axes] = None,
):
    """ """
    if fig is None and ax is None:
        fig, ax = plt.subplots(tight_layout=True)

    # SINGULAR VALUE DECOMPOSITION
    U1, S1, V1_t = np.linalg.svd(H)
    S1rad = np.sqrt(S1)

    ax.stem(S1rad, linefmt="k-")

    ax.set_title(f"Singular values plot, for block-rows(time shift) = {br}")
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
):
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
):
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
):
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
):
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
    ax.quiver(
        xs0, ys0, zs0, (xs1 - xs0), (ys1 - ys0), (zs1 - zs0), length=scaleF, color=color
    )
    if names is not None:
        for ii, nam in enumerate(names):
            ax.text(
                Points_f[ii, 0], Points_f[ii, 1], Points_f[ii, 2], f"{nam}", color="red"
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
):
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
            [[0.2 * scaleF, 0, 0], [0, 0.2 * scaleF, 0], [0, 0, 0.2 * scaleF]]
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


def set_view(ax: plt.Axes, view: str):
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
):
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
    tuple
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
    fig, axs = plt.subplots(figsize=(8, 6), nrows=nr, ncols=nc, sharex=True, sharey=True)
    fig.suptitle("Time Histories of all channels")

    kk = 0  # iterator for the dataset
    for ii in range(nr):
        # if there are more than one columns
        if nc != 1:
            # loop over the columns
            for jj in range(nc):
                ax = axs[ii, jj]
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
            ax = axs[ii]
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
        ax.grid()
    plt.tight_layout()

    return fig, ax


# -----------------------------------------------------------------------------


def plt_ch_info(
    data: np.ndarray,
    fs: float,
    nxseg: int = 1024,
    freqlim: typing.Optional[typing.Tuple] = None,
    logscale: bool = False,
    ch_idx: typing.Union[int, typing.List[int], str] = "all",
    unit: str = "unit",
):
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

    return fig, [ax0, ax10, ax20, ax11, ax21]


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
):
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
        ax.set_title("STFT Magnitude")
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
