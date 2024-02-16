"""
Plotting Utility Functions module.
Part of the pyOMA2 package.
Author:
Dag Pasca
"""

import logging

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy import signal, stats

logger = logging.getLogger(__name__)

# =============================================================================
# PLOT ALGORITMI
# =============================================================================


def CMIF_plot(S_val, freq, freqlim=None, nSv="all", fig=None, ax=None):
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
            int(nSv) < S_val.shape[1]
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


def EFDD_FIT_plot(Fn, Xi, PerPlot, freqlim=None):
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
            freq, 10 * np.log10(Sval[0, 0] / Sval[0, 0][np.argmax(Sval[0, 0])]), c="b"
        )
        ax1.plot(
            fsval,
            10 * np.log10(SDOFbell[idSV].real / SDOFbell[np.argmax(SDOFbell)]),
            c="r",
            label="SDOF bell",
        )
        ax1.set_title("SDOF Bell function")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel(r"dB $[V^2/Hz]$")
        if freqlim is not None:
            ax1.set_xlim(freqlim[0], freqlim[1])

        ax1.legend()

        # Plot 2
        ax2.plot(time[:], normSDOFcorr)
        ax2.set_title("Auto-correlation Function")
        ax2.set_xlabel("Time lag[s]")
        ax2.set_ylabel("Normalized correlation")

        # PLOT 3 (PORTION for FIT)
        ax3.plot(time[: minmax_fit_idx[-1]], normSDOFcorr[: minmax_fit_idx[-1]])
        ax3.scatter(time[minmax_fit_idx], normSDOFcorr[minmax_fit_idx])
        ax3.set_title("Portion for fit")
        ax3.set_xlabel("Time lag[s]")
        ax3.set_ylabel("Normalized correlation")

        # PLOT 4 (FIT)
        ax4.scatter(np.arange(len(minmax_fit_idx)), delta)
        ax4.plot(np.arange(len(minmax_fit_idx)), lam / 2 * np.arange(len(minmax_fit_idx)))

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

        plt.tight_layout()

        figs.append(fig)
        axs.append([ax1, ax2, ax3, ax4])

    return figs, axs


# -----------------------------------------------------------------------------

# COMMENT
def Stab_pLSCF_plot(Fn, Lab, ordmax, freqlim=None, hide_poles=True, Sval=None, nSv=None):
    """
    Plots a stabilization chart for the pLSCF (polyreference Least Squares Complex Frequency domain) method.

    Parameters
    ----------
    Fn : ndarray
        An array containing the frequencies of poles for each model order and identification run.
    Lab : ndarray
        An array of labels indicating the stability of each pole (e.g., stable pole,
        stable damping, stable frequency, unstable pole).
    ordmax : int
        The maximum model order to be displayed on the plot.
    freqlim : tuple of float, optional
        The upper frequency limit for the plot. If None, includes all frequencies. Default is None.
    hide_poles : bool, optional
        If True, only stable poles are plotted. If False, all types of poles are plotted. Default is True.
    Sval : ndarray, optional
        Singular values to be plotted on a twin axis. Not implemented yet. Default is None.
    nSv : int, optional
        Number of singular values to be plotted. Not implemented yet. Default is None.

    Returns
    -------
    tuple
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        ax1 : matplotlib.axes.Axes
            The axes object with the stabilization chart.
    """
    # TO DO: Add sval plot on twin ax

    # Stable pole
    a = np.where(Lab == 3, Fn, np.nan)
    # Stable damping
    b = np.where(Lab == 2, Fn, np.nan)
    # Stable frequency
    c = np.where(Lab == 1, Fn, np.nan)
    # Unstable pole
    d = np.where(Lab == 0, Fn, np.nan)

    fig, ax1 = plt.subplots()
    ax1.set_title("Stabilisation Chart")
    ax1.set_ylabel("Model Order")
    ax1.set_xlabel("Frequency [Hz]")

    if hide_poles:
        x = a.flatten(order="f")
        y = np.array([i // len(a) for i in range(len(x))])

        ax1.plot(x, y, "go", markersize=7, label="Stable pole")

    else:
        x = a.flatten(order="f")
        x1 = b.flatten(order="f")
        x2 = c.flatten(order="f")
        x3 = d.flatten(order="f")

        y = np.array([i // len(a) for i in range(len(x))])

        ax1.plot(x, y, "go", markersize=7, label="Stable pole")

        ax1.scatter(x1, y, marker="o", s=4, c="#FFFF00", label="Stable damping")
        ax1.scatter(x2, y, marker="o", s=4, c="#FFFF00", label="Stable frequency")
        ax1.scatter(x3, y, marker="o", s=4, c="r", label="Unstable pole")

        ax1.legend(loc="lower center", ncol=2)
        ax1.set_ylim(0, ordmax + 1)

    ax1.grid()
    if freqlim is not None:
        ax1.set_xlim(freqlim[0], freqlim[1])
    plt.tight_layout()
    return fig, ax1


# -----------------------------------------------------------------------------

# COMMENT
def Stab_SSI_plot(
    Fn,
    Lab,
    step,
    ordmax,
    ordmin=0,
    freqlim=None,
    hide_poles=True,
    fig=None,
    ax=None,
):
    """
    Plots a stabilization chart for the Stochastic Subspace Identification (SSI) method.

    Parameters
    ----------
    Fn : ndarray
        An array containing the frequencies of poles for each model order and identification step.
    Lab : ndarray
        An array of labels indicating the stability status of each pole, where different numbers represent
        different stability statuses (e.g., stable pole, stable frequency, stable mode shape).
    step : int
        The step size between model orders in the identification process.
    ordmax : int
        The maximum model order to be displayed on the plot.
    ordmin : int, optional
        The minimum model order to be displayed on the plot. Default is 0.
    freqlim : tuple of float, optional
        The upper frequency limit for the plot. If None, includes all frequencies. Default is None.
    hide_poles : bool, optional
        If True, only stable poles are plotted. If False, all types of poles are plotted. Default is True.
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
            The axes object with the stabilization chart.
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
        y = np.array([i // len(a) for i in range(len(x))]) * step + ordmin
        ax.plot(x, y, "go", markersize=7, label="Stable pole")

    else:
        x = a.flatten(order="f")
        y = np.array([i // len(a) for i in range(len(x))]) * step + ordmin

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


def Cluster_SSI_plot(
    Fn,
    Sm,
    Lab,
    ordmin=0,
    freqlim=None,
    hide_poles=True,
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


def Cluster_pLSCF_plot(
    Fn,
    Sm,
    Lab,
    ordmax,
    ordmin=0,
    freqlim=None,
    hide_poles=True,
):
    """
    Plots the frequency-damping clusters of the identified poles using the polyreference
    Least Squares Complex Frequency (pLSCF) method.

    Parameters
    ----------
    Fn : ndarray
        An array containing the frequencies of poles for each model order.
    Sm : ndarray
        An array containing the damping ratios associated with the poles in `Fn`.
    Lab : ndarray
        An array of labels indicating the stability status of each pole, where different numbers
        represent different stability statuses.
    ordmax : int
        The maximum model order to be displayed on the plot.
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
    a = np.where(Lab == 3, Fn, np.nan)
    aa = np.where(Lab == 3, Sm, np.nan)
    # Stable damping
    b = np.where(Lab == 2, Fn, np.nan)
    bb = np.where(Lab == 2, Sm, np.nan)
    # Stable frequency
    c = np.where(Lab == 1, Fn, np.nan)
    cc = np.where(Lab == 1, Sm, np.nan)
    # Unstable pole
    d = np.where(Lab == 0, Fn, np.nan)
    dd = np.where(Lab == 0, Sm, np.nan)

    fig, ax1 = plt.subplots()
    ax1.set_title("Stabilisation Chart")
    ax1.set_ylabel("Model Order")
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

        ax1.plot(x, y, "go", markersize=7, label="Stable pole")

        ax1.scatter(x1, y1, marker="o", s=4, c="#FFFF00", label="Stable damping")
        ax1.scatter(x2, y2, marker="o", s=4, c="#FFFF00", label="Stable frequency")
        ax1.scatter(x3, y3, marker="o", s=4, c="r", label="Unstable pole")

        ax1.legend(loc="lower center", ncol=2)
        ax1.set_ylim(0, ordmax + 1)

    ax1.grid()
    if freqlim is not None:
        ax1.set_xlim(freqlim[0], freqlim[1])
    plt.tight_layout()
    return fig, ax1


# =============================================================================
# PLOT GEO
# =============================================================================


def plt_nodes(ax, nodes_coord, alpha=1, color="k"):
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
        Color or list of colors for the nodes. Default is "k" (black).

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object with the nodes plotted.

    Note
    -----
    This function is designed to work with 3D plots and adds node representations to an existing 3D plot.
    """
    ax.scatter(
        nodes_coord[:, 0], nodes_coord[:, 1], nodes_coord[:, 2], alpha=alpha, color=color
    )
    return ax


# -----------------------------------------------------------------------------


def plt_lines(ax, nodes_coord, lines, alpha=1, color="k"):
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
        Color or list of colors for the lines. Default is "k" (black).

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object with the lines plotted.

    Note
    -----
    This function is designed to work with 3D plots and adds line representations between
    nodes in an existing 3D plot.
    """
    for ii in range(lines.shape[0]):
        StartX, EndX = nodes_coord[lines[ii, 0]][0], nodes_coord[lines[ii, 1]][0]
        StartY, EndY = nodes_coord[lines[ii, 0]][1], nodes_coord[lines[ii, 1]][1]
        StartZ, EndZ = nodes_coord[lines[ii, 0]][2], nodes_coord[lines[ii, 1]][2]
        ax.plot([StartX, EndX], [StartY, EndY], [StartZ, EndZ], alpha=alpha, color=color)
    return ax


# -----------------------------------------------------------------------------


def plt_surf(ax, nodes_coord, surf, alpha=0.5, color="cyan"):
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
    ax,
    nodes_coord,
    directions,
    scaleF=2,
    color="red",
    names=None,
    color_text="red",
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
        ii = 0
        for nam in names:
            ax.text(
                Points_f[ii, 0], Points_f[ii, 1], Points_f[ii, 2], f"{nam}", color="red"
            )
            ii += 1
    return ax


# -----------------------------------------------------------------------------


def set_ax_options(
    ax, bg_color="w", remove_fill=True, remove_grid=True, remove_axis=True
):
    """
    Configures various display options for a given matplotlib axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to be configured.
    bg_color : str, optional
        Background color for the axes panes. Default is "w" (white).
    remove_fill : bool, optional
        If True, removes the fill from the axes panes. Default is True.
    remove_grid : bool, optional
        If True, removes the grid from the axes. Default is True.
    remove_axis : bool, optional
        If True, turns off the axis lines, labels, and ticks. Default is True.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object with the applied configurations.

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
    return ax


# -----------------------------------------------------------------------------


def set_view(ax, view):
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


def plt_data(data, dt, nc=1, names=None, unit="unit", show_rms=False):
    """
    Plots time series data for multiple channels, with an option to include the Root Mean Square (RMS)
      of each signal.

    Parameters
    ----------
    data : ndarray
        A 2D array with dimensions [number of data points, number of channels], representing signal data
        for multiple channels.
    dt : float
        Time interval between data points.
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
    timef = Ndat / (1 / dt)  # final time value
    time = np.linspace(0, timef - dt, Ndat)  # time array

    nr = round(Nch / nc)  # number of rows in the subplot
    fig, axs = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True)
    fig.suptitle("Plot of the Time Histories of all channels")

    kk = 0  # iterator for the dataset
    for ii in range(nr):
        # if there are more than one columns
        if nc != 1:
            # loop over the columns
            for jj in range(nc):
                ax = axs[ii, jj]
                try:
                    # while kk < data.shape[1]
                    ax.plot(time, data[:, kk])
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
                        )
                        ax.legend()
                except Exception as e:
                    logger.exception(e)
                    # if k > data.shape[1]
                    pass
                kk += 1
        # if there is only 1 column
        else:
            ax = axs[ii]
            ax.plot(time, data[:, kk])
            if names is not None:
                ax.set_title(f"{names[kk]}")
            if ii == nr - 1:
                ax.set_xlabel("time [s]")
            if show_rms is True:
                ax.plot(
                    time,
                    np.repeat(a_rmss[kk], len(time)),
                    label=f"arms={a_rmss[kk]:.3f}",
                )
                ax.legend()
            ax.set_ylabel(f"{unit}")
            kk += 1
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------


def plt_ch_info(
    data,
    fs,
    ch_idx="all",
    ch_names=None,
    freqlim=None,
    logscale=True,
    nxseg=None,
    pov=0.0,
    window="boxcar",
):
    """
    Generates plots for time history, power spectral density (PSD), and kernel density estimation
    (KDE) for each channel in a multi-channel dataset.

    Parameters
    ----------
    data : ndarray
        A 2D array with dimensions [number of data points, number of channels], representing signal data for
        multiple channels.
    fs : float
        The sampling frequency of the data.
    ch_idx : list of int or str, optional
        Indices of channels to plot. If "all", plots all channels. Default is "all".
    ch_names : list of str, optional
        Names of the channels, used for titling plots if provided. Default is None.
    freqlim : tuple of float, optional
        The frequency range (lower, upper) for the PSD plot. If None, includes all frequencies.
        Default is None.
    logscale : bool, optional
        If True, uses a logarithmic scale for the PSD plot. If False, uses a linear scale. Default is True.
    nxseg : int, optional
        Number of data points per segment for PSD calculation using the Welch method. If None, uses full
        length of the channel data. Default is None.
    pov : float, optional
        Proportion of overlap between segments in PSD calculation. Default is 0.0.
    window : str, optional
        Type of window function used in PSD calculation. Default is "boxcar".

    Returns
    -------
    tuple
        figs : list of matplotlib.figure.Figure
            A list of matplotlib figure objects for each channel.
        axs : list of matplotlib.axes.Axes
            A list of axis objects for the generated subplots.

    Note
    -----
    Each figure includes three subplots: time history, PSD, and KDE for the respective channel.
    Allows for comprehensive analysis of each channel's signal characteristics.
    """
    if ch_idx != "all":
        data = data[:, ch_idx]
    ndat, nch = data.shape
    figs = []
    for ii in range(nch):
        fig = plt.figure(figsize=(8, 6), layout="constrained")
        spec = fig.add_gridspec(2, 2)

        # select channel
        ch = data[:, ii]

        # plot TH
        ax0 = fig.add_subplot(spec[0, :])
        ax0.plot(ch)
        ax0.set_xlabel("Time [s]")
        ax0.set_title("Time History")
        ax0.set_ylabel("Unit")

        # plot psd
        if nxseg is None:
            nxseg = len(ch)
        noverlap = nxseg * pov
        # FFT = np.fft.rfft(ch,nxseg)
        # freq = np.fft.rfftfreq(nxseg,dt)
        freq, psd = signal.welch(
            ch, fs, nperseg=nxseg, noverlap=noverlap, window=window, scaling="spectrum"
        )
        ax10 = fig.add_subplot(spec[1, 0])

        if logscale is True:
            ax10.plot(freq, 10 * np.log10(psd / psd[np.argmax(psd)]))
        elif logscale is False:
            ax10.plot(freq, np.sqrt(psd))
        if freqlim is not None:
            ax10.set_xlim(freqlim[0], freqlim[1])
        ax10.set_xlabel("Frequency [Hz]")
        ax10.set_ylabel("dB rel. to unit")
        ax10.set_title("PSD")

        # KDE of TH
        ax11 = fig.add_subplot(spec[1, 1])
        kde = stats.gaussian_kde(ch)
        x_grid = np.linspace(ch.min(), ch.max(), 200)
        ax11.plot(x_grid, kde.evaluate(x_grid))
        ax11.set_title("KDE on channel data")

        if ch_names is not None:
            fig.suptitle(f"{ch_names[ii]}")

        axs = [ax0, ax10, ax11]
        figs.append(fig)
    return figs, axs
