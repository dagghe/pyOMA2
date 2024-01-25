"""
Created on Sat Oct 21 19:16:25 2023

@author: dagpa
"""
import logging

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

logger = logging.getLogger(__name__)
# =============================================================================
# PLOT ALGORITMI
# =============================================================================


# FIXME sval & feq li prende dal result dell'algoritmo ma no li trovo in run paramters
def CMIF_plot(S_val, freq, freqlim=None, nSv="all", fig=None, ax=None):
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
    ax.set_xlim(0, freqlim)
    ax.grid()
    # plt.show()
    return fig, ax


# -----------------------------------------------------------------------------


def EFDD_FIT_plot(Fn, Xi, PerPlot, freqlim=None):
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
        ax1.set_xlim(left=0, right=freqlim)

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
    ax1.set_xlim(left=0, right=freqlim)
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
    ax.set_xlim(left=0, right=freqlim)
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
    ax1.set_title("Stabilisation Chart")
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
    ax1.set_xlim(left=0, right=freqlim)
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
    ax1.set_xlim(left=0, right=freqlim)
    plt.tight_layout()
    return fig, ax1


# =============================================================================
# PLOT GEO
# =============================================================================


def plt_nodes(ax, nodes_coord, alpha=1, color="k"):
    ax.scatter(
        nodes_coord[:, 0], nodes_coord[:, 1], nodes_coord[:, 2], alpha=alpha, color=color
    )
    return ax


def plt_lines(ax, nodes_coord, lines, alpha=1, color="k"):
    for ii in range(lines.shape[0]):
        StartX, EndX = nodes_coord[lines[ii, 0]][0], nodes_coord[lines[ii, 1]][0]
        StartY, EndY = nodes_coord[lines[ii, 0]][1], nodes_coord[lines[ii, 1]][1]
        StartZ, EndZ = nodes_coord[lines[ii, 0]][2], nodes_coord[lines[ii, 1]][2]
        ax.plot([StartX, EndX], [StartY, EndY], [StartZ, EndZ], alpha=alpha, color=color)
    return ax


def plt_surf(ax, nodes_coord, surf, alpha=0.5, color="cyan"):
    xy = nodes_coord[:, :2]
    z = nodes_coord[:, 2]
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=surf)
    ax.plot_trisurf(triang, z, alpha=alpha, color=color)
    return ax


def plt_quiver(
    ax,
    nodes_coord,
    directions,
    scaleF=2,
    color="red",
    names=None,
    color_text="red",
):
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


def set_ax_options(
    ax, bg_color="w", remove_fill=True, remove_grid=True, remove_axis=True
):
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


def set_view(ax, view):
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
# plotting sensor's time histories
# =============================================================================


def plt_data(data, dt, nc=1, names=None, unit="unit", show_rms=False):
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
