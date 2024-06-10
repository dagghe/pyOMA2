"""
Module for interactive plots.
Part of the pyOMA2 package.
Authors:
Dag Pasca
Diego Margoni

This module provides the SelFromPlot class for interactive selection of poles in
operational modal analysis plots. It supports FDD, SSI, and pLSCF methods and
integrates matplotlib plots into a Tkinter window for user interaction.
"""

from __future__ import annotations

import glob
import logging
import os
import tkinter as tk
import typing
from typing import Literal, Tuple

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

if typing.TYPE_CHECKING:
    from pyoma2.algorithm import BaseAlgorithm

from pyoma2.functions.plot_funct import CMIF_plot, Stab_plot

logger = logging.getLogger(__name__)

# =============================================================================
# PLOTTING CLASS
# =============================================================================


class SelFromPlot:
    """
    A class for interactive selection of poles.

    This class integrates matplotlib plots into a Tkinter window, enabling users to interactively
    select or deselect poles using mouse clicks and keyboard shortcuts. It supports FDD, SSI, and
    pLSCF methods for operational modal analysis. The design and functionality of this class is
    strongly inspired by the pyEMA module [ZBGS20]_.

    Attributes
    ----------
    algo : BaseAlgorithm
        An instance of a base algorithm class that provides necessary data for plotting.
    freqlim : tuple, optional
        Upper frequency limit for the plot, defaults to half the Nyquist frequency if not provided.
    plot : str
        Type of plot to be displayed. Supported values are "FDD", "SSI", and "pLSCF".
    root : tkinter.Tk
        Root widget of the Tkinter application.
    sel_freq : list
        List of selected frequencies.
    shift_is_held : bool
        Flag to track if the SHIFT key is held down during mouse interactions.
    fig : matplotlib.figure.Figure
        Matplotlib Figure object for plotting.
    ax2 : matplotlib.axes.Axes
        Axes object for the figure.
    MARKER : matplotlib.lines.Line2D
        Line2D object for displaying selected points on the plot.
    show_legend : int
        Flag to control the visibility of the legend in the plot.
    hide_poles : int
        Flag to control the visibility of unstable poles in the plot.

    Methods
    -------
    __init__(algo: BaseAlgorithm, freqlim=None, plot: Literal["FDD", "SSI", "pLSCF"] = "FDD"):
        Initializes the SelFromPlot class with the specified algorithm, frequency limit, and plot type.
    plot_svPSD(update_ticks=False):
        Plots the Singular Values of the Power Spectral Density matrix for FDD analysis.
    get_closest_freq():
        Selects the frequency closest to the mouse click location for FDD plots.
    plot_stab(plot, update_ticks=False):
        Plots the stabilization chart for SSI or pLSCF methods.
    get_closest_pole(plot):
        Selects the pole closest to the mouse click location for SSI or pLSCF plots.
    on_click_FDD(event):
        Handles mouse click events for FDD plots.
    on_click_SSI(event, plot):
        Handles mouse click events for SSI or pLSCF plots.
    on_key_press(event):
        Handles key press events (SHIFT key for selecting poles).
    on_key_release(event):
        Handles key release events.
    on_closing():
        Handles the closing event of the Tkinter window.
    toggle_legend(x):
        Toggles the visibility of the legend in the plot.
    toggle_hide_poles(x):
        Toggles the visibility of unstable poles in the plot.
    sort_selected_poles():
        Sorts the selected poles based on their frequencies.
    show_help():
        Displays a help dialog with instructions for selecting poles.
    save_this_figure():
        Saves the current plot to a file.
    """

    def __init__(
        self,
        algo: BaseAlgorithm,
        freqlim: Tuple[float, float] = None,
        plot: Literal["FDD", "SSI", "pLSCF"] = "FDD",
    ):
        """
        Initializes the SelFromPlot class with specified algorithm, frequency limit, and plot type.

        Parameters
        ----------
        algo : BaseAlgorithm
            An instance of a base algorithm class providing necessary data for plotting.
        freqlim : tuple, optional
            Upper frequency limit for the plot, defaults to half the Nyquist frequency if not provided.
        plot : str, optional
            Type of plot to be displayed. Supported values are "FDD", "SSI", and "pLSCF". Default is "FDD".
        """
        self.algo = algo
        self.plot = plot
        self.fs = self.algo.fs
        self.freqlim = freqlim if freqlim is not None else (0.0, self.fs / 2)
        self.shift_is_held = False
        self.sel_freq = []

        if self.plot in ("SSI", "pLSCF"):
            self.show_legend = 0
            self.hide_poles = 1
            self.pole_ind = []
        elif self.plot == "FDD":
            self.freq_ind = []

        self._initialize_gui()

        if self.plot in ("SSI", "pLSCF"):
            self.plot_stab(self.plot)
        elif self.plot == "FDD":
            self.plot_svPSD()

        self.root.mainloop()

        if self.plot in ("SSI", "pLSCF"):
            self.result = self.sel_freq, self.pole_ind
        elif self.plot == "FDD":
            self.result = self.sel_freq, None

    def _initialize_gui(self):
        """
        Initializes the Tkinter GUI components.
        """
        self.root = tk.Tk()
        self.root.title(
            "Stabilisation Chart"
            if self.plot in ("SSI", "pLSCF")
            else "Singular Values of PSD matrix"
        )

        self.fig = Figure(figsize=(12, 6), tight_layout=True)
        self.ax2 = self.fig.add_subplot(111)

        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save figure", command=self.save_this_figure)
        menubar.add_cascade(label="File", menu=filemenu)

        if self.plot in ("SSI", "pLSCF"):
            hidepolesmenu = tk.Menu(menubar, tearoff=0)
            hidepolesmenu.add_command(
                label="Show unstable poles",
                command=lambda: (self.toggle_hide_poles(0), self.toggle_legend(1)),
            )
            hidepolesmenu.add_command(
                label="Hide unstable poles",
                command=lambda: (self.toggle_hide_poles(1), self.toggle_legend(0)),
            )
            menubar.add_cascade(label="Show/Hide Unstable Poles", menu=hidepolesmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Help", command=self.show_help)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.root.config(menu=menubar)

        canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.ax2.grid()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
        NavigationToolbar2Tk(canvas, self.root)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)
        if self.plot in ("SSI", "pLSCF"):
            self.fig.canvas.mpl_connect(
                "button_press_event", lambda event: self.on_click_SSI(event, self.plot)
            )
        elif self.plot == "FDD":
            self.fig.canvas.mpl_connect("button_press_event", self.on_click_FDD)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def plot_svPSD(self, update_ticks: bool = False):
        """
        Plots the Singular Values of the Power Spectral Density matrix for FDD analysis.

        Parameters
        ----------
        update_ticks : bool, optional
            Flag indicating whether to update tick marks for selected frequencies. Default is False.
        """
        freq = self.algo.result.freq
        S_val = self.algo.result.S_val

        if not update_ticks:
            self.ax2.clear()
            CMIF_plot(S_val, freq, freqlim=self.freqlim, fig=self.fig, ax=self.ax2)
            # Compute the y-values for the selected frequencies
            marker_y_values = [
                10
                * np.log10(
                    (S_val[0, 0, i] / S_val[0, 0, np.argmax(S_val[0, 0, :])]) * 1.25
                )
                for i in self.freq_ind
            ]
            (self.MARKER,) = self.ax2.plot(
                self.sel_freq, marker_y_values, "kv", markersize=8
            )
        else:
            marker_y_values = [
                10
                * np.log10(
                    (S_val[0, 0, i] / S_val[0, 0, np.argmax(S_val[0, 0, :])]) * 1.25
                )
                for i in self.freq_ind
            ]
            self.MARKER.set_xdata(np.asarray(self.sel_freq))
            self.MARKER.set_ydata(marker_y_values)

        self.ax2.grid(True)  # Ensure grid is always displayed
        self.fig.canvas.draw_idle()  # Use draw_idle for better performance and interaction handling

    def get_closest_freq(self):
        """
        Selects the frequency closest to the mouse click location for FDD plots.
        """
        freq = self.algo.result.freq
        sel = np.argmin(np.abs(freq - self.x_data_pole))
        self.freq_ind.append(sel)
        self.sel_freq.append(freq[sel])
        self.sort_selected_poles()

    def plot_stab(self, plot: Literal["SSI", "pLSCF"], update_ticks: bool = False):
        """
        Plots the stabilization chart for SSI or pLSCF methods.

        Parameters
        ----------
        plot : str
            Type of plot to be displayed ("SSI" or "pLSCF").
        update_ticks : bool, optional
            Flag indicating whether to update tick marks for selected poles. Default is False.
        """
        freqlim = self.freqlim
        hide_poles = self.hide_poles

        Fn = self.algo.result.Fn_poles
        Lab = self.algo.result.Lab

        step = self.algo.run_params.step
        ordmin = self.algo.run_params.ordmin
        ordmax = self.algo.run_params.ordmax

        if not update_ticks:
            self.ax2.clear()
            Stab_plot(
                Fn,
                Lab,
                step,
                ordmax,
                ordmin=ordmin,
                freqlim=freqlim,
                hide_poles=hide_poles,
                fig=self.fig,
                ax=self.ax2,
            )
            (self.MARKER,) = self.ax2.plot(
                self.sel_freq, self.pole_ind, "kx", markersize=10
            )
        else:
            self.MARKER.set_xdata(np.asarray(self.sel_freq))
            self.MARKER.set_ydata(self.pole_ind)

        self.ax2.grid(True)  # Ensure grid is always displayed
        self.fig.canvas.draw_idle()  # Use draw_idle for better performance and interaction handling

    def get_closest_pole(self, plot: Literal["SSI", "pLSCF"]):
        """
        Selects the pole closest to the mouse click location for SSI or pLSCF plots.

        Parameters
        ----------
        plot : str
            Type of plot ("SSI" or "pLSCF") for which the pole is being selected.
        """
        if plot in ("SSI", "pLSCF"):
            Fn_poles = self.algo.result.Fn_poles

        y_ind = int(np.argmin(np.abs(np.arange(Fn_poles.shape[1]) - self.y_data_pole)))
        x = Fn_poles[:, y_ind]
        sel = np.nanargmin(np.abs(x - self.x_data_pole))

        self.pole_ind.append(y_ind)
        self.sel_freq.append(Fn_poles[sel, y_ind])

        self.sort_selected_poles()

    def on_click_FDD(self, event):
        """
        Handles mouse click events for FDD plots.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event triggered on the plot.
        """
        if event.button == 1 and self.shift_is_held:
            self.y_data_pole = [event.ydata]
            self.x_data_pole = event.xdata
            self.get_closest_freq()
            self.plot_svPSD()

        elif event.button == 3 and self.shift_is_held:
            if self.sel_freq and self.freq_ind:
                self.sel_freq.pop()
                self.freq_ind.pop()
                self.plot_svPSD()

        elif event.button == 2 and self.shift_is_held:
            if self.sel_freq and self.freq_ind:
                i = np.argmin(np.abs(self.sel_freq - event.xdata))
                self.sel_freq.pop(i)
                self.freq_ind.pop(i)
                self.plot_svPSD()

    def on_click_SSI(self, event, plot: Literal["SSI", "pLSCF"]):
        """
        Handles mouse click events for SSI or pLSCF plots.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event triggered on the plot.
        plot : str
            Type of plot ("SSI" or "pLSCF") where the event occurred.
        """
        if event.button == 1 and self.shift_is_held:
            self.y_data_pole = [event.ydata]
            self.x_data_pole = event.xdata
            self.get_closest_pole(plot)
            self.plot_stab(plot)

        elif event.button == 3 and self.shift_is_held:
            if self.sel_freq and self.pole_ind:
                self.sel_freq.pop()
                self.pole_ind.pop()
                self.plot_stab(plot)

        elif event.button == 2 and self.shift_is_held:
            if self.sel_freq and self.pole_ind:
                i = np.argmin(np.abs(self.sel_freq - event.xdata))
                self.sel_freq.pop(i)
                self.pole_ind.pop(i)
                self.plot_stab(plot)

    def on_key_press(self, event):
        """
        Handles key press events for interactive pole selection.

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            The key event triggered on the plot.
        """
        if event.key == "shift":
            self.shift_is_held = True

    def on_key_release(self, event):
        """
        Handles key release events.

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            The key event triggered on the plot.
        """
        if event.key == "shift":
            self.shift_is_held = False

    def on_closing(self):
        """
        Handles the closing event of the Tkinter window.
        """
        self.root.quit()
        self.root.destroy()

    def toggle_legend(self, x: int):
        """
        Toggles the visibility of the legend in the plot.

        Parameters
        ----------
        x : int
            Flag indicating whether to show (1) or hide (0) the legend.
        """
        self.show_legend = bool(x)
        self.plot_stab(self.plot)

    def toggle_hide_poles(self, x: int):
        """
        Toggles the visibility of unstable poles in the plot.

        Parameters
        ----------
        x : int
            Flag indicating whether to hide (1) or show (0) unstable poles.
        """
        self.hide_poles = bool(x)
        self.plot_stab(self.plot)

    def sort_selected_poles(self):
        """
        Sorts the selected poles based on their frequencies.
        """
        sorted_indices = np.argsort(self.sel_freq)
        self.sel_freq = list(np.array(self.sel_freq)[sorted_indices])

    def show_help(self):
        """
        Displays a help dialog with instructions for selecting poles.
        """
        lines = [
            "Pole selection help",
            " ",
            "- Select a pole: SHIFT + LEFT mouse button",
            "- Deselect a pole: SHIFT + RIGHT mouse button",
            "- Deselect the closest pole (frequency wise): SHIFT + MIDDLE mouse button",
        ]
        tk.messagebox.showinfo("Picking poles", "\n".join(lines))

    def save_this_figure(self):
        """
        Saves the current plot to a file.
        """
        filename = "pole_chart_"
        directory = "pole_figures"

        if not os.path.exists(directory):
            os.mkdir(directory)

        files = glob.glob(directory + "/*.png")
        i = 1
        while True:
            f = os.path.join(directory, f"{filename}{i:03}.png")
            if f not in files:
                break
            i += 1

        self.fig.savefig(f)
