# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:56:19 2024

@author: dpa
"""


import os
import glob
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import tkinter as tk

from pyoma2.functions import (FDD_funct, Gen_funct, SSI_funct, plot_funct,
                              pLSCF_funct)

from pyoma2.algorithm.result import BaseResult, FDDResult, EFDDResult, SSIResult
# from .result import BaseResult

from pyoma2.algorithm.run_params import BaseRunParams, SSIRunParams, FDDRunParams
# from .run_params import BaseRunParams
from pyoma2.functions.plot_funct import CMIF_plot, Stab_SSI_plot


# =============================================================================
# PLOTTING CLASS
# =============================================================================

class SelFromPlot():
    def __init__(self, AlgoName, freqlim=None):
        """ 
        Bla bla bla
        """
        # self.AlgoName = AlgoName
        # CHECK PER VEDERE CHE ALGORITMO 
        if AlgoName == FDD_algo or EFDD_algo:
            self.plot = "FDD"
        elif AlgoName == SSIdat_algo or SSIcov_algo:
            self.plot = "SSI"
        # elif AlgoName == pLSCF_algo:
        #     self.plot = "pLSCF"
        # Importare frequenza campionamento
        self.fs = BaseRunParams.fs

        if freqlim is not None:
            self.freqlim = freqlim
        else:
            self.freqlim = self.fs / 2  # Nyquist frequency

        # inizializzo TK e check su shift
        self.shift_is_held = False
        self.root = tk.Tk()

        self.sel_freq = []

        if self.plot == "SSI" or self.plot == "pLSCF":
            self.show_legend = 0
            self.hide_poles = 1

            self.pole_ind = []

            self.root.title('Stabilisation Chart')
        # # per aggiungere plot di sottofondo
        #     self.ax1 = self.ax2.twinx()

        elif self.plot == "FDD":
            self.freq_ind = []
            self.root.title('Singular Values of PSD matrix')

        # Create fig and ax
        self.fig = Figure(figsize=(16, 8))
        self.ax2 = self.fig.add_subplot(111)
        # self.ax2.grid(True)


        # Tkinter menu
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Save figure', command=self.save_this_figure)
        menubar.add_cascade(label='File', menu=filemenu)

        if self.plot == "SSI" or self.plot == "pLSCF":

            hidepolesmenu = tk.Menu(menubar, tearoff=0)
            hidepolesmenu.add_command(label='Show unstable poles', 
                command=lambda: (self.toggle_hide_poles(0), self.toggle_legend(1)))
            hidepolesmenu.add_command(label='Hide unstable poles', 
                command=lambda: (self.toggle_hide_poles(1), self.toggle_legend(0)))
            menubar.add_cascade(label="Show/Hide Unstable Poles", 
                                menu=hidepolesmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label='Help', command=self.show_help)
        menubar.add_cascade(label='Help', menu=helpmenu)

        self.root.config(menu=menubar)

# =============================================================================
#         # Program execution
        if self.plot == "SSI" or self.plot == "pLSCF":
            self.plot_stab(self.plot)
        elif self.plot == "FDD":
            self.plot_svPSD()
# =============================================================================

        # Integrate matplotlib figure
        canvas = FigureCanvasTkAgg(self.fig, self.root)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        NavigationToolbar2Tk(canvas, self.root)


        # Connecting functions to event manager
        self.fig.canvas.mpl_connect('key_press_event', 
                                    lambda x: self.on_key_press(x))
        self.fig.canvas.mpl_connect('key_release_event', 
                                    lambda x: self.on_key_release(x))
        if self.plot == "SSI" or self.plot == "pLSCF":
            self.fig.canvas.mpl_connect('button_press_event', 
                                        lambda x: self.on_click_SSI(x, self.plot))

        elif self.plot == "FDD":
            self.fig.canvas.mpl_connect('button_press_event', 
                                        lambda x: self.on_click_FDD(x))

        self.root.protocol("WM_DELETE_WINDOW", lambda: self.on_closing())
        self.root.mainloop()


        # ATTENZIONE!!!!
        # CI VA IL RETURN QUI? VORREI IL RETURN AL CLOSE WINDOW EVENT
        # if self.order is not None:
            # return self.sel_freq, self.order
        # else:
            # return self.sel_freq

#------------------------------------------------------------------------------

    def plot_svPSD(self, update_ticks=False):

        freq = FDDResult.freq
        S_val = FDDResult.S_val

        if not update_ticks:
            self.ax2.clear()
            self.ax2.grid(True)

            CMIF_plot(S_val, freq, freqlim=self.freqlim, # come fare con nSv?
                      fig=self.fig, ax=self.ax2)
# =============================================================================
            # ATTENZIONE DA RIVEDERE
            self.MARKER, = self.ax2.plot(self.sel_freq, # ATTENZIONE
                         [10*np.log10(S_val[0,0,i]*1.05) for i in self.freq_ind]
                                       , 'kv', markersize=8)

            plt.tight_layout()

        else:
            # ATTENZIONE DA RIVEDERE
            self.MARKER.set_xdata(np.asarray(self.sel_freq))  # update data
            self.MARKER.set_ydata([10*np.log10(S_val[0,0,i]*1.05) for i in 
                                 self.freq_ind])

            plt.tight_layout()

        self.fig.canvas.draw()

#------------------------------------------------------------------------------

    def get_closest_freq(self):
        """
        On-the-fly selection of the closest poles.        
        """
        
        freq = FDDResult.freq
        # Find closest frequency
        sel = np.argmin(np.abs(freq - self.x_data_pole))

        self.freq_ind.append(sel)
        self.sel_freq.append(freq[sel])
        self.sort_selected_poles()

#------------------------------------------------------------------------------

    def plot_stab(self, plot, update_ticks=False):

        # S_val = self.AlgoName.Results[f"FDD_{simnum}"]["S_val"]

        freqlim = self.freqlim
        hide_poles = self.hide_poles

        Fn = SSIResult.Fn
        Lab = SSIResult.Lab
        Fn = SSIResult.Fn
        Fn = SSIResult.Fn

        step = SSIRunParams.step
        ordmin = SSIRunParams.ordmin
        ordmax = SSIRunParams.ordmax
        step = SSIRunParams.step

        if not update_ticks:
            self.ax1.clear()
            self.ax2.clear()
            self.ax2.grid(True)

            #-----------------------
            if plot == "SSI":
                Stab_SSI_plot(
                    Fn, Lab, step, ordmax, ordmin=ordmin, freqlim=freqlim, 
                    hide_poles=hide_poles, 
                    # DA FARE 
                    Sval=None, nSv=None, 
                    fig=self.fig, ax=self.ax1,
                )

                self.MARKER, = self.ax1.plot(self.sel_freq, 
                                           [i for i in self.AlgoName.pole_ind]
                                            , 'kx', markersize=10)

            # ATTENZIONE DA FARE
            # #-----------------------
            # elif plot == "pLSCF":
            #     Fr = self.AlgoName.Results[f"pLSCF_{simnum}"]['Fn_poles']
            #     Lab = self.Lab

            #     if self.hide_poles:
            #         x = a.flatten(order='f')
            #         y = np.array([i//len(a) for i in range(len(x))])

            #         self.ax1.plot(x, y, 'go', markersize=7, label="Stable pole")

            #         self.MARKER, = self.ax1.plot(self.AlgoName.sel_freq, 
            #                                    [i for i in self.AlgoName.pole_ind]
            #                                     , 'kx', markersize=10)

            #     else:
            #         # PLOT ALL

            #         self.MARKER, = self.ax1.plot(self.AlgoName.sel_freq, 
            #                                    [i for i in self.AlgoName.pole_ind]
            #                                     , 'kx', markersize=10)

            # #-----------------------
            if self.show_legend:
                self.pole_legend = self.ax1.legend(loc='lower center', ncol=4, frameon=True)

            plt.tight_layout()

        else:
            self.MARKER.set_xdata(np.asarray(self.sel_freq))  # update data
            self.MARKER.set_ydata([i for i in self.pole_ind])

            plt.tight_layout()

        self.fig.canvas.draw()


#------------------------------------------------------------------------------

    def get_closest_pole(self, plot):
        """
        On-the-fly selection of the closest poles.        
        """

        if plot == "SSI":
            Fn_poles = SSIResult.Fn_poles

        # elif plot == "pLSCF":
        #     Fr = self.AlgoName.Results[f"pLSCF_{simnum}"]['Fn_poles']
        #     Sm = self.AlgoName.Results[f"pLSCF_{simnum}"]['xi_poles']

        y_ind = int(np.argmin(np.abs(np.arange(Fn_poles.shape[1])-self.y_data_pole)))  # Find closest pole order index
        x = Fn_poles[:, y_ind]
        # Find closest frequency index
        sel = np.nanargmin(np.abs(x - self.x_data_pole))

        self.pole_ind.append(y_ind)
        self.sel_freq.append(Fn_poles[sel, y_ind])

        self.sort_selected_poles()

#------------------------------------------------------------------------------

    def on_click_FDD(self, event):
        # on button 1 press (left mouse button) + SHIFT is held
        if event.button == 1 and self.shift_is_held:
            self.y_data_pole = [event.ydata]
            self.x_data_pole = event.xdata

            self.get_closest_freq()

            self.plot_svPSD()

        # On button 3 press (left mouse button)
        elif event.button == 3 and self.shift_is_held:
            try:
                del self.sel_freq[-1]  # delete last point
                del self.freq_ind[-1]

                self.plot_svPSD()
            except:
                pass

        elif event.button == 2 and self.shift_is_held:
            i = np.argmin(np.abs(self.sel_freq - event.xdata))
            try:
                del self.sel_freq[i]
                del self.freq_ind[i]

                self.plot_svPSD()
            except:
                pass

        if self.shift_is_held:
            self.plot_svPSD(update_ticks=True)
            
#------------------------------------------------------------------------------ 

    def on_click_SSI(self, event, plot):
        # on button 1 press (left mouse button) + SHIFT is held
        if event.button == 1 and self.shift_is_held:
            self.y_data_pole = [event.ydata]
            self.x_data_pole = event.xdata

            self.get_closest_pole(plot)

            self.plot_stab(plot)

        # On button 3 press (left mouse button)
        elif event.button == 3 and self.shift_is_held:
            try:
                del self.sel_freq[-1]  # delete last point
                del self.pole_ind[-1]

                self.plot_stab(plot)
            except:
                pass

        elif event.button == 2 and self.shift_is_held:
            i = np.argmin(np.abs(self.sel_freq - event.xdata))
            try:
                del self.sel_freq[i]
                del self.pole_ind[i]

                self.plot_stab(plot)
            except:
                pass

        if self.shift_is_held:
            self.plot_stab(plot, update_ticks=True)

#------------------------------------------------------------------------------

    def on_key_press(self, event):
        """Function triggered on key press (SHIFT)."""
        if event.key == 'shift':
            self.shift_is_held = True


    def on_key_release(self, event):
        """Function triggered on key release (SHIFT)."""
        if event.key == 'shift':
            self.shift_is_held = False


    def on_closing(self):
        self.root.destroy()


    def toggle_legend(self, x):
        if x:
            self.show_legend = 1
        else:
            self.show_legend = 0

        self.plot_stab(self.plot)


    def toggle_hide_poles(self, x):
        if x:
            self.hide_poles = 1
        else:
            self.hide_poles = 0

        self.plot_stab(self.plot)


    def sort_selected_poles(self):
        _ = np.argsort(self.sel_freq)
        self.sel_freq = list(np.array(self.sel_freq)[_])


    def show_help(self):
        lines = [
            'Pole selection help',
            ' ',
            '- Select a pole: SHIFT + LEFT mouse button',
            '- Deselect a pole: SHIFT + RIGHT mouse button',
            '- Deselect the closest pole (frequency wise): SHIFT + MIDDLE mouse button',
        ]
        tk.messagebox.showinfo('Picking poles', '\n'.join(lines))


    def save_this_figure(self):
        filename = 'pole_chart_'
        directory = 'pole_figures'

        if not os.path.exists(directory):
            os.mkdir(directory)

        files = glob.glob(directory + '/*.png')
        i = 1
        while True:
            f = os.path.join(directory, filename + f'{i:0>3}.png')
            if f not in files:
                break
            i += 1

        self.fig.savefig(f)
