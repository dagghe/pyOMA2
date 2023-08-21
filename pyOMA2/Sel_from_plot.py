# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 16:31:54 2023

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

from . import tools

# =============================================================================
# PLOTTING CLASS
# =============================================================================

class SelFromPlot():
    def __init__(self, Model, plot, freqlim=None):
        """ 
        Bla bla bla
        """
        self.Model = Model
        
        if freqlim is not None:
            self.freq_max = freqlim
        else:
            self.freq_max = self.Model.samp_freq / 2  # Nyquist frequency

        self.shift_is_held = False

        self.samp_freq = self.Model.samp_freq
        self.freqs = self.Model.Results["FDD"]["freqs"]

        self.root = tk.Tk()

        self.Model.sel_freq = []
        if plot == "SSI" or plot == "pLSCF":
            self.show_legend = 0
            self.hide_poles = 1
            self.S_val = self.Model.Results["FDD"]["S_val"]

            self.Model.sel_xi = []
            self.Model.sel_phi = []
            self.Model.sel_lam = []
            self.Model.pole_ind = []

            self.root.title('Stabilisation Chart')

        elif plot == "FDD":
            self.Model.freq_ind = []
            self.root.title('Singular Values of PSD matrix')

        self.fig = Figure(figsize=(16, 8))

        # Create axes
        self.ax2 = self.fig.add_subplot(111)
        self.ax2.grid(True)
        if plot == "SSI" or plot == "pLSCF":
            self.ax1 = self.ax2.twinx()
        
        # Tkinter menu
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Save figure', command=self.save_this_figure)
        menubar.add_cascade(label='File', menu=filemenu)

        if plot == "SSI" or plot == "pLSCF":
            self.plot = plot
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


        # Program execution
        if plot == "SSI" or plot == "pLSCF":
            self.get_stab(plot)
            self.plot_stab(plot)
        elif plot == "FDD":
            self.plot_svPSD()


        # Integrate matplotlib figure
        canvas = FigureCanvasTkAgg(self.fig, self.root)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        NavigationToolbar2Tk(canvas, self.root)


        # Connecting functions to event manager
        self.fig.canvas.mpl_connect('key_press_event', 
                                    lambda x: self.on_key_press(x))
        self.fig.canvas.mpl_connect('key_release_event', 
                                    lambda x: self.on_key_release(x))
        if plot == "SSI" or plot == "pLSCF":
            self.fig.canvas.mpl_connect('button_press_event', 
                                        lambda x: self.on_click_SSI(x, plot))

        elif plot == "FDD":
            self.fig.canvas.mpl_connect('button_press_event', 
                                        lambda x: self.on_click_FDD(x))

        self.root.protocol("WM_DELETE_WINDOW", lambda: self.on_closing())
        self.root.mainloop()

#------------------------------------------------------------------------------

    def plot_svPSD(self, update_ticks=False):

        S_val = self.Model.Results["FDD"]["S_val"]

        if not update_ticks:
            self.ax2.clear()
            self.ax2.grid(True)

            for ii in range(self.Model.Nch):
                self.ax2.plot(self.freqs[:], 10*np.log10(S_val[ii, ii]))
            
            df = self.Model.Results["FDD"]["df"]
            self.ax2.set_xlim(left=0, right=self.freq_max)
            self.ax2.xaxis.set_major_locator(MultipleLocator(self.freq_max / 10))
            self.ax2.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            self.ax2.xaxis.set_minor_locator(MultipleLocator(self.freq_max / 100))
            self.ax2.set_title("Singular values plot - (Freq. res. ={0})".format(df))
            self.ax2.set_xlabel("Frequency [Hz]")
            self.ax2.set_ylabel(r"dB $[g^2/Hz]$")

            self.line, = self.ax2.plot(self.Model.sel_freq, 
                         [10*np.log10(S_val[0,0,i]*1.05) for i in self.Model.freq_ind]
                                       , 'kv', markersize=8)

            plt.tight_layout()

        else:
            self.line.set_xdata(np.asarray(self.Model.sel_freq))  # update data
            self.line.set_ydata([10*np.log10(S_val[0,0,i]*1.05) for i in 
                                 self.Model.freq_ind])

            plt.tight_layout()

        self.fig.canvas.draw()

#------------------------------------------------------------------------------

    def get_closest_freq(self):
        """
        On-the-fly selection of the closest poles.        
        """
        freq = self.Model.Results["FDD"]["freqs"]
        # Find closest frequency
        sel = np.argmin(np.abs(freq - self.x_data_pole))

        self.Model.sel_freq.append(self.Model.Results["FDD"]["freqs"][sel])
        self.Model.freq_ind.append(sel)
        self.sort_selected_poles()


#------------------------------------------------------------------------------

    def get_stab(self, plot, err_fn=0.01, err_xi=0.05, err_ms=0.02):
        """
        
        """

        if plot == "SSI":
            ordmax = self.Model.SSI_ordmax
            ordmin = self.Model.SSI_ordmin
            Fr = self.Model.Results["SSIcov"]['Fn_poles']
            Sm = self.Model.Results["SSIcov"]['xi_poles']
            Ms = self.Model.Results["SSIcov"]['Phi_poles']
            
            self.Lab = tools._stab_SSI(Fr, Sm, Ms, ordmin, ordmax, 
                            err_fn=err_fn, err_xi=err_xi, err_ms=err_ms)

        elif plot == "pLSCF":
            ordmax = self.Model.pLSCF_ordmax
            Fr = self.Model.Results["pLSCF"]['Fn_poles']
            Sm = self.Model.Results["pLSCF"]['xi_poles']
            nch = self.Model.Nch
            self.Lab = tools._stab_pLSCF(Fr, Sm, ordmax, 
                            err_fn=err_fn, err_xi=err_xi, nch=nch)

#------------------------------------------------------------------------------

    def plot_stab(self, plot, update_ticks=False):

        S_val = self.Model.Results["FDD"]["S_val"]

        if not update_ticks:
            self.ax1.clear()
            self.ax2.clear()
            self.ax2.grid(True)

            for ii in range(2):
                self.ax2.plot(self.freqs[:], 10*np.log10(S_val[ii, ii]),"gray")

            self.ax1.set_xlim(left=0, right=self.freq_max)
            self.ax1.xaxis.set_major_locator(MultipleLocator(self.freq_max / 10))
            self.ax1.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            self.ax1.xaxis.set_minor_locator(MultipleLocator(self.freq_max / 100))
            self.ax1.set_xlabel("Frequency [Hz]")
            self.ax2.set_ylabel(r"dB $[g^2/Hz]$")

            #-----------------------
            if plot == "SSI":
                Fr = self.Model.Results["SSIcov"]['Fn_poles']
                Lab = self.Lab
                
                # Stable pole
                a = np.where(Lab == 7, Fr, np.nan)
                # Stable frequency, stable mode shape
                b = np.where(Lab == 6, Fr, np.nan)
                # Stable frequency, stable damping
                c = np.where(Lab == 5, Fr, np.nan)
                # Stable damping, stable mode shape
                d = np.where(Lab == 4, Fr, np.nan)
                # Stable damping
                e = np.where(Lab == 3, Fr, np.nan)
                # Stable mode shape
                f = np.where(Lab == 2, Fr, np.nan)
                # Stable frequency
                g = np.where(Lab == 1, Fr, np.nan)
                # new or unstable
                h = np.where(Lab == 0, Fr, np.nan)

                if self.hide_poles:
                    x = a.flatten(order='f')
                    y = np.array([i//len(a) for i in range(len(x))])

                    self.ax1.plot(x, y, 'go', markersize=7, label="Stable pole")

                    self.line, = self.ax1.plot(self.Model.sel_freq, 
                                               [i for i in self.Model.pole_ind]
                                                , 'kx', markersize=10)

                else:
                    x = a.flatten(order='f')
                    x1 = b.flatten(order='f')
                    x2 = c.flatten(order='f')
                    x3 = d.flatten(order='f')
                    x4 = e.flatten(order='f')
                    x5 = f.flatten(order='f')
                    x6 = g.flatten(order='f')
                    x7 = h.flatten(order='f')

                    y = np.array([i//len(a) for i in range(len(x))])

                    self.ax1.plot(x, y, 'go', markersize=7, label="Stable pole")
                    
                    self.ax1.scatter(x1, y, 
                                       marker='o', s=4, c='#FFFF00',
                                       label="Stable frequency, stable mode shape")
                    self.ax1.scatter(x2, y,
                                       marker='o', s=4, c='#FFFF00',
                                       label="Stable frequency, stable damping")
                    self.ax1.scatter(x3, y, 
                                       marker='o', s=4, c='#FFFF00',
                                       label="Stable damping, stable mode shape")
                    self.ax1.scatter(x4, y, 
                                       marker='o', s=4, c='#FFA500',
                                       label="Stable damping")
                    self.ax1.scatter(x5, y, 
                                       marker='o', s=4, c='#FFA500',
                                       label="Stable mode shape")
                    self.ax1.scatter(x6, y, 
                                       marker='o', s=4, c='#FFA500',
                                       label="Stable frequency")
                    self.ax1.scatter(x7, y, 
                                       marker='o', s=4, c='r',
                                       label="Unstable pole")

                    self.line, = self.ax1.plot(self.Model.sel_freq, 
                                               [i for i in self.Model.pole_ind]
                                                , 'kx', markersize=10)

            #-----------------------
            elif plot == "pLSCF":
                Fr = self.Model.Results["pLSCF"]['Fn_poles']
                Lab = self.Lab
                
                # Stable pole
                a = np.where(Lab == 3, Fr, np.nan)
                # Stable damping
                b = np.where(Lab == 2, Fr, np.nan)
                # Stable frequency
                c = np.where(Lab == 1, Fr, np.nan)
                # Unstable pole
                d = np.where(Lab == 0, Fr, np.nan)

                if self.hide_poles:
                    x = a.flatten(order='f')
                    y = np.array([i//len(a) for i in range(len(x))])
                    
                    self.ax1.plot(x, y, 'go', markersize=7, label="Stable pole")
                    
                    self.line, = self.ax1.plot(self.Model.sel_freq, 
                                               [i for i in self.Model.pole_ind]
                                                , 'kx', markersize=10)

                else:
                    x = a.flatten(order='f')
                    x1 = b.flatten(order='f')
                    x2 = c.flatten(order='f')
                    x3 = d.flatten(order='f')

                    y = np.array([i//len(a) for i in range(len(x))])

                    self.ax1.plot(x, y, 'go', markersize=7, label="Stable pole")
                    
                    self.ax1.scatter(x1, y, 
                                       marker='o', s=4, c='#FFFF00',
                                       label="Stable damping")
                    self.ax1.scatter(x2, y,
                                       marker='o', s=4, c='#FFFF00',
                                       label="Stable frequency")
                    self.ax1.scatter(x3, y, 
                                       marker='o', s=4, c='r',
                                       label="Unstable pole")

                    self.line, = self.ax1.plot(self.Model.sel_freq, 
                                               [i for i in self.Model.pole_ind]
                                                , 'kx', markersize=10)

            #-----------------------
            if self.show_legend:
                self.pole_legend = self.ax1.legend(loc='lower center', ncol=4, frameon=True)

            plt.tight_layout()

        else:
            self.line.set_xdata(np.asarray(self.Model.sel_freq))  # update data
            self.line.set_ydata([i for i in self.Model.pole_ind])

            plt.tight_layout()

        self.fig.canvas.draw()


#------------------------------------------------------------------------------

    def get_closest_pole(self, plot):
        """
        On-the-fly selection of the closest poles.        
        """

        if plot == "SSI":
            Fr = self.Model.Results["SSIcov"]['Fn_poles']
            Sm = self.Model.Results["SSIcov"]['xi_poles']
            Ms = self.Model.Results["SSIcov"]['Phi_poles']
        elif plot == "pLSCF":
            Fr = self.Model.Results["pLSCF"]['Fn_poles']
            Sm = self.Model.Results["pLSCF"]['xi_poles']
            Ls = self.Model.Results["pLSCF"]['lam_poles']

        y_ind = int(np.argmin(np.abs(np.arange(Fr.shape[1])-self.y_data_pole)))  # Find closest pole order index
        x = Fr[:, y_ind]
        # Find closest frequency index
        sel = np.nanargmin(np.abs(x - self.x_data_pole))

        self.Model.pole_ind.append(y_ind)
        self.Model.sel_freq.append(Fr[sel, y_ind])
        self.Model.sel_xi.append(Sm[sel, y_ind])

        if plot == "SSI":
            self.Model.sel_phi.append(Ms[sel, y_ind, :])
        if plot == "pLSCF":
            self.Model.sel_lam.append(Ls[sel, y_ind])

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
                del self.Model.sel_freq[-1]  # delete last point
                del self.Model.freq_ind[-1]

                self.plot_svPSD()
            except:
                pass

        elif event.button == 2 and self.shift_is_held:
            i = np.argmin(np.abs(self.Model.sel_freq - event.xdata))
            try:
                del self.Model.sel_freq[i]
                del self.Model.freq_ind[i]

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
                del self.Model.sel_freq[-1]  # delete last point
                del self.Model.sel_xi[-1]
                del self.Model.pole_ind[-1]
                if plot == "SSI":
                    del self.Model.sel_phi[-1]
                
                self.plot_stab(plot)
            except:
                pass

        elif event.button == 2 and self.shift_is_held:
            i = np.argmin(np.abs(self.Model.sel_freq - event.xdata))
            try:
                del self.Model.sel_freq[i]
                del self.Model.sel_xi[i]
                del self.Model.pole_ind[i]
                if plot == "SSI":
                    del self.Model.sel_phi[i]

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
        _ = np.argsort(self.Model.sel_freq)
        self.Model.sel_freq = list(np.array(self.Model.sel_freq)[_])


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
