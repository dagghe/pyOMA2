# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 10:08:24 2023

@author: dagpa
"""

import os
import glob
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk

from . import tools


def animate_scatters(iteration, Data, scatters):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)
    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    if Data.shape[1] == 3:
        for i in range(Data.shape[0]):
            scatters[i]._offsets3d = (Data[i,0:1,iteration], Data[i,1:2,iteration], Data[i,2:,iteration])
    else:
        for i in range(Data.shape[0]):
            scatters[i].set_offsets((Data[i,0,iteration], Data[i,1,iteration]))
    return scatters

# =============================================================================
# PLOTTING CLASS
# =============================================================================

class AniMode():
    def __init__(self, Model, method, mode_numb):
        """ 
        Bla bla bla
        """
        self.Model = Model
        
        self.mode_numb = mode_numb
        self.nodes_coord = self.Model.nodes_coord
        self.directions = self.Model.directions
        self.Fn = self.Model.Results[f"{method}"]["Fn"]
        self.PHI = self.Model.Results[f"{method}"]["Phi"].real
        self.phi = self.PHI[:, self.mode_numb]
        self.MCF = tools.MCF(self.phi)
        # get the sign
        self.sign = np.sign(self.directions)
        # get the direction
        self.dir = np.abs(self.directions)-np.ones(len(self.directions))
        self.dir = self.dir.astype(int)
        
        self.PlotModesDim = self.Model.PlotModesDim
        
        self.root = tk.Tk()
        self.root.title(f'Mode n*{self.mode_numb}, MCF = {self.MCF}')
        self.fig = Figure(figsize=(6, 8))
        
        canvas = FigureCanvasTkAgg(self.fig, self.root)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        
        if self.PlotModesDim == "3D":
            self.plot3D()
        
        else:
            self.plot2D()
            
        # Integrate matplotlib figure
        NavigationToolbar2Tk(canvas, self.root)
        canvas.draw()
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.on_closing())
        self.root.mainloop()
    
    def plot3D(self):
        nr_iter=200
        Data = np.repeat(self.nodes_coord[:,:, np.newaxis], nr_iter, axis=2).astype(float)
        # Loop through the n* of channels
        for jj in range(self.Model.Nch):
            # take into account the sign 
            ch_jj = self.sign[jj]*self.PHI[:, self.mode_numb][jj]

            traj1 = np.linspace(0, ch_jj, int(nr_iter/4))
            traj2 = np.concatenate([traj1,traj1[::-1]])
            
            traj3 = np.linspace(0, -ch_jj, int(nr_iter/4))
            traj4 = np.concatenate([traj3,traj3[::-1]])
            
            traj5= np.concatenate([traj2,traj4])
            # add to the correct direction
            Data[jj, self.dir[jj], :] += traj5
            
        self.ax = self.fig.add_subplot(111, projection='3d')
        # Setting the axes properties
        self.ax.set_xlim3d([np.min(Data[:,0,:]), np.max(Data[:,0,:])])
        self.ax.set_xlabel('X')

        self.ax.set_ylim3d([np.min(Data[:,1,:]), np.max(Data[:,1,:])])
        self.ax.set_ylabel('Y')

        self.ax.set_zlim3d([np.min(Data[:,2,:]), np.max(Data[:,2,:])])
        self.ax.set_zlabel('Z')
        
        # Initialize scatters
        scatters = [ self.ax.scatter(Data[i,0:1,0], Data[i,1:2,0], Data[i,2:,0]) for i in range(Data.shape[0]) ]
        
        self.ani = animation.FuncAnimation(self.fig, animate_scatters, nr_iter, fargs=(Data, scatters),
                                           interval=20, blit=False, repeat=True)
        plt.tight_layout()
        plt.show()
    
    def plot2D(self):
        nr_iter=200
        Data = np.repeat(self.nodes_coord[:,:, np.newaxis], nr_iter, axis=2).astype(float)
        # Loop through the n* of channels
        for jj in range(self.Model.Nch):
            # print(jj)
            ch_jj = self.sign[jj]*self.PHI[:, self.mode_numb][jj]

            traj1 = np.linspace(0, ch_jj, int(nr_iter/4))
            traj2 = np.concatenate([traj1,traj1[::-1]])
            
            traj3 = np.linspace(0, -ch_jj, int(nr_iter/4))
            traj4 = np.concatenate([traj3,traj3[::-1]])
            
            traj5= np.concatenate([traj2,traj4])
            
            Data[jj, int(self.dir[jj]), :] += traj5
            
        self.ax = self.fig.add_subplot(111)
        
        # Setting the axes properties
        self.ax.set_xlim([np.min(Data[:,0,:]), np.max(Data[:,0,:])])
        self.ax.set_xlabel('X')

        self.ax.set_ylim([np.min(Data[:,1,:]), np.max(Data[:,1,:])])
        self.ax.set_ylabel('Y')

        # Initialize scatters
        scatters = [ self.ax.scatter(Data[i,0,0], Data[i,1,0]) for i in range(Data.shape[0]) ]
        
        self.ani = animation.FuncAnimation(self.fig, animate_scatters, nr_iter, fargs=(Data, scatters),
                                           interval=20, blit=False, repeat=True)
        plt.tight_layout()
        plt.show()


    def on_closing(self):
        self.root.destroy()
