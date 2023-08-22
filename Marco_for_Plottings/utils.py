import os
import numpy as np
import pandas as pd
import pickle
from matplotlib.text import Annotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch


def create_results_folder(RESULTS_PATH):
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH) 

def import_data(DATA_FILE: str):
    _, file_extension = os.path.splitext(DATA_FILE)

    if file_extension == '.txt':
        data = pd.read_csv(DATA_FILE, header=None, sep="\s+", index_col=False) 
        data = data.to_numpy()
    elif file_extension == '.csv':
        data = pd.read_csv(DATA_FILE, header=None, sep=";", index_col=False) 
        data = data.to_numpy()
    elif file_extension == '.xlsx' or file_extension == '.xls':
        data = pd.read_excel(DATA_FILE, header=None, index_col=False) 
        data = data.to_numpy()
    elif file_extension == '.pkl':
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"File extension not recognized: {file_extension} : Supported file format .txt, .csv, .pkl")
    return data

def plot3Dframe(nodes, connectivity, kwargs_plot_lines, kwargs_plot_markers, annotatepoints=False, figsize=(5,6), hold_on=False):
    num_frames = connectivity.shape[0]
    if hold_on:
        _fig=plt.gcf()
        ax=plt.gca()
    else:
        _fig = plt.figure(figsize=figsize,facecolor='white')
        ax = plt.axes(projection="3d")
        setattr(ax, 'annotate3D', annotate3d)
        setattr(ax, 'arrow3D', arrow3d)
    for k in range(num_frames):
        x1 = nodes[connectivity[k, 0] - 1, 0]
        y1 = nodes[connectivity[k, 0] - 1, 1]
        z1 = nodes[connectivity[k, 0] - 1, 2]
        x2 = nodes[connectivity[k, 1] - 1, 0]
        y2 = nodes[connectivity[k, 1] - 1, 1]
        z2 = nodes[connectivity[k, 1] - 1, 2]
        xx = [x1, x2]; yy = [y1, y2]; zz = [z1, z2]
        ax.plot3D(xx, yy, zz, **kwargs_plot_lines)
    for i in range(nodes.shape[0]):
        xs = nodes[i, 0]; ys = nodes[i, 1]; zs = nodes[i, 2]
        ax.scatter(xs, ys, zs, **kwargs_plot_markers)
        if annotatepoints:
            ax.annotate3D(ax,text=f'P{i + 1}', xyz=(xs, ys, zs), xytext=(3, 3), textcoords='offset points')
    return _fig,ax

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def do_3d_projection(self, renderer = None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        # super().draw(renderer)
        return np.min(zs)


def arrow3d(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)



class Annotation3D(Annotation):

    def __init__(self, text, xyz=(0,0,0), *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

def annotate3d(ax, text, xyz, *args, **kwargs):
    annotation = Annotation3D(text, xyz=xyz, *args, **kwargs)
    ax.add_artist(annotation)



def plot2Dframe(nodes, connectivity, kwargs_plot_lines, kwargs_plot_markers, annotatepoints=False, figsize=(5,6), hold_on=False):
    num_frames = connectivity.shape[0]
    if hold_on:
        _fig=plt.gcf()
        ax=plt.gca()
    else:
        _fig, ax = plt.subplots(figsize=figsize,facecolor='white')
        # ax = plt.axes(projection="3d")
        # setattr(ax, 'annotate3D', annotate3d)
        # setattr(ax, 'arrow3D', arrow3d)
    for k in range(num_frames):
        x1 = nodes[connectivity[k, 0] - 1, 0]
        y1 = nodes[connectivity[k, 0] - 1, 1]
        # z1 = nodes[connectivity[k, 0] - 1, 2]
        x2 = nodes[connectivity[k, 1] - 1, 0]
        y2 = nodes[connectivity[k, 1] - 1, 1]
        # z2 = nodes[connectivity[k, 1] - 1, 2]
        xx = [x1, x2]; yy = [y1, y2] #; zz = [z1, z2]
        ax.plot(xx, yy, **kwargs_plot_lines)
    for i in range(nodes.shape[0]):
        xs = nodes[i, 0]; ys = nodes[i, 1] #; zs = nodes[i, 2]
        ax.scatter(xs, ys, **kwargs_plot_markers)
        if annotatepoints:
            ax.annotate(ax,text=f'P{i + 1}', xy=(xs, ys), xytext=(3, 3), textcoords='offset points')
    return _fig,ax