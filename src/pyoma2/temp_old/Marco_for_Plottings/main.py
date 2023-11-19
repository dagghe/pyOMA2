import matplotlib
# import PyOMA as oma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MonSetup import MonSetup
from utils import *
from Wmodel import Wmodel

matplotlib.rcParams["font.family"] = "cmu serif"
matplotlib.rcParams["text.usetex"] = True
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


# user gives the node number of the wireframe model and which dof is monitored among x,y, and z
# length of list equal to the num of columns of data file
SETUP_LAYOUT = ["37x", "37y", "38x", "38z", "30x"]
DATA_PATH = "Setup_1.txt"
MOD_NODES_COORD_FILE = "nodes.csv"
MOD_CONNECTIVITY_FILE = "connectivity.csv"
OUTPATH = "RESULTS"
FS = 50  # sampling frequency in Hz
BR = 15  # block rows argument, i.e. time shift for SSI algorithm
SCFCARROW_LIST = [10, 10, 10]  # Scale factor for arrows of every mode shape
SCFC_LIST = [10, 10, 10]  # Scale factors for every mode shape
AX_DIST = [10, 10, 10]
np.set_printoptions(suppress=True)


create_results_folder(OUTPATH)

data = pd.read_csv(
    DATA_PATH, header=None, sep=r"\s+", index_col=False, skiprows=18
).to_numpy()
#  remove time column
data = np.copy(data[:, 1:])

Setup1 = MonSetup(data, SETUP_LAYOUT, FS)
Setup1.detrend()
Setup1.decimate(q=2)
# signalfigures = Setup1.plot_signals(OUTPATH, lw=1)
Setup1.svd_psd(OUTPATH)
Setup1.runssicov(OUTPATH, BR)
Setup1.overlap_stabdiag_svpsd(OUTPATH, selection_criteria=3, num_sv=4)

FreQ = [1.6, 3.9, 5.9]  # identified peaks

Setup1.get_modal_prop(method="EFDD", FreQ=FreQ, save_to_file_path=OUTPATH, plot=True)
Setup1.get_modal_prop(method="SSIcov", FreQ=FreQ, save_to_file_path=OUTPATH, deltaf=0.1)
Setup1.crossmac_ssi_fdd(save_to_file_path=OUTPATH)

print("ok")

nodes_model = import_data(MOD_NODES_COORD_FILE).astype("float64")
connectivity_model = import_data(MOD_CONNECTIVITY_FILE)

modelSetup1modes = []
figs3DSetup1 = []
for ii in range(len(FreQ)):
    modelSetup1modes.append(
        Wmodel(
            nodes_model,
            connectivity_model,
            Setup1.get_mode_shape(method="efdd", num=ii + 1),
        )
    )  # num = 1,2,...

    kwargsWmodel = {"color": "#e3e3e3", "linestyle": "solid"}
    kwargsmarkers = {"color": "#e3e3e3", "marker": "none"}
    kwargsmonnodes = {"color": "red", "marker": "o"}
    kwargsannotations = {
        "color": "red",
        "fontsize": 10,
        "xytext": (-3, -3),
        "textcoords": "offset points",
        "ha": "right",
        "va": "bottom",
    }
    kwargsarrows = {
        "color": "red",
        "arrowstyle": "-|>",
        "lw": 1,
        "mutation_scale": SCFCARROW_LIST[ii],
    }
    figs3DSetup1.append(
        modelSetup1modes[-1].plot_mode_shape3D(
            SCFC_LIST[ii],
            kwargsWmodel,
            kwargsmarkers,
            kwargsmonnodes,
            kwargsannotations,
            kwargsarrows,
            save_to_file_path=OUTPATH,
        )
    )
    # ax = plt.gca()
    # ax.view_init(elev=70, azim=-125)
    # ax.set_box_aspect([2,2,1])
    # ax.dist=AX_DIST[ii]
    kwargsWmodel = {"color": "#e3e3e3", "linestyle": "solid", "zorder": 1}
    kwargsarrows2D = {
        "color": "red",
        "lw": 1,
        "zorder": 2,
        "head_length": 0.8,
        "head_width": 0.8,
    }
    zlim = 6
    modelSetup1modes[-1].plot_2D_top_view(
        zlim,
        SCFC_LIST[ii],
        kwargsWmodel,
        kwargsmarkers,
        kwargsmonnodes,
        kwargsannotations,
        kwargsarrows2D,
        save_to_file_path=OUTPATH,
    )
    zlim = 3
    modelSetup1modes[-1].plot_2D_top_view(
        zlim,
        SCFC_LIST[ii],
        kwargsWmodel,
        kwargsmarkers,
        kwargsmonnodes,
        kwargsannotations,
        kwargsarrows2D,
        save_to_file_path=OUTPATH,
    )

print("ok")
