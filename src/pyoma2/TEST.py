# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 22:47:49 2024

@author: dagpa
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyoma2.algorithm import pLSCF_algo_MS
from pyoma2.OMA import MultiSetup_PreGER

# load example dataset for single setup
# import data files
set1 = np.load(
    r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyOMA2\src\pyoma2\test_data\3SL\set1.npy",
    allow_pickle=True,
)
set2 = np.load(
    r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyOMA2\src\pyoma2\test_data\3SL\set2.npy",
    allow_pickle=True,
)
set3 = np.load(
    r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyOMA2\src\pyoma2\test_data\3SL\set3.npy",
    allow_pickle=True,
)

# list of datasets and reference indices
data = [set1, set2, set3]
ref_ind = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

# Create multisetup
msp = MultiSetup_PreGER(fs=100, ref_ind=ref_ind, datasets=data)

# decimate data
msp.decimate_data(q=2)

# import geometry files
# Names of the channels
Names = [
    [
        "ch1_1",
        "ch2_1",
        "ch3_1",
        "ch4_1",
        "ch5_1",
        "ch6_1",
        "ch7_1",
        "ch8_1",
        "ch9_1",
        "ch10_1",
    ],
    [
        "ch1_2",
        "ch2_2",
        "ch3_2",
        "ch4_2",
        "ch5_2",
        "ch6_2",
        "ch7_2",
        "ch8_2",
        "ch9_2",
        "ch10_2",
    ],
    [
        "ch1_3",
        "ch2_3",
        "ch3_3",
        "ch4_3",
        "ch5_3",
        "ch6_3",
        "ch7_3",
        "ch8_3",
        "ch9_3",
        "ch10_3",
    ],
]
# Background
BG_nodes = np.loadtxt(
    r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyOMA2\src\pyoma2\test_data\3SL\BG_nodes.txt"
)
BG_lines = np.loadtxt(
    r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyOMA2\src\pyoma2\test_data\3SL\BG_lines.txt"
).astype(int)
# Geometry 1
sens_coord = pd.read_csv(
    r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyOMA2\src\pyoma2\test_data\3SL\sens_coord.txt",
    sep="\t",
)
sens_dir = np.loadtxt(
    r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyOMA2\src\pyoma2\test_data\3SL\sens_dir.txt"
)
# Geometry 2
sens_lines = np.loadtxt(
    r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyOMA2\src\pyoma2\test_data\3SL\sens_lines.txt"
).astype(int)
pts_coord = pd.read_csv(
    r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyOMA2\src\pyoma2\test_data\3SL\pts_coord.txt",
    sep="\t",
)
sens_map = pd.read_csv(
    r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyOMA2\src\pyoma2\test_data\3SL\sens_map.txt",
    sep="\t",
)
sens_sign = pd.read_csv(
    r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyOMA2\src\pyoma2\test_data\3SL\sens_sign.txt",
    sep="\t",
)

# Define geometry1
msp.def_geo1(
    Names,  # Names of the channels
    sens_coord,  # coordinates of the sensors
    sens_dir,  # sensors' direction
    bg_nodes=BG_nodes,  # BG nodes
    bg_lines=BG_lines,
)  # BG lines

# Define geometry 2
msp.def_geo2(
    Names,  # Names of the channels
    pts_coord,
    sens_map,
    order_red="xy",
    sens_sign=sens_sign,
    sens_lines=sens_lines,
    bg_nodes=BG_nodes,
    bg_lines=BG_lines,
)


# Initialise the algorithms
plscf = pLSCF_algo_MS(name="pLSCF", ordmax=80)

# Add algorithms to the class
msp.add_algorithms(plscf)
msp.run_by_name("pLSCF")

# Plot
plscf.plot_STDiag(freqlim=(1, 20))
