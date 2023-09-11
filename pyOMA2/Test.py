# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 18:48:53 2023

@author: dagpa
"""

import pyOMA2new as oma

import pandas as pd
import numpy as np
from scipy import signal

# _file = r"C:\Users\dpa\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Python Scripts\pyOMA_TEST\5DOF_fixed_ex1.txt"
# _file = r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Python Scripts\pyOMA_TEST\5DOF_fixed_ex1.txt"

# open the file with pandas and then convert to numpy array
# data = pd.read_csv(_file, header=0, sep="\t", index_col=False) 
# data = data.to_numpy()

# open the example data
data, (fn_ex, FI_ex, xi_ex) = oma.Exdata()

# sampling frequency
fs = 100


# -----------------------------------------------------------------------------
# Filtering
data = signal.detrend(data, axis=0) # Trend rmoval
q = 2 # Decimation factor
data = signal.decimate(data,  q, ftype='fir', axis=0) # Decimation
fs = fs/q # [Hz] Decimated sampling frequency
# -----------------------------------------------------------------------------

# nodes_coord = np.array([[0,3],[0,6],[0,9],[0,12],[0,15]])
nodes_coord = np.array([[0,0,3],[0,0,6],[0,0,9],[0,0,12],[0,0,15]])

directions = np.ones(5)

Test = oma.Model(data, fs)

Test.def_geo(nodes_coord,directions)


#%%
# Run pLSCF
sel_freq = [0.89, 2.59, 4.1, 5.25, 6.0]

Test.pLSCF(40)
# Test.sel_pole_pLSCF(freqlim=20)
Test.get_mod_pLSCF(sel_freq=sel_freq, order =39)

Test.anim_mode("pLSCF", 0)

#%%

# Run SSIcov
Test.SSIcov(100, ordmax=100)
# Test.sel_pole_SSI()
Test.get_mod_SSI(sel_freq=[0.89, 2.59, 4.1, 5.25, 6.0], order = 30)

# Run SSIdat
Test.SSIdat(50, ordmax=100)
# Test.sel_pole_SSI(freqlim=6.25)
Test.get_mod_SSI(sel_freq=[0.89, 2.59, 4.1, 5.25, 6.0], order = 44)

# Run (E)FDD
Test.FDDsvp(df=0.05)
Test.sel_peak_FDD(freqlim=20)
# Test.sel_peak_EFDD(freqlim=6.25)
Test.get_mod_EFDD(sel_freq=[0.89, 2.59, 4.1, 5.25, 6.0])

# Save dictionary of results
Res = Test.Results

# Calculate MAC
MAC1 = oma.MAC(FI_ex, Test.Results["pLSCF"]["Phi"])
MAC2 = oma.MAC(FI_ex, Test.Results["SSIcov"]["Phi"])
MAC3 = oma.MAC(FI_ex, Test.Results["SSIdat"]["Phi"])
MAC4 = oma.MAC(FI_ex, Test.Results["FDD"]["Phi"])

