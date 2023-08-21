# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 18:48:53 2023

@author: dagpa
"""

import pyOMA2 as oma

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
q = 5 # Decimation factor
data = signal.decimate(data,  q, ftype='fir', axis=0) # Decimation
fs = fs/q # [Hz] Decimated sampling frequency
# -----------------------------------------------------------------------------



Test = oma.Model(data, fs)




#%%
# Run pLSCF
Test.pLSCF(40)
Test.sel_pole_pLSCF(freqlim=6.25)

# Run SSIcov
Test.SSIcov(30, ordmax=100)
Test.sel_pole_SSIcov(freqlim=6.25)

# Run (E)FDD
Test.FDDsvp()
# Test.sel_peak_FDD(freqlim=8)
Test.sel_peak_EFDD(freqlim=6.25)

# Save dictionary of results
Res = Test.Results

# Save mode shapes for each method
FI_pLSCF = Test.Results["pLSCF"]["Phi"]
FI_SSI = Test.Results["SSIcov"]["Phi"]
FI_EFDD = Test.Results["FDD"]["Phi"]

# Calculate MAC
MAC1 = oma.MAC(FI_ex, FI_pLSCF)
MAC2 = oma.MAC(FI_ex, FI_SSI)
MAC3 = oma.MAC(FI_ex, FI_EFDD)

tol = 1e-4
MAC1[MAC1.real < tol] = 0.
MAC2[MAC2.real < tol] = 0.
MAC3[MAC3.real < tol] = 0.

