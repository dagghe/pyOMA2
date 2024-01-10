import numpy as np
from scipy import signal

from pyoma2.algorithm import EFDD_algo, FDD_algo, FSDD_algo, SSIcov_algo, SSIdat_algo
from pyoma2.OMA import MultiSetup, SingleSetup  # noqa: F401
from pyoma2.utils.utils import read_from_test_data

if __name__ == "__main__":
    """Running single setup"""
    print("**** Running single setup ****")

    # read data
    # htc1 = read_from_test_data("src/pyoma2/test_data/3SL/3SL_set1.txt")
    palisaden1 = read_from_test_data(
        "src/pyoma2/test_data/palisaden/Palisaden_3_11082022_DAG.txt"
    )
    # ATTENZIONE
    # non funziona direttamento con dataframe
    palisaden1 = palisaden1.to_numpy()[:, 1:]

    # create single setup 1
    fs1 = 200
# ------------------------------------------------------------------------------
    # filtering
    _sos = signal.butter(12, (0.5, 40.), "bandpass",output="sos",fs=fs1)
    palisaden1 = signal.sosfiltfilt(_sos, palisaden1, axis=0)
    # Decimation
    q = 4  # Decimation factor
    palisaden1 = signal.decimate(palisaden1, q, axis=0)  # Decimation
    fs1 = fs1/q  # [Hz] Decimated sampling frequency
# ------------------------------------------------------------------------------

    # create single setup
    Pali_ss = SingleSetup(palisaden1, fs1)

    # Plot the Time Histories
    # fig, ax = Pali_ss.plot_data()
    
# =============================================================================
# 
# =============================================================================
    # Initialise the algorithms
    fdd = FDD_algo(name="FDD")
    fsdd = FSDD_algo(name="FSDD", nxseg=2048, method_SD="per", pov=0.5)
    ssicov = SSIcov_algo(name="SSIcov", method="cov_matmul", br=40, ordmax=80)
    ssidat = SSIdat_algo(name="SSIdat", br=20, ordmax=50,)
    # Overwrite/update run parameters for an algorithm
    fdd.run_params = FDD_algo.RunParamCls(nxseg=1024, method_SD="cor")

    # save run_params for SSIcov algorithm
    # run_param_ssi = dict(ssicov.run_params)
    run_param_ssidat = dict(ssidat.run_params)
    run_param_fdd = dict(fdd.run_params)

    # Add algorithms to the class
    Pali_ss.add_algorithms(ssidat,ssicov, fsdd, fdd)
    # After having added the algo, its methods are accessible from the class
    run_param_ssi = dict(Pali_ss[ssicov.name].run_params)

    # Run all or run by name
    # Pali_ss.run_by_name("SSIcov")
    # Pali_ss.run_by_name("FSDD")
    Pali_ss.run_all()
# ------------------------------------------------------------------------------
    # plot "statici"
    fdd.plot_CMIF(freqlim=5)
    fsdd.plot_CMIF(freqlim=5)

    ssicov.plot_STDiag(freqlim=5,hide_poles=False)
    ssidat.plot_STDiag(freqlim=5,hide_poles=False)
    ssicov.plot_cluster(freqlim=5)
# ------------------------------------------------------------------------------
    # save dict of results
    ssi_res = dict(ssicov.result)
    ssi1_res = dict(ssidat.result)
    fsdd_res = dict(fsdd.result)
    
#%% =============================================================================
    Pali_ss.MPE_fromPlot("SSIcov",freqlim=5)
    Pali_ss.MPE_fromPlot("FSDD",freqlim=5)
    
#%% =============================================================================
    # GEO
    




#%% =============================================================================
    # e poi
#     fsdd.mpe_fromPlot()
#     # oppure
#     # e poi
#     efdd.MPE_fromPlot()
#     fsdd.MPE(sel_freq=[1.,2.,3.], 
#              DF1=0.1,DF2= 1.0,cm=2,MAClim=0.85,sppk=3,npmax=20)
#     # o anche qui
#     Pali_ss[ssidat.name].MPE(sel_freq=[1.,2.,3.], 
#                              order_in="find_min",deltaf=0.05,rtol=1e-2,)
    
#     # anche qui poi poter chiamare i risultati su un dizionario o simili
#     # fdd_res = Pali_ss[fdd.name].result
#     efdd_res = Pali_ss[efdd.name].result
#     # ecc ecc
#     # o anche le singole variabili
#     Fn_efdd = Pali_ss[efdd.name].result.Fn
# # =============================================================================


