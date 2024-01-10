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
    
    # filtering
    _sos = signal.butter(12, (0.5, 40.), "bandpass",output="sos",fs=fs1)
    palisaden1 = signal.sosfiltfilt(_sos, palisaden1, axis=0)
    
    q = 2  # Decimation factor
    palisaden1 = signal.decimate(palisaden1, q, axis=0)  # Decimation
    fs1 = fs1/q  # [Hz] Decimated sampling frequency
    # # create single setup 1
    # q = 2  # Decimation factor
    # palisaden1 = signal.decimate(palisaden1, q, axis=0)  # Decimation
    # fs1 = fs1/q  # [Hz] Decimated sampling frequency

    Pali_ss = SingleSetup(palisaden1, fs1)
    # fig, ax = Pali_ss.plot_data()
# =============================================================================
# 
# =============================================================================
    # # INITIALIZE ALGORITHM WITHOUT RUN PARAMETERS # #
# Se diventa troppo complicato settare i run param dopo, allora obblighiamo
# gli utenti a definire subito i run param specifici per ciascun algoritmo.
    # fdd = FDD_algo(name="FDD",nxseg=1024, method_SD="cor")
    # FIXME TUTTO SI ROMPE
    # efdd = EFDD_algo(name="EFDD", nxseg=1024, method_SD="cor")
    fsdd = FSDD_algo(name="FSDD", nxseg=4096, method_SD="per", pov=0.5)
    ssicov = SSIcov_algo(name="SSIcov", method="cov_matmul", br=30, ordmax=60)
    # ssidat = SSIdat_algo(name="SSIdat", br=20, ordmax=100,)
# N.B. gli algo FDD, EFDD e FSDD potrebbero anche essere definiti soltanto
# usando name, dato che comunque i run param hanno tutti dei valori di default
# SSI invece no perche ha br che non ha valori di default

# certo non sarebbe male avere anche l opzione di:
    # fdd.set_run_param = FDD_algo.RunParamCls(nxseg=512, method_SD="per", pov=0.5)
# (chiaramente sovrascrivendo quelli gia definiti all iniz o di default)

# Comunque un MUST deve essere l opzione per poter chiamare i run param dall
# algo e ottenere qualcosa tipo un dizionario.
    run_param_ssi = dict(ssicov.run_params)
# oppure 
    # run_param_ssi = Pali_ss[ssicov.name].run_params

    Pali_ss.add_algorithms(ssicov, fsdd)
    # Pali_ss.add_algorithms(ssidat,ssicov, fsdd, fdd)
    # Pali_ss.run_by_name("SSIcov")
    Pali_ss.run_all()
# =============================================================================
    # plot "statici"
    # efdd.plot_CMIF()
    # fdd.plot_CMIF(freqlim=5)
    fsdd.plot_CMIF(freqlim=5)

    ssicov.plot_STDiag(freqlim=5,hide_poles=False)
    # ssidat.plot_STDiag(freqlim=5,hide_poles=False)

    # ssicov.plot_cluster()
    ssi_res = dict(ssicov.result)
    fsdd_res = dict(fsdd.result)
    
#%% =============================================================================
    # Pali_ss.MPE_fromPlot("SSIcov",freqlim=5)
    Pali_ss.MPE_fromPlot("FSDD",freqlim=5)
    
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


