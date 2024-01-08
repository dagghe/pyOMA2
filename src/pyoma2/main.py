import numpy as np

from pyoma2.algorithm import EFDD_algo, FDD_algo, FSDD_algo, SSIcov_algo, SSIdat_algo
from pyoma2.OMA import MultiSetup, SingleSetup  # noqa: F401
from pyoma2.utils.utils import read_from_test_data

if __name__ == "__main__":
    """Running single setup"""
    print("**** Running single setup ****")

    # read data
    htc1 = read_from_test_data("src/pyoma2/test_data/3SL/3SL_set1.txt")
    palisaden1 = read_from_test_data(
        "src/pyoma2/test_data/palisaden/Palisaden_3_11082022_DAG.txt"
    )
    # ATTENZIONE
    # non funziona direttamento con dataframe
    palisaden1 = palisaden1.to_numpy()[:, 1:]

    # create single setup 1
    fs1 = 200
    Pali_ss = SingleSetup(palisaden1, fs1)

    fig, ax = Pali_ss.plot_data()
# =============================================================================
# 
# =============================================================================
    # # INITIALIZE ALGORITHM WITHOUT RUN PARAMETERS # #
# Se diventa troppo complicato settare i run param dopo, allora obblighiamo
# gli utenti a definire subito i run param specifici per ciascun algoritmo.
    fdd = FDD_algo(name="FDD")
    # FIXME TUTTO SI ROMPE
    efdd = EFDD_algo(name="EFDD", nxseg=2048, method_SD="cor")
    fsdd = FSDD_algo(name="FSDD", nxseg=1024, method_SD="per", pov=0.5)
    ssicov = SSIcov_algo(name="SSIcov", br=80, ordmax=150)
    ssidat = SSIcov_algo(name="SSIdat", br=80, ordmax=150,
                         ref_ind=[0, 1, 2, 3],ordmin=0,step=1,err_fn=0.1,
                         err_xi=0.05,err_phi=0.03,xi_max=0.1,)
# N.B. gli algo FDD, EFDD e FSDD potrebbero anche essere definiti soltanto
# usando name, dato che comunque i run param hanno tutti dei valori di default
# SSI invece no perche ha br che non ha valori di default

# certo non sarebbe male avere anche l opzione di:
    fdd.set_run_param = FDD_algo.RunParam(nxseg=512, method_SD="per", pov=0.5)
# (chiaramente sovrascrivendo quelli gia definiti all iniz o di default)

# Comunque un MUST deve essere l opzione per poter chiamare i run param dall
# algo e ottenere qualcosa tipo un dizionario.
    run_param_ssi = ssicov.run_param
# oppure
    run_param_ssi = Pali_ss[ssicov.name].run_param

    # Pali_ss.add_algorithms(ssidat, ssicov, efdd, fsdd, fdd)
    Pali_ss.add_algorithms(efdd, fsdd, fdd)
    Pali_ss.run_all()
# =============================================================================
    # plot "statici"
    fdd.plot_MIF()
# =============================================================================
    # e poi
    fdd.MPE_fromPlot()
    # oppure
    Pali_ss[fdd.name].MPE_fromPlot(DF=0.1)
    # e poi
    efdd.MPE_fromPlot()
    fsdd.MPE(sel_freq=[1.,2.,3.], 
             DF1=0.1,DF2= 1.0,cm=2,MAClim=0.85,sppk=3,npmax=20)
    # o anche qui
    Pali_ss[ssicov.name].MPE_fromPlot()
    Pali_ss[ssidat.name].MPE(sel_freq=[1.,2.,3.], 
                             order_in="find_min",deltaf=0.05,rtol=1e-2,)
    
    # anche qui poi poter chiamare i risultati su un dizionario o simili
    fdd_res = Pali_ss[fdd.name].result
    efdd_res = Pali_ss[efdd.name].result
    # ecc ecc
    # o anche le singole variabili
    Fn_efdd = Pali_ss[efdd.name].result.Fn
# =============================================================================


