import matplotlib.pyplot as plt
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
    palisaden1 = palisaden1.to_numpy()[:, :]

    # create single setup 1
    fs1 = 200
    # ------------------------------------------------------------------------------
    # filtering
    _sos = signal.butter(12, (0.5, 40.0), "bandpass", output="sos", fs=fs1)
    palisaden1 = signal.sosfiltfilt(_sos, palisaden1, axis=0)
    # Decimation
    q = 5  # Decimation factor
    palisaden1 = signal.decimate(palisaden1, q, axis=0)  # Decimation
    fs1 = fs1 / q  # [Hz] Decimated sampling frequency
    # ------------------------------------------------------------------------------

    # create single setup
    Pali_ss = SingleSetup(palisaden1, fs1)

    # Plot the Time Histories
    fig, ax = Pali_ss.plot_data(unit="Volts", show_rms=True)
    # # to show fig
    # fig.show()
    # plt.show()

    # =============================================================================
    #
    # =============================================================================
    # Initialise the algorithms
    fdd = FDD_algo(name="FDD")
    fsdd = FSDD_algo(name="FSDD", nxseg=2048, method_SD="per", pov=0.5)
    ssicov = SSIcov_algo(name="SSIcov", method="cov_mm", br=40, ordmax=80)
    ssicov1 = SSIcov_algo(name="SSIcov1", method="cov_unb", br=40, ordmax=80)
    # Overwrite/update run parameters for an algorithm
    fdd.run_params = FDD_algo.RunParamCls(nxseg=1024, method_SD="cor")

    # save run_params for SSIcov algorithm
    # run_param_ssi = dict(ssicov.run_params)
    # run_param_ssidat = dict(ssidat.run_params)
    # run_param_fdd = dict(fdd.run_params)

    # Add algorithms to the class
    Pali_ss.add_algorithms(ssicov1, ssicov, fsdd, fdd)

    # After having added the algo, its methods are accessible from the class
    run_param_ssi = dict(Pali_ss[ssicov.name].run_params)

    # Run all or run by name
    Pali_ss.run_by_name("SSIcov")
    Pali_ss.run_by_name("FSDD")
    # Pali_ss.run_all()
    # ------------------------------------------------------------------------------
    # plot "statici"
    # fdd.plot_CMIF(freqlim=5)
    fsdd.plot_CMIF(freqlim=5)
    # ssicov1.plot_STDiag(freqlim=5,hide_poles=False)
    ssicov.plot_STDiag(freqlim=5, hide_poles=False)
    # ssidat.plot_STDiag(freqlim=5,hide_poles=False)
    ssicov.plot_cluster(freqlim=5)
    # ------------------------------------------------------------------------------
    # save dict of results
    ssi_res = ssicov.result.model_dump()
    # ssidat_res = dict(ssidat.result)
    # fsdd_res = dict(fsdd.result)

    #%% =============================================================================
    Pali_ss.MPE_fromPlot("SSIcov", freqlim=5)
    Pali_ss.MPE_fromPlot("FSDD", freqlim=5)

    ssi_res = dict(ssicov.result)
    fsdd_res = dict(fsdd.result)

    #%% =============================================================================
    # GEO
    # import geometry files
    import os

    import pandas as pd

    _fold = r"src/pyoma2/test_data/palisaden"
    BG_nodes = _fold + os.sep + "BG_nodes.txt"
    BG_lines = _fold + os.sep + "BG_lines.txt"
    sens_coord = _fold + os.sep + "sens_coord.txt"
    sens_dir = _fold + os.sep + "sens_dir.txt"
    sens_map = _fold + os.sep + "sens_map.txt"
    sens_sign = _fold + os.sep + "sens_sign.txt"
    sens_lines = _fold + os.sep + "sens_lines.txt"
    pts_coord = _fold + os.sep + "pts_coord.txt"

    Names = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"]
    BG_nodes = np.loadtxt(
        BG_nodes,
    )
    BG_lines = np.loadtxt(
        BG_lines,
    ).astype(int)
    sens_coord = pd.read_csv(sens_coord, sep="\t")
    sens_dir = np.loadtxt(
        sens_dir,
    )[:, :]

    sens_lines = np.loadtxt(
        sens_lines,
    ).astype(int)
    pts_coord = pd.read_csv(pts_coord, sep="\t")
    sens_map = pd.read_csv(sens_map, sep="\t")
    sens_sign = pd.read_csv(sens_sign, sep="\t")

    Pali_ss.def_geo1(Names, sens_coord, sens_dir, bg_nodes=BG_nodes, bg_lines=BG_lines)

    Pali_ss.def_geo2(
        Names,
        pts_coord,
        sens_map,
        order_red="xy",
        sens_sign=sens_sign,
        sens_lines=sens_lines,
        bg_nodes=BG_nodes,
        bg_lines=BG_lines,
    )

    Geo1 = dict(Pali_ss.Geo1)
    Geo2 = dict(Pali_ss.Geo2)
    geo = Pali_ss.Geo2
    geo1 = Pali_ss.Geo1

    Pali_ss.plot_geo1(scaleF=2)

    Pali_ss[fsdd.name].plot_FIT(freqlim=5)

    Pali_ss[fsdd.name].plot_mode_g1(Geo1=geo1, mode_numb=2, view="3D", scaleF=2)

    Pali_ss[ssicov.name].anim_mode_g2(Geo2=geo, mode_numb=1, view="xy", scaleF=3)

    Pali_ss[fsdd.name].anim_mode_g2(Geo2=geo, mode_numb=3, view="xy", scaleF=3)
    # Pali_ss.plot_geo2()

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
