import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pyoma2.algorithm import (  # noqa: F401
    EFDD_algo,
    EFDD_algo_MS,
    FDD_algo,
    FDD_algo_MS,
    FSDD_algo,
    SSIcov_algo,
    SSIcov_algo_MS,
    SSIdat_algo,
    SSIdat_algo_MS,
)
from pyoma2.functions.Gen_funct import PRE_MultiSetup, merge_mode_shapes  # noqa: F401
from pyoma2.OMA import MultiSetup_PoSER, MultiSetup_PreGER, SingleSetup  # noqa: F401
from pyoma2.utils.utils import read_from_test_data

    #%% =============================================================================

if __name__ == "__main__":
    """Running single setup"""
    print("**** Running single setup ****")

    # import data for single setup
    pali=np.loadtxt(r"src\pyoma2\test_data\palisaden\Palisaden_dataset.txt")

    # create single setup
    fs = 200
    # ------------------------------------------------------------------------------
    # filtering
    # _sos = signal.butter(12, (0.5, 40.0), "bandpass", output="sos", fs=fs1)
    # pali = signal.sosfiltfilt(_sos, pali, axis=0)
    # Decimation
    q = 5  # Decimation factor
    pali = signal.decimate(pali, q, axis=0)  # Decimation
    fs = fs / q  # [Hz] Decimated sampling frequency
    # ------------------------------------------------------------------------------

    # create single setup
    Pali_ss = SingleSetup(pali, fs)

    # Plot the Time Histories
    fig, ax = Pali_ss.plot_data(unit="Volts", show_rms=True)

    # =============================================================================
    #
    # =============================================================================
    # Initialise the algorithms
    fdd = FDD_algo(name="FDD")
    fsdd = FSDD_algo(name="FSDD", nxseg=512, method_SD="per", pov=0.5)
    ssicov = SSIcov_algo(name="SSIcov", method="cov_mm", br=40, ordmax=80)

    # Overwrite/update run parameters for an algorithm
    fdd.run_params = FDD_algo.RunParamCls(nxseg=1024, method_SD="cor")

    # Add algorithms to the class
    Pali_ss.add_algorithms(ssicov, fsdd, fdd)

    # After having added the algo, its methods are accessible from the class
    run_param_ssi = dict(Pali_ss[ssicov.name].run_params)

    # Run all or run by name
    Pali_ss.run_by_name("SSIcov")
    Pali_ss.run_by_name("FSDD")
    # Pali_ss.run_all()
    # ------------------------------------------------------------------------------
    # plot
    # fdd.plot_CMIF(freqlim=5)
    fsdd.plot_CMIF(freqlim=5)
    ssicov.plot_STDiag(freqlim=5, hide_poles=False)
    ssicov.plot_cluster(freqlim=5)
    # ------------------------------------------------------------------------------
    # save dict of results
    ssi_res = ssicov.result.model_dump()
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

if __name__ == "__main__":
    """Running multi setup"""
    print("**** Running multi setup ****")
    # import data for multi setup
    set1=np.load(r"src\pyoma2\test_data\3SL\set1.npy",allow_pickle=True)
    set2=np.load(r"src\pyoma2\test_data\3SL\set2.npy",allow_pickle=True)
    set3=np.load(r"src\pyoma2\test_data\3SL\set3.npy",allow_pickle=True)

    data = [set1,set2,set3]
    ref_ind=[[0,1,2],[0,1,2],[0,1,2]]

    build_ms = MultiSetup_PreGER(fs=100., ref_ind=ref_ind, datasets=data)

    # Initialise the algorithms
    efdd = EFDD_algo_MS(name="EFDD", nxseg=1024, method_SD="per", pov=0.5)
    ssicov = SSIcov_algo_MS(name="SSIcov", method="cov_mm", br=80, ordmax=100)

    # Add algorithms to the class
    build_ms.add_algorithms(ssicov, efdd)
    build_ms.run_all()

    run_param_ssi = dict(build_ms[ssicov.name].run_params)
    run_param_efdd = dict(build_ms[efdd.name].run_params)
    ssi_res = dict(ssicov.result)
    efdd_res = dict(efdd.result)

    # plot
    # fdd.plot_CMIF(freqlim=5)
    efdd.plot_CMIF(freqlim=25)
    ssicov.plot_STDiag(freqlim=25, hide_poles=False)
    ssicov.plot_cluster(freqlim=25)


#%% =============================================================================

if __name__ == "__main__":
    """Running multi setup POSER"""
    print("**** Running multi setup POSER ****")

    # import data for single setup
    pali = np.loadtxt(
        pathlib.Path(r"src/pyoma2/test_data/palisaden/Palisaden_dataset.txt")
    )

    # create single setup
    fs = 200
    # ------------------------------------------------------------------------------
    # filtering
    # _sos = signal.butter(12, (0.5, 40.0), "bandpass", output="sos", fs=fs1)
    # pali = signal.sosfiltfilt(_sos, pali, axis=0)
    # Decimation
    q = 5  # Decimation factor
    pali = signal.decimate(pali, q, axis=0)  # Decimation
    fs = fs / q  # [Hz] Decimated sampling frequency
    # ------------------------------------------------------------------------------

    # create single setup
    Pali_ss1 = SingleSetup(pali, fs)
    Pali_ss2 = SingleSetup(pali, fs)

    # # Plot the Time Histories
    # fig, ax = Pali_ss1.plot_data(unit="Volts", show_rms=True)

    # =============================================================================
    #
    # =============================================================================
    # # Initialise the algorithms for setup 1
    fdd1 = FDD_algo(name="FDD")
    fsdd1 = FSDD_algo(name="FSDD", nxseg=512, method_SD="per", pov=0.5)
    ssicov1 = SSIcov_algo(name="SSIcov", method="cov_mm", br=40, ordmax=80)

    # Overwrite/update run parameters for an algorithm
    fdd1.run_params = FDD_algo.RunParamCls(nxseg=1024, method_SD="cor")

    # Add algorithms to the class
    Pali_ss1.add_algorithms(ssicov1, fsdd1, fdd1)
    Pali_ss1.run_all()

    # # Initialise the algorithms for setup 2
    fdd2 = FDD_algo(name="FDD")
    fsdd2 = FSDD_algo(name="FSDD", nxseg=512, method_SD="per", pov=0.5)
    ssicov2 = SSIcov_algo(name="SSIcov", method="cov_mm", br=40, ordmax=80)

    # Overwrite/update run parameters for an algorithm
    fdd2.run_params = FDD_algo.RunParamCls(nxseg=1024, method_SD="cor")

    # Add algorithms to the class
    Pali_ss2.add_algorithms(ssicov2, fsdd2, fdd2)
    Pali_ss2.run_all()

    # ------------------------------------------------------------------------------
    # plot
    # fdd.plot_CMIF(freqlim=5)
    fsdd1.plot_CMIF(freqlim=5)
    ssicov1.plot_STDiag(freqlim=5, hide_poles=False)
    ssicov1.plot_cluster(freqlim=5)
    # ------------------------------------------------------------------------------

    #%% =============================================================================
    Pali_ss1.MPE_fromPlot("SSIcov", freqlim=5)
    Pali_ss1.MPE_fromPlot("FSDD", freqlim=5)

    # plot
    # fdd.plot_CMIF(freqlim=5)
    fsdd2.plot_CMIF(freqlim=5)
    ssicov2.plot_STDiag(freqlim=5, hide_poles=False)
    ssicov2.plot_cluster(freqlim=5)
    # ------------------------------------------------------------------------------

    #%% =============================================================================
    Pali_ss2.MPE_fromPlot("SSIcov", freqlim=5)
    Pali_ss2.MPE_fromPlot("FSDD", freqlim=5)

    ################ INITIALIZATION OF THE MULTISETUP #############################
    ref_ind = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    msp = MultiSetup_PoSER(ref_ind=ref_ind, single_setups=[Pali_ss1, Pali_ss2])
    result = msp.merge_results()

    fdd_merged = result[FDD_algo.__class__.__name__]
    fsdd_merged = result[FSDD_algo.__class__.__name__]
    ssicov_merged = result[SSIcov_algo.__class__.__name__]
