"""
Created on Tue Jan 23 21:46:10 2024

@author: dagpa
"""
import importlib.resources as pkg_resources
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from pyoma2.algorithm import (
    FDD_algo,
    FSDD_algo,
    SSIcov_algo,
)
from pyoma2.OMA import SingleSetup

major, minor, _ = sys.version_info[0:3]
if major == 3 or minor == 8:
    # import file from relative library path, py == 3.8 way
    pkg_resources_file = partial(pkg_resources.path, resource="__init__.py")
    get_base_path = lambda path: Path(path).parent  # noqa E731
else:
    # import file from relative library path, py > 3.8 way
    pkg_resources_file = pkg_resources.files
    get_base_path = lambda path: Path(path)  # noqa E731


def main():
    with pkg_resources_file("pyoma2.test_data.palisaden") as path:
        base_path = get_base_path(path=path)

        # import data for single setup
        pali = np.load(
            base_path.joinpath("Palisaden_dataset.npy"),
            allow_pickle=True,
        )

        # import geometry files
        # Names of the channels
        Names = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"]
        # Common Backgroung nodes and lines
        BG_nodes = np.loadtxt(base_path.joinpath("BG_nodes.txt"))
        BG_lines = np.loadtxt(base_path.joinpath("BG_lines.txt")).astype(int)
        # Geometry 1
        sens_coord = pd.read_csv(base_path.joinpath("sens_coord.txt"), sep="\t")
        sens_dir = np.loadtxt(base_path.joinpath("sens_dir.txt"))
        # Geometry 2
        sens_lines = np.loadtxt(base_path.joinpath("sens_lines.txt")).astype(int)
        pts_coord = pd.read_csv(base_path.joinpath("pts_coord.txt"), sep="\t")
        sens_map = pd.read_csv(base_path.joinpath("sens_map.txt"), sep="\t")
        sens_sign = pd.read_csv(base_path.joinpath("sens_sign.txt"), sep="\t")

        # =============================================================================
        fs = 100  # sampling frequency

        # -----------------------------------------------------------------------------
        # filtering and decimation
        # _sos = signal.butter(12, (0.5, 5.0), "bandpass", output="sos", fs=fs)
        # pali = signal.sosfiltfilt(_sos, pali, axis=0)

        _DecFac = 2  # Decimation factor
        pali = signal.decimate(pali, _DecFac, axis=0)  # Decimation
        fs = fs / _DecFac  # [Hz] Decimated sampling frequency

        # -----------------------------------------------------------------------------
        # create single setup
        Pali_ss = SingleSetup(pali, fs)

        # Plot the Time Histories
        fig, ax = Pali_ss.plot_data(
            unit="Volts",
        )

        # -----------------------------------------------------------------------------
        # Define geometry1
        Pali_ss.def_geo1(
            Names,  # Names of the channels
            sens_coord,  # coordinates of the sensors
            sens_dir,  # sensors' direction
            bg_nodes=BG_nodes,  # BG nodes
            bg_lines=BG_lines,  # BG lines
        )

        # Define geometry 2
        Pali_ss.def_geo2(
            Names,  # Names of the channels
            pts_coord,
            sens_map,
            order_red="xy",
            sens_sign=sens_sign,
            sens_lines=sens_lines,
            bg_nodes=BG_nodes,
            bg_lines=BG_lines,
        )

        # Plot the geometry
        Pali_ss.plot_geo1(scaleF=2)

        # -----------------------------------------------------------------------------
        # Initialise the algorithms
        fdd = FDD_algo(name="FDD")
        fsdd = FSDD_algo(name="FSDD", nxseg=2048, method_SD="per", pov=0.5)
        ssicov = SSIcov_algo(name="SSIcov", method="cov_mm", br=50, ordmax=80)

        # Overwrite/update run parameters for an algorithm
        fdd.run_params = FDD_algo.RunParamCls(nxseg=512, method_SD="cor")

        # Add algorithms to the single setup class
        Pali_ss.add_algorithms(ssicov, fsdd, fdd)

        # Run all or run by name
        Pali_ss.run_by_name("SSIcov")
        Pali_ss.run_by_name("FSDD")
        # Pali_ss.run_all()

        # plot
        fsdd.plot_CMIF(freqlim=5)
        ssicov.plot_STDiag(freqlim=5, hide_poles=False)
        ssicov.plot_cluster(freqlim=5)

        # save dict of results
        ssi_res = ssicov.result.model_dump()
        fsdd_res = dict(fsdd.result)

        # %% =============================================================================
        # Select modes to extract from plots
        # Pali_ss.MPE_fromPlot("SSIcov", freqlim=5)
        # Pali_ss.MPE_fromPlot("FSDD", freqlim=5)
        # or directly
        Pali_ss.MPE("SSIcov", sel_freq=[1.88, 2.42, 2.68], order=40)
        Pali_ss.MPE("FSDD", sel_freq=[1.88, 2.42, 2.68], MAClim=0.95)

        # save dict of results
        ssi_res = dict(ssicov.result)  # noqa F841
        fsdd_res = dict(fsdd.result)  # noqa F841

        # plot additional info (goodness of fit) for EFDD or FSDD
        Pali_ss[fsdd.name].plot_FIT(freqlim=5)

        # %% =============================================================================
        # MODE SHAPES PLOT
        # Plot mode 2 (geometry 1)
        Pali_ss[fsdd.name].plot_mode_g1(
            Geo1=Pali_ss.Geo1, mode_numb=2, view="3D", scaleF=2
        )
        # Animate mode 1 (geometry 2)
        Pali_ss[ssicov.name].anim_mode_g2(
            Geo2=Pali_ss.Geo2, mode_numb=1, view="xy", scaleF=3
        )
        # Animate mode 3 (geometry 2)
        Pali_ss[fsdd.name].anim_mode_g2(
            Geo2=Pali_ss.Geo2, mode_numb=3, view="xy", scaleF=3
        )


if __name__ == "__main__":
    main()
