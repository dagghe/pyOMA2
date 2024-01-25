"""
Created on Wed Jan 24 09:28:43 2024

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
    SSIcov_algo_MS,
)
from pyoma2.OMA import MultiSetup_PreGER

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
    with pkg_resources_file("pyoma2.test_data.3SL") as path:
        base_path = get_base_path(path=path)

        # import data files
        set1 = np.load(base_path.joinpath("set1.npy"), allow_pickle=True)
        set2 = np.load(base_path.joinpath("set2.npy"), allow_pickle=True)
        set3 = np.load(base_path.joinpath("set3.npy"), allow_pickle=True)

        # import geometry files
        BG_nodes = base_path.joinpath("BG_nodes.txt")
        BG_lines = base_path.joinpath("BG_lines.txt")
        # Geometry 1
        sens_coord = base_path.joinpath("sens_coord.txt")
        sens_dir = base_path.joinpath("sens_dir.txt")
        # Geometry 2
        sens_map = base_path.joinpath("sens_map.txt")
        sens_sign = base_path.joinpath("sens_sign.txt")
        sens_lines = base_path.joinpath("sens_lines.txt")
        pts_coord = base_path.joinpath("pts_coord.txt")

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
        BG_nodes = np.loadtxt(BG_nodes)
        BG_lines = np.loadtxt(BG_lines).astype(int)
        sens_coord = pd.read_csv(sens_coord, sep="\t")
        sens_dir = np.loadtxt(sens_dir)

        sens_lines = np.loadtxt(sens_lines).astype(int)
        pts_coord = pd.read_csv(pts_coord, sep="\t")
        sens_map = pd.read_csv(sens_map, sep="\t")
        sens_sign = pd.read_csv(sens_sign, sep="\t")

        # =============================================================================
        fs = 100
        _q = 2  # Decimation factor
        set1 = signal.decimate(set1, _q, axis=0)  # Decimation
        set2 = signal.decimate(set2, _q, axis=0)  # Decimation
        set3 = signal.decimate(set3, _q, axis=0)  # Decimation
        fs = fs / _q  # [Hz] Decimated sampling frequency
        # ------------------------------------------------------------------------------
        data = [set1, set2, set3]
        ref_ind = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

        # Create multisetup
        msp = MultiSetup_PreGER(fs=fs, ref_ind=ref_ind, datasets=data)

        # -----------------------------------------------------------------------------
        # Define geometry1
        msp.def_geo1(
            Names,  # Names of the channels
            sens_coord,  # coordinates of the sensors
            sens_dir,  # sensors' direction
            bg_nodes=BG_nodes,  # BG nodes
            bg_lines=BG_lines,  # BG lines
        )

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

        # Plot the geometry
        msp.plot_geo1(scaleF=2)

        # -----------------------------------------------------------------------------
        # Initialise the algorithms
        ssicov = SSIcov_algo_MS(name="SSIcov", method="cov_mm", br=80, ordmax=100)

        # Add algorithms to the class
        msp.add_algorithms(ssicov)
        msp.run_all()

        # Plot
        ssicov.plot_STDiag(freqlim=20)

        # %% =============================================================================
        # msp.MPE_fromPlot("SSIcov", freqlim=20)

        # ------------------------------------------------------------------------------
        msp.MPE(
            "SSIcov",
            sel_freq=[2.63, 2.69, 3.43, 8.29, 8.42, 10.62, 14.00, 14.09, 17.57],
            order=80,
        )

        # %% =============================================================================
        # Plot mode 2 (geometry 1)
        ssicov.plot_mode_g1(Geo1=msp.Geo1, mode_numb=2, view="3D", scaleF=2)
        # Animate mode 3 (geometry 2)
        ssicov.anim_mode_g2(Geo2=msp.Geo2, mode_numb=3, view="xy", scaleF=3)


if __name__ == "__main__":
    main()
