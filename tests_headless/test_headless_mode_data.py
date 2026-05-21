"""Headless functional guard for the get_mode_geo*_data API.

Runs in the core-only ``headless`` CI environment: builds a setup, defines
geometry, attaches a modal result and calls the new headless mode-shape data
API — all without any GUI extra installed — and asserts no GUI package is
pulled. The directory sits outside ``tests/`` so it loads in a GUI-free env.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

GUI_PACKAGES = ("matplotlib", "pyvista", "pyvistaqt", "vtk", "PyQt5")


def test_get_mode_geo1_data_is_headless() -> None:
    """get_mode_geo1_data assembles geo1 mode data and pulls no GUI package."""
    for name in (*GUI_PACKAGES, "matplotlib.pyplot"):
        sys.modules.pop(name, None)

    from pyoma2.algorithms.data.result import BaseResult
    from pyoma2.setup import SingleSetup
    from pyoma2.support.geometry import ModeGeo1Data

    ss = SingleSetup(data=np.random.default_rng(0).random((100, 3)), fs=100)
    ss.def_geo1(
        sens_names=["s1", "s2", "s3"],
        sens_coord=pd.DataFrame(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            index=["s1", "s2", "s3"],
            columns=["x", "y", "z"],
        ),
        sens_dir=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )
    res = BaseResult(Fn=np.array([2.5]), Phi=np.array([[1.0], [2.0], [3.0]]))

    data = ss.get_mode_geo1_data(res, mode_nr=1, scaleF=5.0)

    assert isinstance(data, ModeGeo1Data)
    assert data.deformed_coord.shape == (3, 3)
    for gui in GUI_PACKAGES:
        assert gui not in sys.modules, f"get_mode_geo1_data pulled GUI package {gui!r}"


def test_get_mode_geo2_data_is_headless() -> None:
    """get_mode_geo2_data assembles geo2 mode data and pulls no GUI package."""
    for name in (*GUI_PACKAGES, "matplotlib.pyplot"):
        sys.modules.pop(name, None)

    from pyoma2.algorithms.data.result import BaseResult
    from pyoma2.setup import SingleSetup
    from pyoma2.support.geometry import ModeGeo2Data

    ss = SingleSetup(data=np.random.default_rng(0).random((100, 2)), fs=100)
    pts_coord = pd.DataFrame(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        index=["p1", "p2"],
        columns=["x", "y", "z"],
    )
    sens_map = pd.DataFrame(
        [["s1", 0, 0], [0, "s2", 0]],
        index=pts_coord.index,
        columns=pts_coord.columns,
    )
    ss.def_geo2(sens_names=["s1", "s2"], pts_coord=pts_coord, sens_map=sens_map)
    res = BaseResult(Fn=np.array([3.0]), Phi=np.array([[0.5], [2.0]]))

    data = ss.get_mode_geo2_data(res, mode_nr=1, scaleF=2.0)

    assert isinstance(data, ModeGeo2Data)
    assert data.deformed_coord.shape == (2, 3)
    for gui in GUI_PACKAGES:
        assert gui not in sys.modules, f"get_mode_geo2_data pulled GUI package {gui!r}"
