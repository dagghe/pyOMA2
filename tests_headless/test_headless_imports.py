"""Guard tests for the headless install.

These run in the core-only ``headless`` tox/CI environment. The directory sits
outside ``tests/`` on purpose: ``tests/conftest.py`` and ``tests/factory.py``
import matplotlib at module scope, so a test under ``tests/`` could not load in
a GUI-free environment. ``pyproject.toml``'s ``testpaths = ["tests"]`` keeps the
normal suite from collecting this directory.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

HEADLESS_SURFACE = (
    "pyoma2",
    "pyoma2.algorithms",
    "pyoma2.setup",
    "pyoma2.functions.gen",
    "pyoma2.functions.fdd",
    "pyoma2.functions.ssi",
    "pyoma2.functions.clus",
    "pyoma2.support.geometry",
)

GUI_PACKAGES = ("matplotlib", "pyvista", "pyvistaqt", "vtk", "PyQt5")


def test_headless_surface_pulls_no_gui():
    """Importing the core surface must not import any GUI package."""
    for name in (*GUI_PACKAGES, "matplotlib.pyplot"):
        sys.modules.pop(name, None)

    for module in HEADLESS_SURFACE:
        importlib.import_module(module)

    for gui in GUI_PACKAGES:
        assert gui not in sys.modules, f"headless import pulled GUI package {gui!r}"


def test_plotting_without_extra_raises_friendly_error():
    """A plotting call without the 'plot' extra raises a clear ImportError."""
    if importlib.util.find_spec("matplotlib") is not None:
        pytest.skip("matplotlib is installed; this check needs the core-only env")

    import numpy as np
    from pyoma2.setup import SingleSetup

    ss = SingleSetup(data=np.random.rand(200, 3), fs=50)
    with pytest.raises(ImportError, match=r"optional 'plot' dependencies"):
        ss.plot_data()
