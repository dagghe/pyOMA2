from __future__ import annotations

import math

import pytest
from pyoma2._optional import require


def test_require_returns_the_imported_module():
    assert require("math", "plot") is math


def test_require_raises_friendly_importerror_naming_the_extra():
    with pytest.raises(ImportError, match=r"optional '3d' dependencies"):
        require("pyoma2._module_that_does_not_exist", "3d")


def test_require_error_includes_install_hint():
    with pytest.raises(ImportError, match=r"pip install pyOMA_2\[plot\]"):
        require("pyoma2._module_that_does_not_exist", "plot")
