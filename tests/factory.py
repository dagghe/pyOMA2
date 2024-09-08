from __future__ import annotations

import typing
import unittest.mock

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyoma2.algorithms import BaseAlgorithm
from pyoma2.algorithms.data.result import BaseResult
from pyoma2.algorithms.data.run_params import BaseRunParams

if typing.TYPE_CHECKING:
    pass

FakeFigure = unittest.mock.MagicMock(spec=Figure)
FakeAxes = unittest.mock.MagicMock(spec=Axes)


class FakeRunParams(BaseRunParams):
    """FakeRunParams is a subclass of BaseRunParams."""

    param1: int = 1
    param2: str = "test"


class FakeResult(BaseResult):
    """FakeResult is a subclass of BaseResult."""

    Fn: npt.ArrayLike = np.array([1.0, 2.0, 3.0])
    Phi: npt.ArrayLike = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
        ]
    )
    result1: int = 1
    result2: str = "test"


class FakeAlgorithm(BaseAlgorithm[FakeRunParams, FakeResult, typing.Iterable[float]]):
    """FakeAlgorithm is a subclass of BaseAlgorithm."""

    RunParamCls = FakeRunParams
    ResultCls = FakeResult

    def run(self) -> FakeResult:
        return FakeResult()

    def mpe(self, *args, **kwargs) -> typing.Any:
        return np.array([1.0, 2.0, 3.0])

    def mpe_from_plot(self, *args, **kwargs) -> typing.Any:
        return np.array([1.0, 2.0, 3.0])


class FakeAlgorithm2(FakeAlgorithm):
    """FakeAlgorithm2 is a subclass of FakeAlgorithm."""


def assert_array_equal_with_nan(arr1: npt.ArrayLike, arr2: npt.ArrayLike) -> bool:
    """Utility function to compare two arrays with NaN values.

    Args:
        arr1 (npt.ArrayLike)
        arr2 (npt.ArrayLike)

    Returns:
        bool: True if the arrays are equal, False otherwise
    """
    nan_equal = np.isnan(arr1) == np.isnan(arr2)
    allclose_equal = np.allclose(arr1[~np.isnan(arr1)], arr2[~np.isnan(arr2)])
    return np.all(nan_equal) and allclose_equal
