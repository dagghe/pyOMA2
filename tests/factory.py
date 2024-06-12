import typing
import unittest.mock

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyoma2.algorithms import BaseAlgorithm
from pyoma2.algorithms.data.result import BaseResult
from pyoma2.algorithms.data.run_params import BaseRunParams
from pyoma2.support.geometry import Geometry1

FakeFigure = unittest.mock.MagicMock(spec=Figure)
FakeAxes = unittest.mock.MagicMock(spec=Axes)


class FakeRunParams(BaseRunParams):
    """FakeRunParams is a subclass of BaseRunParams."""

    param1: int = 1
    param2: str = "test"


class FakeResult(BaseResult):
    """FakeResult is a subclass of BaseResult."""

    Fn: npt.ArrayLike = np.array([1.0, 2.0, 3.0])
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

    def mpe_fromPlot(self, *args, **kwargs) -> typing.Any:
        return np.array([1.0, 2.0, 3.0])

    def plot_mode_g1(
        self,
        geo1: Geometry1,
        mode_numb: int,
        scaleF: int = 1,
        view: typing.Literal["3D", "xy", "xz", "yz", "x", "y", "z"] = "3D",
        remove_fill: bool = True,
        remove_grid: bool = True,
        remove_axis: bool = True,
    ) -> typing.Any:
        return FakeFigure, FakeAxes


class FakeAlgorithm2(FakeAlgorithm):
    """FakeAlgorithm2 is a subclass of FakeAlgorithm."""
