import typing

import numpy as np
import numpy.typing as npt
from pyoma2.algorithm import BaseAlgorithm
from pyoma2.algorithm.data.result import BaseResult
from pyoma2.algorithm.data.run_params import BaseRunParams


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


class FakeAlgorithm2(FakeAlgorithm):
    """FakeAlgorithm2 is a subclass of FakeAlgorithm."""
