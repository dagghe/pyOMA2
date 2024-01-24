from __future__ import annotations

import abc
import typing

from pydantic import BaseModel

from pyoma2.algorithm.data.result import BaseResult

T_RunParams = typing.TypeVar("T_RunParams", bound=BaseModel)
T_Result = typing.TypeVar("T_Result", bound=BaseResult)
T_Data = typing.TypeVar("T_Data", bound=typing.Iterable)


class BaseAlgorithm(typing.Generic[T_RunParams, T_Result, T_Data], abc.ABC):
    """Abstract class for Modal Analysis algorithms"""

    result: typing.Optional[T_Result] = None
    run_params: typing.Optional[T_RunParams] = None
    name: typing.Optional[str] = None
    RunParamCls: typing.Type[T_RunParams]
    ResultCls: typing.Type[T_Result]

    # additional attributes set by the Setup Class
    fs: typing.Optional[float]  # sampling frequency
    dt: typing.Optional[float]  # sampling interval
    data: typing.Optional[T_Data]  # data

    def __init__(
        self,
        run_params: typing.Optional[T_RunParams] = None,
        name: typing.Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Initialize the algorithm with the run parameters"""
        if run_params:
            self.run_params = run_params
        elif kwargs:
            self.run_params = self.RunParamCls(**kwargs)

        self.name = name or self.__class__.__name__

    def _pre_run(self):
        if self.fs is None or self.data is None:
            raise ValueError(
                f"{self.name}: Sampling frequency and data must be set before running the algorithm, "
                "use a Setup class to run it"
            )
        if not self.run_params:
            raise ValueError(
                f"{self.name}: Run parameters must be set before running the algorithm, "
                "use a Setup class to run it"
            )

    @abc.abstractmethod
    def run(self) -> T_Result:
        """Run main algorithm using self.run_params and save result in the base result"""
        self._pre_run()

    def set_run_params(self, run_params: T_RunParams) -> "BaseAlgorithm":
        """Sets the run parameters"""
        self.run_params = run_params
        return self

    def set_result(self, result: T_Result) -> "BaseAlgorithm":
        """Sets the result"""
        self.result = result
        return self

    @abc.abstractmethod
    def mpe(self, *args, **kwargs) -> typing.Any:
        """Return modes"""
        # METODO 2 (manuale)
        if not self.result:
            raise ValueError("Run algorithm first")

    @abc.abstractmethod
    def mpe_fromPlot(self, *args, **kwargs) -> typing.Any:
        """Select peaks"""
        # METODO 2 (grafico)
        if not self.result:
            raise ValueError(f"{self.name}:Run algorithm first")

    def set_data(self, data: T_Data, fs: float) -> "BaseAlgorithm":
        """Set data and sampling frequency for the algorithm"""
        self.data = data
        self.fs = fs
        self.dt = 1 / fs
        return self

    def __class_getitem__(cls, item):
        # tricky way to evaluate at runtime the type of the RunParamCls and ResultCls
        cls.RunParamCls = item[0]
        cls.ResultCls = item[1]
        return cls

    def __init_subclass__(cls, **kwargs):
        """Check that subclasses define RunParamCls and ResultCls"""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "RunParamCls") or not issubclass(cls.RunParamCls, BaseModel):
            raise ValueError(
                f"{cls.__name__}: RunParamCls must be defined in subclasses of BaseAlgorithm"
            )
        if not hasattr(cls, "ResultCls") or not issubclass(cls.ResultCls, BaseResult):
            raise ValueError(
                f"{cls.__name__}: ResultCls must be defined in subclasses of BaseResult"
            )
