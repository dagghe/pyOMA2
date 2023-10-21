import abc
import typing

from pydantic import (
    validate_call,
)  # controlla che i parametri passati siano quelli giusti

from .result import BaseResult
from .run_params import BaseRunParams


class BaseAlgorithm(abc.ABC):
    """Abstract class for Modal Analysis algorithms"""

    def __init__(
        self,
        run_params: BaseRunParams,
        name: typing.Optional[str] = None,
    ):
        self.run_params = run_params
        self.name = name or self.__class__.__name__
        self.result: BaseResult

    @abc.abstractmethod
    def run(self) -> typing.Any:
        """Run main algorithm using self.run_params"""

    @abc.abstractmethod
    def get_modes(self, *args, **kwargs) -> typing.Any:
        """Return modes"""
        if not self.result:
            raise ValueError("Run algorithm first")

    @abc.abstractmethod
    def select_peaks(self, *args, **kwargs) -> typing.Any:
        """Select peaks"""
        if not self.result:
            raise ValueError("Run algorithm first")

    @abc.abstractmethod
    def mod_ex(self, *args, **kwargs) -> typing.Any:
        if not self.result:
            raise ValueError("Run algorithm first")


class FDDAlgorithm(BaseAlgorithm):
    def run(self) -> typing.Any:
        print(self.run_params)

    @validate_call
    def get_modes(self, sel_freq: float, ndf: int = 5) -> typing.Any:
        super().get_modes(sel_freq=sel_freq, ndf=ndf)

    @validate_call
    def select_peaks(
        self, freqlim: typing.Optional[float] = None, ndf: int = 5
    ) -> typing.Any:
        super().select_peaks(freqlim=freqlim, ndf=ndf)

    @validate_call
    def mod_ex(self, ndf: int = 5) -> typing.Any:
        super().mod_ex(ndf=ndf)


class SSIcovAlgorithm(BaseAlgorithm):
    def run(self) -> typing.Any:
        print(self.run_params)

    @validate_call
    def get_modes(self, sel_freq: float, order: str = "find_min") -> typing.Any:
        super().get_modes(sel_freq=sel_freq, order=order)

    @validate_call
    def select_peaks(
        self,
        freqlim: typing.Optional[float] = None,
        ordmin: int = 0,
        ordmax: typing.Optional[int] = None,
        method: str = "1",
    ) -> typing.Any:
        super().select_peaks(
            freqlim=freqlim,
            ordmin=ordmin,
            ordmax=ordmax,
            method=method,
        )

    @validate_call
    def mod_ex(self, *args, **kwargs) -> typing.Any:
        super().mod_ex(*args, **kwargs)


"""...same for other alghorithms"""
