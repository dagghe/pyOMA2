from __future__ import annotations

import abc
import typing

from pyoma2.algorithms.data.result import BaseResult

from .data import BaseGeometry

if typing.TYPE_CHECKING:
    from pyoma2.algorithms.data.result import BaseResult


T_Geo = typing.TypeVar("T_Geo", bound=BaseGeometry)


class BasePlotter(typing.Generic[T_Geo], abc.ABC):
    """An abstract base class for plotting geometry and mode shapes."""

    def __init__(self, geo: T_Geo, res: typing.Optional[BaseResult] = None):
        self.geo = geo
        self.res = res

    @abc.abstractmethod
    def plot_geo(self, *args, **kwargs):
        """Plot the geometry."""

    @abc.abstractmethod
    def plot_mode(self, *args, **kwargs):
        """Plot the mode shapes."""
