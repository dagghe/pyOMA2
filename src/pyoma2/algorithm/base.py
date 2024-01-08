import abc
import typing

from pyoma2.algorithm.data.result import BaseResult
from pyoma2.algorithm.data.run_params import BaseRunParams

T_RunParams = typing.TypeVar("T_RunParams", bound=BaseRunParams)
T_Result = typing.TypeVar("T_Result", bound=BaseResult)


# METODI PER PLOT "STATICI" DOVE SI AGGIUNGONO?
# ALLA CLASSE BASE o A QUELLA SPECIFICA?


class BaseAlgorithm(typing.Generic[T_RunParams, T_Result], abc.ABC):
    """Abstract class for Modal Analysis algorithms"""

    result: T_Result | None = None
    run_params: T_RunParams | None = None
    name: str | None = None
    RunParam: typing.Type[T_RunParams]
    ResultType: typing.Type[T_Result]

    # additional attributes set by the Setup Class
    fs: float | None = None  # sampling frequency
    dt: float | None = None  # sampling interval
    data: typing.Iterable[float] | None = None  # data

    def __init__(
        self,
        run_params: T_RunParams | None = None,
        name: typing.Optional[str] = None,
        *args, **kwargs
    ):
        """Initialize the algorithm with the run parameters"""
        if run_params:

            self.run_params = run_params
        elif kwargs:
            self.run_params = self.RunParam(**kwargs)

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

    def set_run_params(self, run_param: T_RunParams) -> "BaseAlgorithm":
        """Sets the run parameters"""
        self.run_params = run_param
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

    def set_data(self, data: typing.Iterable[float], fs: float) -> "BaseAlgorithm":
        """Set data and sampling frequency for the algorithm"""
        self.data = data
        self.fs = fs
        self.dt = 1 / fs
        return self

    def __class_getitem__(cls, item):
        # tricky way to evaluate at runtime the type of the RunParam and ResultType
        cls.RunParam = item[0]
        cls.ResultType = item[1]
        return cls

    def __init_subclass__(cls, **kwargs):
        """Check that subclasses define RunParam and ResultType"""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "RunParam") or not issubclass(
            cls.RunParam, BaseRunParams
        ):
            raise ValueError(
                f"{cls.__name__}: RunParam must be defined in subclasses of BaseAlgorithm"
            )
        if not hasattr(cls, "ResultType") or not issubclass(cls.ResultType, BaseResult):
            raise ValueError(
                f"{cls.__name__}: ResultType must be defined in subclasses of BaseResult"
            )
