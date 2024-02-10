from __future__ import annotations

import abc
import typing

from pydantic import BaseModel

from pyoma2.algorithm.data.result import BaseResult

T_RunParams = typing.TypeVar("T_RunParams", bound=BaseModel)
T_Result = typing.TypeVar("T_Result", bound=BaseResult)
T_Data = typing.TypeVar("T_Data", bound=typing.Iterable)


class BaseAlgorithm(typing.Generic[T_RunParams, T_Result, T_Data], abc.ABC):
    """
    This module provides the abstract base class for the algorithms. It defines
    the fundamental attributes and methods used by the other algorithms.

    Classes
    -------
    BaseAlgorithm(typing.Generic[T_RunParams, T_Result, T_Data], abc.ABC)
        An abstract base class that defines the structure and functionalities common to all
        modal analysis algorithms.

    See Also
    --------
    pydantic.BaseModel, pyoma2.algorithm.data.result.BaseResult

    Notes
    -----
    This module requires subclasses of `BaseAlgorithm` to specify their own implementations
    of run parameters and results. These implementations must be derived from `pydantic.BaseModel`
    and `pyoma2.algorithm.data.result.BaseResult`, respectively. The `BaseAlgorithm` class provides
    methods to set run parameters, results, and data necessary for the execution of the algorithm.

    Attributes
    ----------
    result : typing.Optional[T_Result]
        The result of the algorithm after execution, storing relevant modal analysis outputs.
    run_params : typing.Optional[T_RunParams]
        The parameters required to run the algorithm.
    name : typing.Optional[str]
        The name of the algorithm.
    RunParamCls : typing.Type[T_RunParams]
        The class reference for run parameters, derived from `pydantic.BaseModel`.
    ResultCls : typing.Type[T_Result]
        The class reference for result, derived from `pyoma2.algorithm.data.result.BaseResult`.
    fs : typing.Optional[float]
        The sampling frequency of the input data.
    dt : typing.Optional[float]
        The sampling interval, calculated from the sampling frequency.
    data : typing.Optional[T_Data]
        The input data for the algorithm.

    Methods
    -------
    __init__(...)
        Initializes the algorithm with the given run parameters and name.
    run(...)
        Abstract method to execute the algorithm. Subclasses must provide an implementation.
    set_run_params(...)
        Sets the run parameters for the algorithm.
    set_result(...)
        Sets the result for the algorithm.
    mpe(...)
        Abstract method to return modes. Subclasses must provide an implementation.
    mpe_fromPlot(...)
        Abstract method to select peaks from plots. Subclasses must provide an implementation.
    set_data(...)
        Sets the input data and sampling frequency for the algorithm.
    __class_getitem__(...)
        A method to evaluate the type of `RunParamCls` and `ResultCls` at runtime.
    __init_subclass__(...)
        Ensures that subclasses define `RunParamCls` and `ResultCls`.

    """

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
        """
        Initialize the algorithm with optional run parameters and a name.

        Parameters
        ----------
        run_params : typing.Optional[T_RunParams], optional
            The parameters required to run the algorithm. If not provided, can be set later.
        name : typing.Optional[str], optional
            The name of the algorithm. If not provided, defaults to the class name.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments used to instantiate run parameters if `run_params` is not provided.

        Notes
        -----
        This constructor allows flexible initialization of the algorithm. If `run_params` are not provided
        during initialization, they can be set later using the `set_run_params` method. Similarly, if a `name`
        is not provided, the class name is used as the default name of the algorithm.
        """
        if run_params:
            self.run_params = run_params
        elif kwargs:
            self.run_params = self.RunParamCls(**kwargs)

        self.name = name or self.__class__.__name__

    def _pre_run(self):
        """
        Internal method to perform pre-run checks.

        Raises
        ------
        ValueError
            If the sampling frequency (`fs`) or the input data (`data`) is not set.
            If the run parameters (`run_params`) are not set.

        Notes
        -----
        This method is called internally by the `run` method to ensure that the necessary prerequisites
        for running the algorithm are satisfied.
        """
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
        """
        Abstract method to execute the algorithm.

        This method must be implemented by all subclasses. It should use the set `run_params` and input `data`
        to perform the modal analysis and save the result in the `result` attribute.

        Returns
        -------
        T_Result
            The result of the algorithm execution.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.

        Notes
        -----
        Implementing classes should handle the algorithm logic within this method and ensure that the
        output is an instance of the `ResultCls`.
        """
        self._pre_run()

    def set_run_params(self, run_params: T_RunParams) -> "BaseAlgorithm":
        """
        Set the run parameters for the algorithm.

        Parameters
        ----------
        run_params : T_RunParams
            The run parameters for the algorithm.

        Returns
        -------
        BaseAlgorithm
            Returns the instance with updated run parameters.

        Notes
        -----
        This method allows dynamically setting or updating the run parameters for the algorithm
        after its initialization.
        """
        self.run_params = run_params
        return self

    def set_result(self, result: T_Result) -> "BaseAlgorithm":
        """
        Set the result of the algorithm.

        Parameters
        ----------
        result : T_Result
            The result obtained from running the algorithm.

        Returns
        -------
        BaseAlgorithm
            Returns the instance with the set result.

        Notes
        -----
        This method is used to assign the result after the algorithm execution. The result should be
        an instance of the `ResultCls`.
        """
        self.result = result
        return self

    @abc.abstractmethod
    def mpe(self, *args, **kwargs) -> typing.Any:
        """
        Abstract method to return the modal parameters extracted by the algorithm.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        typing.Any
            The modal parameters extracted by the algorithm.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        ValueError
            If the algorithm has not been run or the result is not set.

        Notes
        -----
        Implementing classes should override this method to provide functionality for extracting
        and returning modal parameters based on the algorithm's results.
        """
        # METODO 2 (manuale)
        if not self.result:
            raise ValueError("Run algorithm first")

    @abc.abstractmethod
    def mpe_fromPlot(self, *args, **kwargs) -> typing.Any:
        """
        Abstract method to select peaks or modal parameters from plots.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        typing.Any
            The selected peaks or modal parameters.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        ValueError
            If the algorithm has not been run or the result is not set.

        Notes
        -----
        Implementing classes should provide mechanisms for selecting and returning peaks or modal parameters
        from graphical plots or visual representations of the data.
        """
        # METODO 2 (grafico)
        if not self.result:
            raise ValueError(f"{self.name}:Run algorithm first")

    def set_data(self, data: T_Data, fs: float) -> "BaseAlgorithm":
        """
        Set the input data and sampling frequency for the algorithm.

        Parameters
        ----------
        data : T_Data
            The input data for the algorithm.
        fs : float
            The sampling frequency of the data.

        Returns
        -------
        BaseAlgorithm
            Returns the instance with the set data and sampling frequency.

        Notes
        -----
        This method is typically used by the Setup class to provide the necessary data and sampling
        frequency to the algorithm before its execution.
        """
        self.data = data
        self.fs = fs
        self.dt = 1 / fs
        return self

    def __class_getitem__(cls, item):
        """
        Class method to evaluate the types of `RunParamCls` and `ResultCls` at runtime.

        This method dynamically sets the `RunParamCls` and `ResultCls` class attributes based on the
        provided `item` types.

        Parameters
        ----------
        item : tuple
            A tuple containing the types for `RunParamCls` and `ResultCls`.

        Returns
        -------
        cls : BaseAlgorithm
            The class with evaluated `RunParamCls` and `ResultCls`.

        Notes
        -----
        This class method is a workaround to dynamically determine the types of `RunParamCls` and `ResultCls`
        at runtime. It is particularly useful for type checking and ensuring consistency across different
        subclasses of `BaseAlgorithm`.
        """
        # tricky way to evaluate at runtime the type of the RunParamCls and ResultCls
        cls.RunParamCls = item[0]
        cls.ResultCls = item[1]
        return cls

    def __init_subclass__(cls, **kwargs):
        """
        Initialize subclass of `BaseAlgorithm`.

        This method ensures that subclasses of `BaseAlgorithm` define `RunParamCls` and `ResultCls`.

        Raises
        ------
        ValueError
            If `RunParamCls` or `ResultCls` are not defined or not subclasses of `BaseModel` and `BaseResult`,
            respectively.

        Notes
        -----
        This method is automatically called when a subclass of `BaseAlgorithm` is defined. It checks that
        `RunParamCls` and `ResultCls` are correctly set in the subclass. This is essential for the proper
        functioning
        of the algorithm's infrastructure.
        """
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "RunParamCls") or not issubclass(cls.RunParamCls, BaseModel):
            raise ValueError(
                f"{cls.__name__}: RunParamCls must be defined in subclasses of BaseAlgorithm"
            )
        if not hasattr(cls, "ResultCls") or not issubclass(cls.ResultCls, BaseResult):
            raise ValueError(
                f"{cls.__name__}: ResultCls must be defined in subclasses of BaseResult"
            )
