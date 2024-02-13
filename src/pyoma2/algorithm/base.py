"""
Abstract Base Class Module used by various OMA algorithms.
Part of the pyOMA2 package
Authors:
Dag Pasca
Diego Margoni
"""

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
    Abstract base class that defines the foundational structure and common functionalities
    for various OMA algorithms in the pyOMA2 module.
    This class sets a standard interface and workflow for implementing different operational
    modal analysis algorithms.

    Type Parameters
    ----------------
    T_RunParams : BaseModel
        The type of run parameters specific to each OMA algorithm, derived from BaseModel.
    T_Result : BaseResult
        The type of results produced by the algorithm, derived from BaseResult.
    T_Data : Iterable
        The type of input data the algorithm operates on, typically an iterable structure.

    Attributes
    ----------
    result : Optional[T_Result]
        Stores the results produced by the algorithm after its execution.
    run_params : Optional[T_RunParams]
        Holds the parameters necessary to run the algorithm.
    name : Optional[str]
        The name of the algorithm, used for identification and logging purposes.
    RunParamCls : Type[T_RunParams]
        The class used to instantiate run parameters for the algorithm.
    ResultCls : Type[T_Result]
        The class used to encapsulate and represent the results of the algorithm.
    fs : Optional[float]
        The sampling frequency of the input data.
    dt : Optional[float]
        The sampling interval, derived from the sampling frequency.
    data : Optional[T_Data]
        The data on which the algorithm operates.

    Methods
    -------
    __init__(...)
        Initializes the algorithm with optional run parameters and a name.
    _pre_run(...)
        Internal method for pre-run checks and validations.
    run(...) -> T_Result
        Abstract method that executes the algorithm and returns the results.
    set_run_params(...) -> BaseAlgorithm
        Sets the run parameters for the algorithm.
    set_result(...) -> BaseAlgorithm
        Assigns the result to the algorithm after execution.
    mpe(...) -> Any
        Abstract method for extracting modal parameters from the algorithm results.
    mpe_fromPlot(...) -> Any
        Abstract method for selecting modal parameters from plots.
    set_data(...) -> BaseAlgorithm
        Sets the input data and sampling frequency for the algorithm.

    Class Methods
    --------------
    __class_getitem__(cls, item)
        Evaluates the types of `RunParamCls` and `ResultCls` at runtime.
    __init_subclass__(cls, **kwargs)
        Ensures that subclasses define `RunParamCls` and `ResultCls`.

    Notes
    -----
    - `BaseAlgorithm` is an abstract class and cannot be instantiated directly.
    - Subclasses must implement the `run`, `mpe`, and `mpe_fromPlot` methods.
    - The class is designed to be generic and flexible, accommodating various types of algorithms
      within the pyOMA2 framework.
    - The `result` attribute is only populated after the `run` method is executed.
    - Type parameters allow for strong typing and consistency in the implementation of different algorithms.
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
        functioning of the algorithm's infrastructure.
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
