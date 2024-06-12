from typing import Any

import pytest
from pyoma2.algorithms import BaseAlgorithm
from pyoma2.algorithms.data.run_params import BaseRunParams
from pyoma2.setup import SingleSetup


def test_child_algo_must_define_run_param_cls():
    """
    Check that a subclass of BaseAlgorithm must define RunParamCls
    """
    with pytest.raises(ValueError) as excinfo:
        # Attempt to define or instantiate a subclass of BaseAlgorithm
        # without defining RunParamCls
        class MyClass(BaseAlgorithm):
            def run(self):
                return super().run()

            def mpe(self, *args, **kwargs) -> Any:
                return super().mpe(*args, **kwargs)

            def mpe_fromPlot(self, *args, **kwargs) -> Any:
                return super().mpe_fromPlot(*args, **kwargs)

    assert "RunParamCls must be defined in subclasses of BaseAlgorithm" in str(
        excinfo.value
    )


def test_run_param_cls_is_subclass_of_base_run_params():
    """
    Check that RunParamCls must be a subclass of BaseRunParams
    """
    with pytest.raises(ValueError) as excinfo:
        # Attempt to define or instantiate a subclass of BaseAlgorithm
        # with a RunParamCls that is not a subclass of BaseRunParams
        class MyClass(BaseAlgorithm):
            RunParamCls = object

            def run(self):
                return super().run()

            def mpe(self, *args, **kwargs) -> Any:
                return super().mpe(*args, **kwargs)

            def mpe_fromPlot(self, *args, **kwargs) -> Any:
                return super().mpe_fromPlot(*args, **kwargs)

    assert "RunParamCls must be defined in subclasses of BaseAlgorithm" in str(
        excinfo.value
    )


def test_child_algo_must_define_result_cls():
    """
    Check that a subclass of BaseAlgorithm must define ResultCls
    """
    with pytest.raises(ValueError) as excinfo:
        # Attempt to define or instantiate a subclass of BaseAlgorithm without defining ResultCls
        class MyClass(BaseAlgorithm):
            RunParamCls = BaseRunParams

            def run(self):
                return super().run()

            def mpe(self, *args, **kwargs) -> Any:
                return super().mpe(*args, **kwargs)

            def mpe_fromPlot(self, *args, **kwargs) -> Any:
                return super().mpe_fromPlot(*args, **kwargs)

    assert "ResultCls must be defined in subclasses of BaseAlgorithm" in str(
        excinfo.value
    )


def test_result_cls_is_subclass_of_base_result():
    """
    Check that ResultCls must be a subclass of BaseResult
    """
    with pytest.raises(ValueError) as excinfo:
        # Attempt to define or instantiate a subclass of BaseAlgorit
        # with a ResultCls that is not a subclass of BaseResult
        class MyClass(BaseAlgorithm):
            RunParamCls = BaseRunParams
            ResultCls = object

            def run(self):
                return super().run()

            def mpe(self, *args, **kwargs) -> Any:
                return super().mpe(*args, **kwargs)

            def mpe_fromPlot(self, *args, **kwargs) -> Any:
                return super().mpe_fromPlot(*args, **kwargs)

    assert "ResultCls must be defined in subclasses of BaseAlgorithm" in str(
        excinfo.value
    )


def test_run_cant_be_called_without_run_param(
    fake_single_setup_fixture_no_param: SingleSetup,
):
    """
    Check that run can't be called without setting run_params
    """

    with pytest.raises(ValueError) as excinfo:
        # Attempt to call run without setting run_params
        fake_single_setup_fixture_no_param.run_all()

    assert (
        "Run parameters must be set before running the algorithm, use a Setup class to run it"
        in str(excinfo.value)
    )


def test_result_from_setup(fake_single_setup_fixture_with_param: SingleSetup):
    """
    Check that result is not none after run with the setupclass or after call set_result
    """
    assert all(
        [
            algo.result is None
            for algo in fake_single_setup_fixture_with_param.algorithms.values()
        ]
    )
    fake_single_setup_fixture_with_param.run_all()
    assert all(
        [
            algo.result is not None
            for algo in fake_single_setup_fixture_with_param.algorithms.values()
        ]
    )
