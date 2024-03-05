from typing import Any

import pytest

from pyoma2.algorithm import BaseAlgorithm
from pyoma2.algorithm.data.run_params import BaseRunParams


def test_child_algo_must_define_run_param_cls():
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
