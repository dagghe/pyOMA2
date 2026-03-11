"""Tests for GeometryMixin ndarray input normalization (issues #41, #45)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyoma2.setup import SingleSetup


@pytest.fixture()
def ss() -> SingleSetup:
    """Minimal SingleSetup for geometry tests."""
    data = np.random.default_rng(42).random((100, 3))
    return SingleSetup(data=data, fs=100)


class TestDefGeo1:
    """Tests for def_geo1 with various input types."""

    def _make_geo1_inputs(self):
        """Create minimal valid geo1 inputs."""
        sens_names = ["s1", "s2", "s3"]
        sens_coord = pd.DataFrame(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            index=["s1", "s2", "s3"],
            columns=["x", "y", "z"],
        )
        sens_dir = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        return sens_names, sens_coord, sens_dir

    def test_with_ndarray_sens_dir(self, ss: SingleSetup) -> None:
        """Issue #41/#45: def_geo1 should accept ndarray for sens_dir."""
        sens_names, sens_coord, sens_dir = self._make_geo1_inputs()
        ss.def_geo1(
            sens_names=sens_names,
            sens_coord=sens_coord,
            sens_dir=sens_dir,
        )
        assert ss.geo1 is not None
        assert ss.geo1.sens_names == sens_names
        np.testing.assert_array_equal(ss.geo1.sens_dir, sens_dir)

    def test_with_dataframe_sens_dir(self, ss: SingleSetup) -> None:
        """def_geo1 should still work with DataFrame for sens_dir."""
        sens_names, sens_coord, sens_dir_arr = self._make_geo1_inputs()
        sens_dir_df = pd.DataFrame(
            sens_dir_arr,
            index=sens_coord.index,
            columns=sens_coord.columns,
        )
        ss.def_geo1(
            sens_names=sens_names,
            sens_coord=sens_coord,
            sens_dir=sens_dir_df,
        )
        assert ss.geo1 is not None
        assert ss.geo1.sens_names == sens_names

    def test_with_ndarray_optional_params(self, ss: SingleSetup) -> None:
        """def_geo1 should accept ndarrays for optional params."""
        sens_names, sens_coord, sens_dir = self._make_geo1_inputs()
        # Use 1-indexed values: check_on_geo1 converts to 0-indexed
        sens_lines = np.array([[1, 2], [2, 3]])
        bg_nodes = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        bg_lines = np.array([[1, 2]])
        ss.def_geo1(
            sens_names=sens_names,
            sens_coord=sens_coord,
            sens_dir=sens_dir,
            sens_lines=sens_lines,
            bg_nodes=bg_nodes,
            bg_lines=bg_lines,
        )
        assert ss.geo1 is not None
        # Values are adjusted to 0-indexed by check_on_geo1
        np.testing.assert_array_equal(ss.geo1.sens_lines, np.array([[0, 1], [1, 2]]))
        np.testing.assert_array_equal(ss.geo1.bg_nodes, bg_nodes)
        np.testing.assert_array_equal(ss.geo1.bg_lines, np.array([[0, 1]]))

    def test_with_none_optional_params(self, ss: SingleSetup) -> None:
        """def_geo1 should work with None optional params."""
        sens_names, sens_coord, sens_dir = self._make_geo1_inputs()
        ss.def_geo1(
            sens_names=sens_names,
            sens_coord=sens_coord,
            sens_dir=sens_dir,
        )
        assert ss.geo1 is not None
        assert ss.geo1.sens_lines is None
        assert ss.geo1.bg_nodes is None


class TestDefGeo2:
    """Tests for def_geo2 with various input types."""

    def _make_geo2_inputs(self):
        """Create minimal valid geo2 inputs."""
        sens_names = ["s1", "s2"]
        pts_coord = pd.DataFrame(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            index=["p1", "p2"],
            columns=["x", "y", "z"],
        )
        sens_map = pd.DataFrame(
            [["s1", 0, 0], [0, "s2", 0]],
            index=pts_coord.index,
            columns=pts_coord.columns,
        )
        return sens_names, pts_coord, sens_map

    def test_with_ndarray_optional_params(self, ss: SingleSetup) -> None:
        """def_geo2 should accept ndarrays for optional params."""
        sens_names, pts_coord, sens_map = self._make_geo2_inputs()
        bg_nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        # Use 1-indexed: check_on_geo2 converts to 0-indexed
        bg_lines = np.array([[1, 2]])
        ss.def_geo2(
            sens_names=sens_names,
            pts_coord=pts_coord,
            sens_map=sens_map,
            bg_nodes=bg_nodes,
            bg_lines=bg_lines,
        )
        assert ss.geo2 is not None
        np.testing.assert_array_equal(ss.geo2.bg_nodes, bg_nodes)
        np.testing.assert_array_equal(ss.geo2.bg_lines, np.array([[0, 1]]))

    def test_with_none_optional_params(self, ss: SingleSetup) -> None:
        """def_geo2 should work with None optional params."""
        sens_names, pts_coord, sens_map = self._make_geo2_inputs()
        ss.def_geo2(
            sens_names=sens_names,
            pts_coord=pts_coord,
            sens_map=sens_map,
        )
        assert ss.geo2 is not None
        assert ss.geo2.bg_nodes is None
