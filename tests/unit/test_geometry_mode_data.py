"""Tests for the headless mode-shape data models, builders and mixin methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyoma2.support.geometry import ModeGeo1Data, ModeGeo2Data


def test_mode_geo1_data_construction() -> None:
    """ModeGeo1Data stores the geo1 mode-shape arrays; optionals default to None."""
    m = ModeGeo1Data(
        sens_names=["s1", "s2"],
        sens_coord=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        sens_dir=np.array([[1, 0, 0], [1, 0, 0]]),
        phi=np.array([0.5, 1.0]),
        mode_displ=np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        deformed_coord=np.array([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        fn=2.5,
        mode_nr=1,
        scaleF=1.0,
    )
    assert m.sens_names == ["s1", "s2"]
    assert m.mode_nr == 1
    assert m.fn == 2.5
    assert m.sens_lines is None
    assert m.bg_nodes is None
    np.testing.assert_array_equal(m.deformed_coord[1], [2.0, 0.0, 0.0])


def test_mode_geo2_data_construction() -> None:
    """ModeGeo2Data stores the geo2 mode-shape arrays; optionals default to None."""
    sens_map = pd.DataFrame([["s1", 0, 0], [0, "s2", 0]])
    sens_sign = pd.DataFrame([[1, 1, 1], [1, 1, 1]])
    df_phi_map = pd.DataFrame([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    m = ModeGeo2Data(
        sens_names=["s1", "s2"],
        pts_coord=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        sens_map=sens_map,
        sens_sign=sens_sign,
        phi=np.array([0.5, 1.0]),
        df_phi_map=df_phi_map,
        mode_displ=np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        deformed_coord=np.array([[0.5, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        displ_magnitude=np.array([0.5, 1.0]),
        fn=3.0,
        mode_nr=2,
        scaleF=1.0,
    )
    assert m.sens_names == ["s1", "s2"]
    assert m.mode_nr == 2
    assert m.cstrn is None
    assert m.sens_surf is None
    np.testing.assert_array_equal(m.displ_magnitude, [0.5, 1.0])


def test_mode_geo1_data_requires_mandatory_fields() -> None:
    """ModeGeo1Data raises a validation error when a mandatory field is missing."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModeGeo1Data(sens_names=["s1"])
