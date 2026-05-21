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


def _result(phi, fn):
    """A minimal BaseResult with the given Phi (Nch, Nmodes) and Fn (Nmodes,)."""
    from pyoma2.algorithms.data.result import BaseResult

    return BaseResult(Fn=np.asarray(fn, dtype=float), Phi=np.asarray(phi))


def _make_geo1_setup():
    """A SingleSetup with a defined geo1 (3 sensors, sens_dir = identity)."""
    from pyoma2.setup import SingleSetup

    ss = SingleSetup(data=np.random.default_rng(0).random((100, 3)), fs=100)
    ss.def_geo1(
        sens_names=["s1", "s2", "s3"],
        sens_coord=pd.DataFrame(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            index=["s1", "s2", "s3"],
            columns=["x", "y", "z"],
        ),
        sens_dir=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )
    return ss


def test_build_mode_geo1_data_displacement() -> None:
    """build_mode_geo1_data: mode_displ = sens_dir*phi, deformed = coord + displ*scaleF."""
    from pyoma2.support.geometry.mode_data import build_mode_geo1_data

    ss = _make_geo1_setup()
    res = _result(phi=np.array([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0]]), fn=[2.5, 4.0])
    data = build_mode_geo1_data(ss.geo1, res, mode_nr=1, scaleF=10.0)

    np.testing.assert_allclose(data.phi, [1.0, 2.0, 3.0])
    # sens_dir is the identity -> mode_displ = diag(phi)
    np.testing.assert_allclose(
        data.mode_displ, [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    np.testing.assert_allclose(
        data.deformed_coord,
        ss.geo1.sens_coord[["x", "y", "z"]].to_numpy() + data.mode_displ * 10.0,
    )
    assert data.fn == 2.5
    assert data.mode_nr == 1
    assert data.scaleF == 10.0


def test_build_mode_geo1_data_uses_real_part() -> None:
    """build_mode_geo1_data takes the real part of a complex Phi."""
    from pyoma2.support.geometry.mode_data import build_mode_geo1_data

    ss = _make_geo1_setup()
    res = _result(phi=np.array([[1.0 + 5.0j], [2.0 - 1.0j], [3.0 + 0.0j]]), fn=[2.5])
    data = build_mode_geo1_data(ss.geo1, res, mode_nr=1)
    np.testing.assert_allclose(data.phi, [1.0, 2.0, 3.0])
    assert data.scaleF == 1.0


def test_build_mode_geo1_data_rejects_out_of_range_mode_nr() -> None:
    """build_mode_geo1_data rejects mode_nr outside 1..n_modes."""
    from pyoma2.support.geometry.mode_data import build_mode_geo1_data

    ss = _make_geo1_setup()
    res = _result(phi=np.array([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0]]), fn=[2.5, 4.0])
    # mode_nr=0 would otherwise negative-index to the last mode
    with pytest.raises(ValueError, match="mode_nr must be between 1 and 2"):
        build_mode_geo1_data(ss.geo1, res, mode_nr=0)
    with pytest.raises(ValueError, match="mode_nr must be between 1 and 2"):
        build_mode_geo1_data(ss.geo1, res, mode_nr=3)


def test_get_mode_geo1_data_guards() -> None:
    """get_mode_geo1_data raises before geo1 is defined / before a run."""
    from pyoma2.algorithms.data.result import BaseResult
    from pyoma2.setup import SingleSetup

    ss = SingleSetup(data=np.random.default_rng(0).random((100, 3)), fs=100)
    with pytest.raises(ValueError, match="geo1 is not defined"):
        ss.get_mode_geo1_data(_result(phi=np.zeros((3, 1)), fn=[1.0]), mode_nr=1)

    ss2 = _make_geo1_setup()
    with pytest.raises(ValueError, match="Run algorithm first"):
        ss2.get_mode_geo1_data(BaseResult(), mode_nr=1)


def test_get_mode_geo1_data_returns_model() -> None:
    """get_mode_geo1_data returns a ModeGeo1Data assembled by the builder."""
    ss = _make_geo1_setup()
    res = _result(phi=np.array([[1.0], [2.0], [3.0]]), fn=[2.5])
    data = ss.get_mode_geo1_data(res, mode_nr=1, scaleF=2.0)
    assert isinstance(data, ModeGeo1Data)
    assert data.scaleF == 2.0
    np.testing.assert_allclose(data.deformed_coord[2], [2.0, 0.0, 6.0])


def _make_geo2_setup():
    """A SingleSetup with a defined geo2 (2 sensors, 2 points, no constraints)."""
    from pyoma2.setup import SingleSetup

    ss = SingleSetup(data=np.random.default_rng(0).random((100, 2)), fs=100)
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
    ss.def_geo2(sens_names=["s1", "s2"], pts_coord=pts_coord, sens_map=sens_map)
    return ss


def test_build_mode_geo2_data_displacement() -> None:
    """build_mode_geo2_data maps phi onto points, pre-scales it and deforms them."""
    from pyoma2.support.geometry.mode_data import build_mode_geo2_data

    ss = _make_geo2_setup()
    res = _result(phi=np.array([[0.5], [2.0]]), fn=[3.0])
    data = build_mode_geo2_data(ss.geo2, res, mode_nr=1, scaleF=2.0)

    # geo2 pre-scales phi by scaleF
    np.testing.assert_allclose(data.phi, [1.0, 4.0])
    # sens_map: point p1 axis x -> s1, point p2 axis y -> s2; sens_sign all ones
    np.testing.assert_allclose(data.mode_displ, [[1.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    np.testing.assert_allclose(data.deformed_coord, [[1.0, 0.0, 0.0], [1.0, 4.0, 0.0]])
    np.testing.assert_allclose(data.displ_magnitude, [1.0, 4.0])
    assert data.fn == 3.0
    assert data.scaleF == 2.0


def test_build_mode_geo2_data_matches_dfphi_map_func() -> None:
    """build_mode_geo2_data reuses gen.dfphi_map_func for the point mapping."""
    from pyoma2.functions.gen import dfphi_map_func
    from pyoma2.support.geometry.mode_data import build_mode_geo2_data

    ss = _make_geo2_setup()
    res = _result(phi=np.array([[0.5], [2.0]]), fn=[3.0])
    data = build_mode_geo2_data(ss.geo2, res, mode_nr=1, scaleF=2.0)

    expected = dfphi_map_func(
        np.array([1.0, 4.0]), ss.geo2.sens_names, ss.geo2.sens_map, cstrn=ss.geo2.cstrn
    )
    np.testing.assert_allclose(data.df_phi_map.to_numpy(), expected.to_numpy())


def test_build_mode_geo2_data_uses_real_part() -> None:
    """build_mode_geo2_data takes the real part of a complex Phi."""
    from pyoma2.support.geometry.mode_data import build_mode_geo2_data

    ss = _make_geo2_setup()
    res = _result(phi=np.array([[0.5 + 4.0j], [2.0 - 1.0j]]), fn=[3.0])
    data = build_mode_geo2_data(ss.geo2, res, mode_nr=1, scaleF=2.0)
    # phi = real(Phi) * scaleF = [0.5, 2.0] * 2.0
    np.testing.assert_allclose(data.phi, [1.0, 4.0])


def test_build_mode_geo2_data_rejects_out_of_range_mode_nr() -> None:
    """build_mode_geo2_data rejects mode_nr outside 1..n_modes."""
    from pyoma2.support.geometry.mode_data import build_mode_geo2_data

    ss = _make_geo2_setup()
    res = _result(phi=np.array([[0.5, 1.0], [2.0, 1.0]]), fn=[3.0, 5.0])
    # mode_nr=0 would otherwise negative-index to the last mode
    with pytest.raises(ValueError, match="mode_nr must be between 1 and 2"):
        build_mode_geo2_data(ss.geo2, res, mode_nr=0)
    with pytest.raises(ValueError, match="mode_nr must be between 1 and 2"):
        build_mode_geo2_data(ss.geo2, res, mode_nr=3)


def test_get_mode_geo2_data_guards() -> None:
    """get_mode_geo2_data raises before geo2 is defined / before a run."""
    from pyoma2.algorithms.data.result import BaseResult
    from pyoma2.setup import SingleSetup

    ss = SingleSetup(data=np.random.default_rng(0).random((100, 2)), fs=100)
    with pytest.raises(ValueError, match="geo2 is not defined"):
        ss.get_mode_geo2_data(_result(phi=np.zeros((2, 1)), fn=[1.0]), mode_nr=1)

    ss2 = _make_geo2_setup()
    with pytest.raises(ValueError, match="Run algorithm first"):
        ss2.get_mode_geo2_data(BaseResult(), mode_nr=1)


def test_get_mode_geo2_data_returns_model() -> None:
    """get_mode_geo2_data returns a ModeGeo2Data assembled by the builder."""
    ss = _make_geo2_setup()
    res = _result(phi=np.array([[0.5], [2.0]]), fn=[3.0])
    data = ss.get_mode_geo2_data(res, mode_nr=1)
    assert isinstance(data, ModeGeo2Data)
    assert data.mode_nr == 1
    assert data.scaleF == 1.0
    np.testing.assert_allclose(data.displ_magnitude, [0.5, 2.0])
