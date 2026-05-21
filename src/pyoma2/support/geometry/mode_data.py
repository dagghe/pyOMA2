"""Headless builders that assemble mode-shape geometry data.

These functions map a modal result onto a geometry model and return the
:class:`~pyoma2.support.geometry.data.ModeGeo1Data` /
:class:`~pyoma2.support.geometry.data.ModeGeo2Data` Pydantic models. They are
headless — numpy / pandas / pydantic and ``gen.dfphi_map_func`` only — so they
run in a core (GUI-free) install.
"""

from __future__ import annotations

import typing

import numpy as np

from pyoma2.functions.gen import dfphi_map_func

from .data import Geometry1, Geometry2, ModeGeo1Data, ModeGeo2Data

if typing.TYPE_CHECKING:
    from pyoma2.algorithms.data.result import BaseResult


def build_mode_geo1_data(
    geo1: Geometry1,
    res: BaseResult,
    mode_nr: int,
    scaleF: float = 1.0,
) -> ModeGeo1Data:
    """Assemble :class:`ModeGeo1Data` for one mode of a Geometry1 setup.

    Parameters
    ----------
    geo1 : Geometry1
        The geometry-1 model (sensor coordinates, directions, connectivity).
    res : BaseResult
        Modal result; ``res.Phi`` is ``(Nch, Nmodes)`` and ``res.Fn`` is
        ``(Nmodes,)``.
    mode_nr : int
        Mode number to extract (1-based).
    scaleF : float, default 1.0
        Displacement scale factor applied when computing ``deformed_coord``.

    Returns
    -------
    ModeGeo1Data
        The assembled headless mode-shape data for geo1.
    """
    idx = int(mode_nr) - 1
    phi = res.Phi[:, idx].real
    fn = res.Fn[idx]
    sens_coord = geo1.sens_coord[["x", "y", "z"]].to_numpy()
    sens_dir = geo1.sens_dir
    mode_displ = sens_dir * phi.reshape(-1, 1)
    deformed_coord = sens_coord + mode_displ * scaleF
    return ModeGeo1Data(
        sens_names=geo1.sens_names,
        sens_coord=sens_coord,
        sens_dir=sens_dir,
        phi=phi,
        mode_displ=mode_displ,
        deformed_coord=deformed_coord,
        fn=float(fn),
        mode_nr=int(mode_nr),
        scaleF=float(scaleF),
        sens_lines=geo1.sens_lines,
        bg_nodes=geo1.bg_nodes,
        bg_lines=geo1.bg_lines,
        bg_surf=geo1.bg_surf,
    )


def build_mode_geo2_data(
    geo2: Geometry2,
    res: BaseResult,
    mode_nr: int,
    scaleF: float = 1.0,
) -> ModeGeo2Data:
    """Assemble :class:`ModeGeo2Data` for one mode of a Geometry2 setup.

    Parameters
    ----------
    geo2 : Geometry2
        The geometry-2 model (point coordinates, sensor mapping, signs,
        constraints, connectivity).
    res : BaseResult
        Modal result; ``res.Phi`` is ``(Nch, Nmodes)`` and ``res.Fn`` is
        ``(Nmodes,)``.
    mode_nr : int
        Mode number to extract (1-based).
    scaleF : float, default 1.0
        Displacement scale factor; pre-multiplied into ``phi`` (matching the
        geo2 plotters), so ``deformed_coord = pts_coord + mode_displ``.

    Returns
    -------
    ModeGeo2Data
        The assembled headless mode-shape data for geo2.
    """
    idx = int(mode_nr) - 1
    phi = res.Phi[:, idx].real * scaleF
    fn = res.Fn[idx]
    pts_coord = geo2.pts_coord.to_numpy()
    df_phi_map = dfphi_map_func(phi, geo2.sens_names, geo2.sens_map, cstrn=geo2.cstrn)
    mode_displ = df_phi_map.to_numpy() * geo2.sens_sign.to_numpy()
    deformed_coord = pts_coord + mode_displ
    displ_magnitude = np.linalg.norm(mode_displ, axis=1)
    return ModeGeo2Data(
        sens_names=geo2.sens_names,
        pts_coord=pts_coord,
        sens_map=geo2.sens_map,
        sens_sign=geo2.sens_sign,
        phi=phi,
        df_phi_map=df_phi_map,
        mode_displ=mode_displ,
        deformed_coord=deformed_coord,
        displ_magnitude=displ_magnitude,
        fn=float(fn),
        mode_nr=int(mode_nr),
        scaleF=float(scaleF),
        cstrn=geo2.cstrn,
        sens_lines=geo2.sens_lines,
        sens_surf=geo2.sens_surf,
        bg_nodes=geo2.bg_nodes,
        bg_lines=geo2.bg_lines,
        bg_surf=geo2.bg_surf,
    )
