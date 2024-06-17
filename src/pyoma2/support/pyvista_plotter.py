# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 21:25:39 2024

@author: dagpa
"""

import typing
import warnings

import numpy as np

# import numpy.typing as npt
try:
    import pyvista as pv
    import pyvistaqt as pvqt
except ImportError:
    warnings.warn(
        "Optional package 'pyvista' is not installed. Some features may not be available.",
        ImportWarning,
        stacklevel=2,
    )
    warnings.warn(
        "Install 'pyvista' with 'pip install pyvista' or 'pip install pyoma_2[pyvista]'",
        ImportWarning,
        stacklevel=2,
    )
    pv = None
    pvqt = None
from pyoma2.algorithms.data.result import BaseResult, MsPoserResult
from pyoma2.functions import gen
from pyoma2.support.geometry import Geometry2


class PvGeoPlotter:
    """ """

    def __init__(
        self,
        Geo: Geometry2,
        Res: typing.Union[BaseResult, MsPoserResult] = None,
    ) -> None:
        self.Geo = Geo
        self.Res = Res

        if pv is None or pvqt is None:
            raise ImportError(
                "Optional package 'pyvista' is not installed. Some features may not be available."
                "Install 'pyvista' with 'pip install pyvista' or 'pip install pyoma_2[pyvista]'"
            )

    def plot_geo(
        self,
        scaleF=1,
        col_sens="red",
        plot_points=True,
        points_sett="default",
        plot_lines=True,
        lines_sett="default",
        plot_surf=True,
        surf_sett="default",
        pl=None,
        bg_plotter: bool = True,
        notebook: bool = False,
    ):
        # import geometry
        Geo = self.Geo

        # define the plotter object type
        if pl is None:
            if notebook:
                pl = pv.Plotter(notebook=True)
            elif bg_plotter:
                pl = pvqt.BackgroundPlotter()
            else:
                pl = pv.Plotter()

        # define default settings for plot
        undef_sett = dict(
            color="gray",
            opacity=0.7,
        )

        if points_sett == "default":
            points_sett = undef_sett

        if lines_sett == "default":
            lines_sett = undef_sett

        if surf_sett == "default":
            surf_sett = undef_sett

        # GEOMETRY
        points = Geo.pts_coord.to_numpy()
        lines = Geo.sens_lines
        surfs = Geo.sens_surf
        # geometry in pyvista format
        if lines is not None:
            lines = np.array([np.hstack([2, line]) for line in lines])
        if surfs is not None:
            surfs = np.array([np.hstack([3, surf]) for surf in surfs])

        # PLOTTING
        if plot_points:
            pl.add_points(points, **points_sett)
        if plot_lines:
            line_mesh = pv.PolyData(points, lines=lines)
            pl.add_mesh(line_mesh, **lines_sett)
        if plot_surf:
            face_mesh = pv.PolyData(points, faces=surfs)
            pl.add_mesh(face_mesh, **surf_sett)

        # # Add axes
        # pl.add_axes(line_width=5, labels_off=False)
        # pl.show()

        # add sensor points + arrows for direction
        sens_names = Geo.sens_names
        ch_names = Geo.sens_map.to_numpy()
        ch_names = np.array(
            [name if name in sens_names else "nan" for name in ch_names.flatten()]
        ).reshape(ch_names.shape)

        ch_names_fl = ch_names.flatten()[ch_names.flatten() != "nan"]
        ch_names_fl = [str(ele) for ele in ch_names_fl]
        # Plot points where ch_names_1 is not np.nan
        valid_indices = ch_names != "nan"  # FIXME
        valid_points = points[np.any(valid_indices, axis=1)]

        pl.add_points(
            valid_points,
            render_points_as_spheres=True,
            color=col_sens,
            point_size=10,
        )

        points_new = []
        directions = []
        for i, (row1, row2) in enumerate(zip(ch_names, points)):
            for j, elem in enumerate(row1):
                if elem != "nan":
                    vector = [0, 0, 0]
                    # vector[j] = 1
                    vector[j] = Geo.sens_sign.values[i, j]
                    directions.append(vector)
                    points_new.append(row2)

        points_new = np.array(points_new)
        directions = np.array(directions)
        # Add arrow to plotter
        pl.add_arrows(points_new, directions, mag=scaleF, color=col_sens)
        pl.add_point_labels(
            points_new + directions * scaleF,
            ch_names_fl,
            font_size=20,
            always_visible=True,
            shape_color="white",
        )

        # Add axes
        pl.add_axes(line_width=5, labels_off=False)
        pl.show()

        return pl

    def plot_mode(
        self,
        mode_nr: int = 1,
        scaleF: float = 1.0,
        plot_lines: bool = True,
        plot_surf: bool = True,
        plot_undef: bool = True,
        def_sett: dict = "default",
        undef_sett: dict = "default",
        pl=None,
        bg_plotter: bool = True,
        notebook: bool = False,
    ):
        # import geometry and results
        Geo = self.Geo
        Res = self.Res

        # define the plotter object type
        if pl is None:
            if notebook:
                pl = pv.Plotter(notebook=True)
            elif bg_plotter:
                pl = pvqt.BackgroundPlotter()
            else:
                pl = pv.Plotter()

        # define default settings for plot
        def_settings = dict(cmap="plasma", opacity=0.7, show_scalar_bar=False)
        undef_settings = dict(color="gray", opacity=0.3)

        if def_sett == "default":
            def_sett = def_settings

        if undef_sett == "default":
            undef_sett = undef_settings

        # GEOMETRY
        points = Geo.pts_coord.to_numpy()
        lines = Geo.sens_lines
        surfs = Geo.sens_surf
        # geometry in pyvista format
        if lines is not None:
            lines = np.array([np.hstack([2, line]) for line in lines])
        if surfs is not None:
            surfs = np.array([np.hstack([3, surf]) for surf in surfs])

        # Mode shape
        if Res is not None:
            phi = Res.Phi[:, int(mode_nr - 1)].real * scaleF
        else:
            raise ValueError("You must pass the Res class to plot a mode shape!")

        # APPLY POINTS TO SENSOR MAPPING
        df_phi_map = gen.dfphi_map_func(
            phi, Geo.sens_names, Geo.sens_map, cstrn=Geo.cstrn
        )
        # calculate deformed shape (NEW POINTS)
        newpoints = points + df_phi_map.to_numpy() * Geo.sens_sign.to_numpy()

        # If true plot undeformed shape
        if plot_undef:
            pl.add_points(points, **undef_sett)
            if plot_lines:
                line_mesh = pv.PolyData(points, lines=lines)
                pl.add_mesh(line_mesh, **undef_sett)
            if plot_surf:
                face_mesh = pv.PolyData(points, faces=surfs)
                pl.add_mesh(face_mesh, **undef_sett)

        # PLOT MODE SHAPE
        pl.add_points(newpoints, scalars=df_phi_map.values, **def_sett)
        if plot_lines:
            line_mesh = pv.PolyData(newpoints, lines=lines)
            pl.add_mesh(line_mesh, scalars=df_phi_map.values, **def_sett)
        if plot_surf:
            face_mesh = pv.PolyData(newpoints, faces=surfs)
            pl.add_mesh(face_mesh, scalars=df_phi_map.values, **def_sett)

        pl.add_text(
            rf"Mode nr. {mode_nr}, fn = {Res.Fn[mode_nr-1]:.3f}Hz",
            position="upper_edge",
            color="black",
            # font_size=26,
        )
        pl.add_axes(line_width=5, labels_off=False)
        pl.show()

        return pl

    def animate_mode(
        self,
        mode_nr: int = 1,
        scaleF: float = 1.0,
        plot_lines: bool = True,
        plot_surf: bool = True,
        def_sett: dict = "default",
        saveGIF: bool = False,
        pl=None,
    ):
        # define default settings for plot
        def_settings = dict(cmap="plasma", opacity=0.7, show_scalar_bar=False)

        if def_sett == "default":
            def_sett = def_settings

        # import geometry and results
        Geo = self.Geo
        Res = self.Res
        points = pv.pyvista_ndarray(Geo.pts_coord.to_numpy())
        lines = Geo.sens_lines
        surfs = Geo.sens_surf
        # geometry in pyvista format
        if lines is not None:
            lines = np.array([np.hstack([2, line]) for line in lines])
        if surfs is not None:
            surfs = np.array([np.hstack([3, surf]) for surf in surfs])

        # Mode shape
        phi = Res.Phi[:, int(mode_nr - 1)].real * scaleF

        # mode shape mapped to points
        df_phi_map = gen.dfphi_map_func(
            phi, Geo.sens_names, Geo.sens_map, cstrn=Geo.cstrn
        )
        # add together coordinates and mode shape displacement
        # newpoints = (points + df_phi_map.to_numpy() * Geo.sens_sign.to_numpy() )

        # copy the dataset as we will modify its coordinates
        points_c = points.copy()

        if pl is None:
            pl = pv.Plotter(off_screen=False) if saveGIF else pvqt.BackgroundPlotter()

        # PLOT MODE SHAPE
        def_pts = pl.add_points(points_c, scalars=df_phi_map.values, **def_sett)

        if plot_lines:
            line_mesh = pv.PolyData(points_c, lines=lines)
            pl.add_mesh(line_mesh, scalars=df_phi_map.values, **def_sett)
        if plot_surf:
            face_mesh = pv.PolyData(points_c, faces=surfs)
            pl.add_mesh(face_mesh, scalars=df_phi_map.values, **def_sett)

        pl.add_text(
            rf"Mode nr. {mode_nr}, fn = {Res.Fn[mode_nr-1]:.3f}Hz",
            position="upper_edge",
            color="black",
            # font_size=26,
        )

        if saveGIF:
            pl.enable_anti_aliasing("fxaa")
            n_frames = 30
            pl.open_gif(f"Mode nr. {mode_nr}.gif")
            for phase in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
                def_pts.mapper.dataset.points = points + df_phi_map.to_numpy() * np.cos(
                    phase
                )
                line_mesh.points = points + df_phi_map.to_numpy() * np.cos(phase)
                face_mesh.points = points + df_phi_map.to_numpy() * np.cos(phase)
                pl.add_axes(line_width=5, labels_off=False)
                pl.write_frame()
            pl.show(auto_close=False)
        else:

            def update_shape():
                n_frames = 30
                for phase in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
                    def_pts.mapper.dataset.points = (
                        points + df_phi_map.to_numpy() * np.cos(phase)
                    )
                    line_mesh.points = points + df_phi_map.to_numpy() * np.cos(phase)
                    face_mesh.points = points + df_phi_map.to_numpy() * np.cos(phase)
                    pl.add_axes(line_width=5, labels_off=False)
                    pl.update()

            pl.add_callback(update_shape, interval=100)
            # pl.show()
        # pl.close()


# # =============================================================================
# # TEST
# # =============================================================================
# _path=r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyomaTEST\HTC_geom\geo2.xlsx"
# _file=r"X:\OneDrive - Norsk Treteknisk Institutt\Dokumenter\Dev\pyomaTEST\HTC_geom\PHI.npy"
# ref_ind = [[4, 5], [6, 7], [6, 7], [6, 7]]
# Phi=np.load(_file)
# Geo = Gen_funct.import_excel_GEO2(_path,ref_ind)
# Geo = Geometry2(
#             sens_names=Geo[0],
#             pts_coord=Geo[1],
#             sens_map=Geo[2],
#             cstrn=Geo[3],
#             sens_sign=Geo[4],
#             sens_lines=Geo[5],
#             sens_surf=Geo[6],
#             bg_nodes=Geo[7],
#             bg_lines=Geo[8],
#             bg_surf=Geo[9],
#         )

# Res = BaseResult(
#     Fn= np.arange(Phi.shape[1]),
#     Phi=Phi)


# Plotter = pv_GeoPlotter(Geo,Res)

# Plotter.plot_mode(mode_nr=7,scaleF=8000,
# #                       def_sett=dict(cmap="magma", opacity=0.7,show_scalar_bar=False),
# #                       undef_sett=dict(color="gray", opacity=0.1,),
#                       )

# Plotter.animate_mode(mode_nr=7,scaleF=10000,
#                       # def_sett=dict(cmap="magma", opacity=0.7,show_scalar_bar=False),
#                       # undef_sett=dict(color="gray", opacity=0.1,),
#                       )
