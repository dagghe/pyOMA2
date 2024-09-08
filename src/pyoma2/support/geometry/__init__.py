"""
This module provides classes for handling geometry-related data, specifically designed
to store and manipulate sensor and background geometry information. It includes two
classes: Geometry1 and Geometry2, each offering unique plotting capabilities:

- Geometry1 enables users to visualise mode
  shapes with arrows that represent the placement, direction, and
  magnitude of displacement for each sensor.
- Geometry2 allows for the plotting and
  animation of mode shapes, with sensors mapped to user defined
  points.

Modules:
--------
- data: Contains the definitions for Geometry1, Geometry2, and BaseGeometry classes.
- mpl_plotter: Provides Geo1MplPlotter and Geo2MplPlotter classes for plotting using Matplotlib.
- pyvista_plotter: Provides PvGeoPlotter class for 3D visualization and animation using PyVista.
- mixin: Contains mixin classes to add additional functionality to geometry classes.

Classes:
--------
- Geometry1: A class to store and manage sensor geometry data, enabling visualization of mode shapes with arrows.
- Geometry2: A class to store and manage sensor and background geometry data, allowing for plotting and animation of mode shapes.
- BaseGeometry: A base class for storing and managing sensor and background geometry data.
- Geo1MplPlotter: A class to plot Geometry1 data using Matplotlib.
- Geo2MplPlotter: A class to plot Geometry2 data using Matplotlib.
- PvGeoPlotter: A class to visualize and animate mode shapes in 3D using PyVista.
- GeometryMixin: A mixin class to add additional functionality to geometry classes.

Functions:
----------
- save_to_file: Save geometry data to a file.
- load_from_file: Load geometry data from a file.

Examples:
---------
# Example usage of Geometry1 and Geometry2
from pyoma2.support.geometry import Geometry1, Geometry2, Geo1MplPlotter, Geo2MplPlotter, PvGeoPlotter

# Define geometry data for Geometry1
geo1 = Geometry1(
    sens_names=["Sensor1", "Sensor2"],
    sens_coord=pd.DataFrame([[0, 0, 0], [1, 1, 1]], columns=["x", "y", "z"]),
    sens_dir=np.array([[1, 0, 0], [0, 1, 0]])
)

# Plot Geometry1 data using Matplotlib
plotter1 = Geo1MplPlotter(geo1)
plotter1.plot()

# Define geometry data for Geometry2
geo2 = Geometry2(
    sens_names=["Sensor1", "Sensor2"],
    pts_coord=pd.DataFrame([[0, 0, 0], [1, 1, 1]], columns=["x", "y", "z"]),
    sens_map=pd.DataFrame([["Sensor1", "Sensor2"]], columns=["Sensor1", "Sensor2"])
)

# Plot Geometry2 data using Matplotlib
plotter2 = Geo2MplPlotter(geo2)
plotter2.plot()

# Visualize and animate Geometry2 data using PyVista
pv_plotter = PvGeoPlotter(geo2)
pv_plotter.plot_geo()
pv_plotter.animate_mode(mode_nr=1, saveGIF=True)
"""

from .data import BaseGeometry, Geometry1, Geometry2
from .mixin import GeometryMixin
from .mpl_plotter import Geo1MplPlotter, Geo2MplPlotter
from .pyvista_plotter import PvGeoPlotter
