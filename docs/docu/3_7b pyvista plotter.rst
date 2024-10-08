The ``pyvista_plotter`` module
------------------------------

This module, part of the pyOMA2 package, provides tools for visualizing and
animating mode shapes derived from Operational Modal Analysis (OMA). It offers a flexible interface
to create interactive 3D visualizations of mode shapes, incorporating both structural geometry and
mode shape data from OMA results. The module leverages `PyVista` for rich 3D visualizations and
supports saving animations as GIFs.

Classes:
    :class:`.PvGeoPlotter`: A class to plot and animate mode shapes in 3D.
    It takes geometry and result objects as inputs and provides functionalities to visualize mode
    shapes with various customizable options such as scale factor, view type, and others.

.. Warning::
    The module is designed for use within the pyOMA2 package. It requires OMA results and
    geometry data specific to the structures being analyzed.

.. automodule:: pyoma2.support.geometry.pyvista_plotter
   :members:
   :show-inheritance:
