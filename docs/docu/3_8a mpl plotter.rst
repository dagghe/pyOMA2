The ``mpl_plotter`` module
--------------------------

This module, part of the pyOMA2 package, is dedicated to visualising mode shapes
from Operational Modal Analysis (OMA) results. It provides an interface to create 3D
visualisations of mode shapes, integrating the geometry of the structure and the mode shape data
from OMA analysis. The module leverages `Matplotlib`'s capabilities to create visualizations that
can be interactively viewed or saved.

Classes:
    :class:`.Geo1MplPlotter`: A class to plot mode shapes in 3D specifically for geometry 1.
    It takes geometry and result objects as inputs and provides functionalities to visualise
    mode shapes with various customizable options such as scale factor, view type, and others.

    :class:`.Geo2MplPlotter`: A class to plot mode shapes in 3D specifically for geometry 2.
    It takes geometry and result objects as inputs and provides functionalities to visualise
    mode shapes with various customizable options such as scale factor, view type, and others.

.. Warning::
    The module is designed for use within the pyOMA2 package. It requires OMA results and
    geometry data specific to the structures being analyzed.

.. automodule:: pyoma2.support.geometry.mpl_plotter
   :members:
   :show-inheritance:
