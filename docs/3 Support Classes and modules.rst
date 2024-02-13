.. autosummary::
   :toctree: generated

Support classes and modules
===========================

These classes (and functions) are not instantiated or called directly by the users, but internally by the "main" classes.


**Summary:**

.. autosummary::

    pyoma2.OMA.BaseSetup
    pyoma2.algorithm.base.BaseAlgorithm
    pyoma2.algorithm.data.geometry
    pyoma2.algorithm.data.result
    pyoma2.algorithm.data.run_params
    pyoma2.plot.Sel_from_plot
    pyoma2.plot.anim_mode
    pyoma2.utils.logging_handler


The ``BaseSetup`` class
-----------------------

.. autoclass:: pyoma2.OMA.BaseSetup
   :members:
   :show-inheritance:


The ``BaseAlgorithm`` class
---------------------------

.. autoclass:: pyoma2.algorithm.base.BaseAlgorithm
   :members:
   :show-inheritance:


The ``geometry`` module
-----------------------

.. automodule:: pyoma2.algorithm.data.geometry
   :members:
   :show-inheritance:

The ``result`` module
---------------------

.. automodule:: pyoma2.algorithm.data.result
   :members:
   :show-inheritance:

The ``run_params`` module
-------------------------

.. automodule:: pyoma2.algorithm.data.run_params
   :members:
   :show-inheritance:


The ``Sel_from_plot`` module
----------------------------

This module provides interactive plotting functionalities for selecting and analyzing poles
in Operational Modal Analysis (OMA). It is designed to work with Frequency Domain Decomposition
(FDD) and Stochastic Subspace Identification (SSI) algorithms, enabling users to visually
inspect and interact with stabilization charts and plots of the singular value of the PSD matrix.
The module integrates matplotlib plots into a Tkinter GUI, allowing for intuitive interaction
such as pole selection through mouse clicks and keyboard shortcuts.

Classes:
    :class:`.SelFromPlot`: A class for creating interactive plots where users can select or
                 deselect poles for further analysis in OMA. It supports various types
                 of plots (FDD, SSI, pLSCF) and provides utilities for saving figures,
                 toggling legends, and handling user inputs through a graphical interface.

Key Features:
    - Interactive selection of poles directly from stabilization charts and PSD plots.
    - Compatibility with FDD, SSI, and pLSCF algorithm outputs.
    - Integration of matplotlib plots within a Tkinter window for enhanced user interaction.
    - Support for exporting plots and managing display settings like legends and pole visibility.

References:
    This module is inspired by and expands upon functionalities found in the pyEMA package,
    offering specialized features tailored for the pyOMA2 package's requirements.

    .. [1] Janko Slavič et al. sdypy-EMA, GitHub repository,
        https://github.com/sdypy/sdypy-EMA

.. Note::
    The module is designed to be used as part of the pyOMA2 package and relies on its
    internal data structures and algorithms.

.. automodule:: pyoma2.plot.Sel_from_plot
   :members:
   :show-inheritance:


The ``anim_mode`` module
------------------------

This module, part of the pyOMA2 package, is dedicated to visualizing and animating mode shapes
from Operational Modal Analysis (OMA) results. It provides an interface to create animated 3D
visualizations of mode shapes, integrating the geometry of the structure and the mode shape data
from OMA analysis. The module leverages matplotlib's animation capabilities to create dynamic
visualizations that can be interactively viewed or saved as GIFs.

Classes:
    AniMode: A class to animate mode shapes in 3D. It takes geometry and result objects as inputs
             and provides functionalities to visualize mode shapes with various customizable
             options such as scale factor, view type, and others.

Key Features:
    - Animated 3D visualization of mode shapes based on OMA results.
    - Customizable options for scale factor, view angle, and plot aesthetics.
    - Supports saving the animation as a GIF file.
    - Interactive Tkinter window with embedded matplotlib figure for visualization.

Dependencies:
    - matplotlib for plotting and animations.
    - Tkinter for GUI components.
    - numpy and pandas for data manipulation.
    - pyOMA2's Geometry2, BaseResult, and MsPoserResult classes for accessing geometry and OMA results.

.. Note::
    The module is designed for use within the pyOMA2 package. It requires OMA results and
    geometry data specific to the structures being analyzed.

.. automodule:: pyoma2.plot.anim_mode
   :members:
   :show-inheritance:


The ``logging_handler`` module
------------------------------

.. autofunction:: pyoma2.utils.logging_handler.configure_logging
