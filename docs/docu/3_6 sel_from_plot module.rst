The ``sel_from_plot`` module
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
    This module is inspired by and expands upon functionalities found in the pyEMA package [ZBGS20]_,
    offering specialized features tailored for the pyOMA2 package's requirements.

.. Warning::
    The module is designed to be used as part of the pyOMA2 package and relies on its
    internal data structures and algorithms.

.. automodule:: pyoma2.support.sel_from_plot
   :members:
   :show-inheritance:
