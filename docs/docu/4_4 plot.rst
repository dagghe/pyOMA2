The ``plot`` module
-------------------

This module, part of the pyOMA2 package, offers a suite of plotting functions
designed specifically for use within the pyOMA2 module. These functions aid in
visualizing modal analysis data, such as time histories, frequency responses,
and stabilization charts. They facilitate the intuitive interpretation of Operational
Modal Analysis (OMA) results.

Functions:
    - :func:`.plot_dtot_hist`: Plot a histogram of the total distance matrix.
    - :func:`.adjust_alpha`: Adjust the alpha (opacity) of a given color.
    - :func:`.rearrange_legend_elements`: Rearrange legend elements into a column-wise ordering.
    - :func:`.freq_vs_damp_plot`: Plot frequency versus damping, with points grouped by clusters.
    - :func:`.stab_clus_plot`: Plots a stabilization chart of the poles of a system with clusters.
    - :func:`.CMIF_plot`: Visualizes the Complex Mode Indicator Function (CMIF).
    - :func:`.EFDD_FIT_plot`: Presents detailed plots for EFDD and FSDD algorithms.
    - :func:`.stab_plot`: Generates stabilization charts.
    - :func:`.cluster_plot`: Visualizes frequency-damping clusters.
    - :func:`.svalH_plot`: Plot the singular values of the Hankel matrix.
    - :func:`.plt_nodes`: Function for plotting 3D geometrical representations.
    - :func:`.plt_lines`: Function for plotting 3D geometrical representations.
    - :func:`.plt_surf`: Function for plotting 3D geometrical representations.
    - :func:`.plt_quiver`: Function for plotting 3D geometrical representations.
    - :func:`.set_ax_options`: Utilities for customizing the appearance of 3D plots.
    - :func:`.set_view`: Utilities for customizing the appearance of 3D plots.
    - :func:`.plt_data`: Plots multi-channel time series data with RMS value inclusion.
    - :func:`.plt_ch_info`: Generates comprehensive channel information plots.
    - :func:`.STFT`: Perform the Short Time Fourier Transform (STFT) to generate spectrograms.
    - :func:`.plot_mac_matrix`: Compute and plot the MAC matrix between the columns of two 2D arrays.
    - :func:`.plot_mode_complexity`: Plot the complexity of a mode shape.

.. automodule:: pyoma2.functions.plot
   :members:
