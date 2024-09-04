Example4 - Multisetup with Pre Global Estimation Re-scaling (PreGER) method
===========================================================================

.. code:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   from pyoma2.algorithms import SSIdat_MS
   from pyoma2.setup import MultiSetup_PreGER


For the **PreGER** merging procedure, we adopt a strategy similar to that
used for the single setup class. The first step involves
instantiating the ``MultiSetup_PreGER`` class and passing the list of
datasets, the lists of reference sensors, and their sampling
frequency. Similarly to the single setup class, also for the ``PreGER``
class we can detrend and/or decimate the signals with the
``detrend_data()`` and ``decimate_data()`` methods. Furthermore, we
can plot all the time histories of the channels for the selected
datasets calling the ``plot_data()`` method, and get more info from
each channel from the selected datasets calling the
``plot_ch_info()`` method.

.. code:: python

   # import data files
   set1 = np.load("./src/pyoma2/test_data/3SL/set1.npy", allow_pickle=True)
   set2 = np.load("./src/pyoma2/test_data/3SL/set2.npy", allow_pickle=True)
   set3 = np.load("./src/pyoma2/test_data/3SL/set3.npy", allow_pickle=True)

   # list of datasets and reference indices
   data = [set1, set2, set3]
   ref_ind = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

   # Create multisetup
   msp = MultiSetup_PreGER(fs=100, ref_ind=ref_ind, datasets=data)

   # decimate data
   msp.decimate_data(q=2)

   # plot Time Histories of all channels for the selected datasets
   msp.plot_data(data_idx=[2], nc=2)

   # Plot TH, PSD and KDE of the (selected) channels of the (selected) datasets
   figs, axs = msp.plot_ch_info(data_idx=[1], ch_idx=[2])

.. figure:: /img/Ex4-Fig1.png


Again if we want to be able to plot the mode shapes later, then we
need to define the geometry of the structure.

.. code:: python

   # Geometry 1
   _geo1 = "./src/pyoma2/test_data/3SL/Geo1.xlsx"
   # Geometry 2
   _geo2 = "./src/pyoma2/test_data/3SL/Geo2.xlsx"

   # Define geometry1
   msp.def_geo1_byFILE(_geo1)
   # Define geometry 2
   msp.def_geo2_byFILE(_geo2)


Now we need to instantiate the multi-setup versions of the algorithms
we wish to execute, such as ``SSIdat``.


.. code:: python

   # Initialise the algorithms
   ssidat = SSIdat_MS(name="SSIdat", br=80, ordmax=80)

   # Add algorithms to the class
   msp.add_algorithms(ssidat)
   msp.run_all()

   # Plot
   ssidat.plot_STDiag(freqlim=20)

.. figure:: /img/Ex4-Fig2.png


After the algorithms have been executed we can exctract the desired
poles and plot the mode shapes.

.. code:: python

   # get modal parameters
   msp.MPE(
         "SSIdat",
         sel_freq=[2.63, 2.69, 3.43, 8.29, 8.42, 10.62, 14.00, 14.09, 17.57],
         order=80)

   # plot mode shapes
   ssidat.plot_mode_g1(geo1=msp.geo1, mode_nr=1, view="3D", scaleF=2)
   ssidat.plot_mode_g2(geo2=msp.geo2, mode_nr=6, view="xy", scaleF=2)

.. figure:: /img/Ex4-Fig3.png
.. figure:: /img/Ex4-Fig4.png

.. code:: python

   ssidat.result.Fn

   >>>   array([ 2.63102473,  2.69617968,  3.42605687,  8.27997956,  8.41882261,
               10.59171709, 13.96998337, 14.03397164, 17.49790384])
