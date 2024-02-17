Example1 SingleSetup
====================

First of all we import the necessary modules.

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from pyoma2.algorithm import FDD_algo,FSDD_algo,SSIcov_algo
    from pyoma2.OMA import SingleSetup

Then we import the dataset we want to analyse and assign it to a
variable

.. code:: python

    # load example dataset for single setup
    data = np.load("./src/pyoma2/test_data/palisaden/Palisaden_dataset.npy", allow_pickle=True)

Now we can proceed to instantiate the SingleSetup class, passing the
dataset and the sampling frequency as parameters

.. code:: python

    # create single setup
    Pali_ss = SingleSetup(data, fs=100)


If we want to be able to plot the mode shapes, once we have the
results, we need to define the geometry of the structure. We have two
different method available that offers unique plotting capabilities:

- The first method ``def_geo1()`` enables users to visualise mode
  shapes with arrows that represent the placement, direction, and
  magnitude of displacement for each sensor.
- The second method ``def_geo2()`` allows for the plotting and
  animation of mode shapes, with sensors mapped to user defined
  points.

.. code:: python

    # import geometry files
    # Names of the channels
    Names = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"]
    # Common Backgroung nodes and lines
    BG_nodes = np.loadtxt("./src/pyoma2/test_data/palisaden/BG_nodes.txt")
    BG_lines = np.loadtxt("./src/pyoma2/test_data/palisaden/BG_lines.txt").astype(int)
    # Geometry 1
    sens_coord = pd.read_csv("./src/pyoma2/test_data/palisaden/sens_coord.txt", sep="\t")
    sens_dir = np.loadtxt("./src/pyoma2/test_data/palisaden/sens_dir.txt")
    # Geometry 2
    sens_lines = np.loadtxt("./src/pyoma2/test_data/palisaden/sens_lines.txt").astype(int)
    pts_coord = pd.read_csv("./src/pyoma2/test_data/palisaden/pts_coord.txt", sep="\t")
    sens_map = pd.read_csv("./src/pyoma2/test_data/palisaden/sens_map.txt", sep="\t")
    sens_sign = pd.read_csv("./src/pyoma2/test_data/palisaden/sens_sign.txt", sep="\t")

    # Define geometry1
    Pali_ss.def_geo1(
        Names,  # Names of the channels
        sens_coord,  # coordinates of the sensors
        sens_dir,  # sensors' direction
        bg_nodes=BG_nodes,  # BG nodes
        bg_lines=BG_lines,  # BG lines
    )

    # Define geometry 2
    Pali_ss.def_geo2(
        Names,  # Names of the channels
        pts_coord,  # name and coordinates of the points
        sens_map,  # mapping between points and sensors
        order_red="xy",  # argument to reduce the number of DOF per points
        sens_sign=sens_sign,  # sign of the sensor (respect to reference axis)
        sens_lines=sens_lines,  # lines connecting points
        bg_nodes=BG_nodes, # background nodes
        bg_lines=BG_lines, # background lines
    )

Once we have defined the geometry we can show it calling the
``plot_geo1()`` or ``plot_geo2()`` methods.


.. code:: python

    # Plot the geometry
    Pali_ss.plot_geo1(scaleF=2)


.. figure:: /img/Ex1_geo1.png


If we need to detrend and/or decimate the signals we can do this
using the ``detrend_data()*`` and ``decimate_data()`` methods

.. code:: python

    # Detrend and decimate
    #Pali_ss.detrend_data()
    Pali_ss.decimate_data(q=4) # q=decimation factor
    print(Pali_ss.fs)

    >>> 25.0


We can plot all the time histories of the channels calling the
``plot_data()`` method

.. code:: python

    # Plot the Time Histories
    fig, ax = Pali_ss.plot_data()


.. figure:: /img/Ex1_data.png


We can also get more info regarding the quality of the data for each
channel calling the ``plot_ch_info()`` method


.. code:: python

    # Plot TH, PSD and KDE of the (selected) channels
    fig, ax = Pali_ss.plot_ch_info(ch_idx=[-1])


.. figure:: /img/Ex1_ch_info.png


Now we can instantiate the algorithms that we want to run, e.g.
``EFDD_algo`` and ``SSIcov_algo``. The algorithms must then be added
to the setup class using the ``add_algorithms()`` method. Thereafter,
the algorithms can be executed either individually using the
``run_by_name()`` method or collectively with ``run_all()``.


.. code:: python

    # Initialise the algorithms
    fdd = FDD_algo(name="FDD")
    fsdd = FSDD_algo(name="FSDD", nxseg=2048, method_SD="per", pov=0.5)
    ssicov = SSIcov_algo(name="SSIcov", method="cov_mm", br=50, ordmax=80)

    # Overwrite/update run parameters for an algorithm
    fdd.run_params = FDD_algo.RunParamCls(nxseg=512, method_SD="cor")

    # Add algorithms to the single setup class
    Pali_ss.add_algorithms(ssicov, fsdd, fdd)

    # Run all or run by name
    Pali_ss.run_by_name("SSIcov")
    Pali_ss.run_by_name("FSDD")
    # Pali_ss.run_all()

    # save dict of results
    ssi_res = ssicov.result.model_dump()
    fsdd_res = dict(fsdd.result)


We can now plot some of the results:


.. code:: python

    # plot Singular values of PSD
    fsdd.plot_CMIF(freqlim=(0,5))


.. figure:: /img/Ex1_CMIF.png


.. code:: python

    # plot Stabilisation chart for SSI
    ssicov.plot_STDiag(freqlim=(0,5), hide_poles=False)


.. figure:: /img/Ex1_StDiag.png


.. code:: python

    # plot frequecy-damping clusters for SSI
    ssicov.plot_cluster(freqlim=(0,5))


.. figure:: /img/Ex1_clust.png


We are now ready to extract the modal properties of interest either
from the interactive plots using the ``MPE_fromPlot()`` method or
using the ``MPE()`` method.


.. code:: python

    # Select modes to extract from plots
    # Pali_ss.MPE_fromPlot("SSIcov", freqlim=(0,5))

    # or directly
    Pali_ss.MPE("SSIcov", sel_freq=[1.88, 2.42, 2.68], order=40)

    # update dict of results
    ssi_res = dict(ssicov.result)


.. code:: python

    # Select modes to extract from plots
    # Pali_ss.MPE_fromPlot("FSDD", freqlim=(0,5), MAClim=0.95)

    # or directly
    Pali_ss.MPE("FSDD", sel_freq=[1.88, 2.42, 2.68], MAClim=0.95)

    # update dict of results
    fsdd_res = dict(fsdd.result)

We can compare the results from the two methods

.. code:: python

    ssicov.result.Fn

    >>> array([1.88205042, 2.4211625 , 2.68851009])

    fsdd.result.Fn

    >>> array([1.8787832 , 2.42254302, 2.67381079])


We can also plot some additional info regarding the estimates for the
EFDD and FSDD algorithms

.. code:: python

    # plot additional info (goodness of fit) for EFDD or FSDD
    Pali_ss[fsdd.name].plot_FIT(freqlim=(0,5))


.. figure:: /img/Ex1_fit1.png

.. figure:: /img/Ex1_fit2.png

.. figure:: /img/Ex1_fit3.png


And finally we can plot and/or animate the mode shapes extracted from
the analysis

.. code:: python

    # MODE SHAPES PLOT
    # Plot mode 2 (geometry 1)
    Pali_ss[fsdd.name].plot_mode_g1(
        Geo1=Pali_ss.Geo1, mode_numb=2, view="3D", scaleF=2)


.. figure:: /img/Ex1_plot_mode_g1.png


.. code:: python

    # Animate mode 1 (geometry 2)
    Pali_ss[ssicov.name].anim_mode_g2(
        Geo2=Pali_ss.Geo2, mode_numb=1, view="3D", scaleF=3)

.. image:: /img/Ex1_anim_mode_1.gif
