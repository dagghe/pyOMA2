Example2 - Real dataset
=======================

In this second example we will explore more functionalities with a real dataset [APTF20]_
First of all we import the necessary modules. Then we import the dataset we want to analyse and assign it to a
variable

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pyoma2.algorithms import pLSCF,FSDD,SSIcov
    from pyoma2.setup import SingleSetup

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

    _geo1 =  r"../src/pyoma2/test_data/palisaden/Geo1.xlsx"
    _geo2 =  r"../src/pyoma2/test_data/palisaden/Geo2.xlsx"

    Pali_ss.def_geo1_byFILE(_geo1)
    Pali_ss.def_geo2_byFILE(_geo2)


Once we have defined the geometry we can show it calling the
``plot_geo1()`` or ``plot_geo2()`` methods.

.. code:: python

    # Plot the geometry (geometry1)
    fig, ax = Pali_ss.plot_geo1(scaleF=2)
    # (geometry2) with pyvista
    _ = Pali_ss.plot_geo2(scaleF=2, notebook=True)
    # (geometry2) with matplotlib
    _, _ = Pali_ss.plot_geo2_mpl(scaleF=2)


.. figure:: /img/Ex2-Fig1.png
.. figure:: /img/Ex2-Fig2.png
.. figure:: /img/Ex2-Fig3.png


We can plot all the time histories of the channels calling the
``plot_data()`` method

.. code:: python

    # Plot the Time Histories
    fig, ax = Pali_ss.plot_data()


.. figure:: /img/Ex2-Fig4.png


We can also get more info regarding the quality of the data for each
channel calling the ``plot_ch_info()`` method


.. code:: python

    # Plot TH, PSD and KDE of the (selected) channels
    fig, ax = Pali_ss.plot_ch_info(ch_idx=[-1])


.. figure:: /img/Ex2-Fig5.png


As we can see from the auto correlation there's a low frequency component in the data.
Other than the ``detrend_data()`` and ``decimate_data()`` methods there's also a ``filter_data()`` method that can help us here.

.. code:: python

    # Detrend and decimate
    #Pali_ss.detrend_data()
    Pali_ss.filter_data(Wn=(0.1), order=8, btype="highpass")
    Pali_ss.decimate_data(q=5)
    _, _ = Pali_ss.plot_ch_info(ch_idx=[-1])


.. figure:: /img/Ex2-Fig6.png


Now we can instantiate the algorithms that we want to run, e.g.
``EFDD`` and ``SSIcov``. The algorithms must then be added
to the setup class using the ``add_algorithms()`` method. Thereafter,
the algorithms can be executed either individually using the
``run_by_name()`` method or collectively with ``run_all()``.


.. code:: python

    # Initialise the algorithms
    fsdd = FSDD(name="FSDD", nxseg=1024, method_SD="cor")
    ssicov = SSIcov(name="SSIcov", br=50, ordmax=50)
    plscf = pLSCF(name="polymax",ordmax=30)

    # Overwrite/update run parameters for an algorithm
    fsdd.run_params = FSDD.RunParamCls(nxseg=2048, method_SD="per", pov=0.5)

    # Add algorithms to the single setup class
    Pali_ss.add_algorithms(ssicov, fsdd, plscf)

    # Run all or run by name
    Pali_ss.run_by_name("SSIcov")
    Pali_ss.run_by_name("FSDD")
    Pali_ss.run_by_name("polymax")
    # Pali_ss.run_all()

    # save dict of results
    ssi_res = ssicov.result.model_dump()
    fsdd_res = dict(fsdd.result)


We can now plot some of the results:


.. code:: python

    # plot Singular values of PSD
    fsdd.plot_CMIF(freqlim=(0,5))


.. figure:: /img/Ex2-Fig7.png


.. code:: python

    # plot Stabilisation chart for SSI
    ssicov.plot_stab(freqlim=(0,5), hide_poles=False)


.. figure:: /img/Ex2-Fig8.png


.. code:: python

    # plot frequecy-damping clusters for SSI
    ssicov.plot_cluster(freqlim=(0,5))


.. figure:: /img/Ex2-Fig9.png


.. code:: python

    # plot Stabilisation chart for pLSCF
    _, _ = plscf.plot_stab(freqlim=(1,4), hide_poles=False)


.. figure:: /img/Ex2-Fig10.png


We are now ready to extract the modal properties of interest either
from the interactive plots using the ``mpe_from_plot()`` method or
using the ``MPE()`` method.


.. code:: python

    # Select modes to extract from plots
    # Pali_ss.mpe_from_plot("SSIcov", freqlim=(0,5))

    # or directly
    Pali_ss.MPE("SSIcov", sel_freq=[1.88, 2.42, 2.68], order=40)

    # update dict of results
    ssi_res = dict(ssicov.result)


.. code:: python

    # Select modes to extract from plots
    # Pali_ss.mpe_from_plot("FSDD", freqlim=(0,5), MAClim=0.95)

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
    Pali_ss[fsdd.name].plot_EFDDfit(freqlim=(0,5))


.. figure:: /img/Ex2-Fig11.png

.. figure:: /img/Ex2-Fig12.png

.. figure:: /img/Ex2-Fig13.png


And finally we can plot and/or animate the mode shapes extracted from
the analysis

.. code:: python

    # MODE SHAPES PLOT
    # Plot mode 2 (geometry 1)
    Pali_ss[fsdd.name].plot_mode_g1(
        geo1=Pali_ss.geo1, mode_nr=2, view="3D", scaleF=2)


.. figure:: /img/Ex2-Fig14.png


.. code:: python

    # Animate mode 1 (geometry 2)
    Pali_ss[ssicov.name].anim_mode_g2(
        geo2=Pali_ss.geo2, mode_nr=1, scaleF=3)

.. image:: /img/Ex2-Fig15.gif

It is also possible to save and load the results to a pickled file.

.. code:: python

    from pyoma2.functions.gen import save_to_file, load_from_file

    # Save setup
    save_to_file(Pali_ss, "<Path to your file>/name.pkl")

    # Load setup
    pali2 = load_from_file("<Path to your file>/name.pkl"")

    # plot from loded instance
    pali2[fsdd.name].plot_mode_g2_mpl(
        geo2=pali2.geo2, mode_nr=2, view="3D", scaleF=2)

.. figure:: /img/Ex2-Fig16.png

.. [APTF20] Aloisio, A., Pasca, D., Tomasi, R., & Fragiacomo, M. (2020). Dynamic identification and model updating of an eight-storey CLT building. Engineering Structures, 213, 110593.
