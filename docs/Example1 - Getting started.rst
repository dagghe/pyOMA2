Example1 - Getting started
==========================

In this first example we'll take a look at a simple 5 degrees of freedom (DOF) system.
To access the data and the exact results of the system we can call the ``example_data()`` function in the submodule ``functions.gen``

.. code:: python

    import os
    import sys
    import numpy as np
    # Add the directory we executed the script from to path:
    sys.path.insert(0, os.path.realpath('__file__'))

    # import the function to generate the example dataset
    from pyoma2.functions.gen import example_data

    # assign the returned values
    data, ground_truth = example_data()

    # Print the exact results
    np.set_printoptions(precision=3)
    print(f"the natural frequencies are: {ground_truth[0]} \n")
    print(f"the damping is: {ground_truth[2]} \n")
    print("the (column-wise) mode shape matrix: \n"
    f"{ground_truth[1]} \n")


Now we can instantiate the SingleSetup class, passing the dataset and the sampling frequency as arguments

.. code:: python

    from pyoma2.setup.single import SingleSetup

    simp_5dof = SingleSetup(data, fs=200)


Since the maximum frequency is at approximately 6Hz, we can decimate the signal quite a bit.
To do this we can call the ``decimate_data()`` method

.. code:: python

    # Decimate the data by factor 10
    simp_5dof.decimate_data(q=10)


To analise the data we need to instanciate the desired algorithm to use with a name and the required arguments.

.. code:: python

    from pyoma2.algorithms.fdd import FDD
    from pyoma2.algorithms.ssi import SSIdat

    # Initialise the algorithms
    fdd = FDD(name="FDD", nxseg=1024, method_SD="cor")
    ssidat = SSIdat(name="SSIdat", br=30, ordmax=30)

    # Add algorithms to the class
    simp_5dof.add_algorithms(fdd, ssidat)

    # run
    simp_5dof.run_all()


We can now check the results

.. code:: python


    # plot singular values of the spectral density matrix
    _, _ = fdd.plot_CMIF(freqlim=(0,8))

    # plot the stabilisation diagram
    _, _ = ssidat.plot_stab(freqlim=(0,10),hide_poles=False)

.. image:: /img/Ex1-Fig1.png
.. image:: /img/Ex1-Fig2.png

We can get the modal parameters with the help of an interactive plot calling the ``mpe_from_plot()`` method,
or we can get the results "manually" with the ``MPE()`` method.

.. code:: python

    # get the modal parameters with the interactive plot
    # simp_ex.mpe_from_plot("SSIdat", freqlim=(0,10))

    # or manually
    simp_5dof.MPE("SSIdat", sel_freq=[0.89, 2.598, 4.095, 5.261, 6.], order="find_min")


Now we can now access all the results and compare them to the exact solution

.. code:: python

    # dict of results
    ssidat_res = dict(ssidat.result)

    from pyoma2.functions.plot import plot_mac_matrix

    # print the results
    print(f"order out: {ssidat_res['order_out']} \n")
    print(f"the natural frequencies are: {ssidat_res['Fn']} \n")
    print(f"the dampings are: {ssidat_res['Xi']} \n")
    print("the (column-wise) mode shape matrix:")
    print(f"{ssidat_res['Phi'].real} \n")

    _, _ = plot_mac_matrix(ssidat_res['Phi'].real, ground_truth[1])

.. image:: /img/Ex1-Fig3.png
