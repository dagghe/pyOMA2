==========================
Extra1 - Tips and Tricks 1
==========================

In this example file we will dive a bit more into pyOMA2 functionalities.
Check out the documentation to get a full description of the various classes, functions, attributes, method and arguments (https://pyoma.readthedocs.io/en/main/).

To start the analysis we need to have the data loaded as an array with shape (Ndat, Nchannels), where Ndat is the number of datapoints (i.e. sampling frequency * period of acquisition), and Nchannels is the number of sensors/channels.

.. code-block:: python

    import os
    import sys
    import numpy as np
    # Add the directory we execute the script from to path:
    sys.path.insert(0, os.path.realpath('__file__'))

    # import the necessary functions and classes
    from pyoma2.functions.gen import example_data
    from pyoma2.setup.single import SingleSetup
    from pyoma2.algorithms.data.run_params import EFDDRunParams, SSIRunParams
    from pyoma2.algorithms.fdd import FSDD
    from pyoma2.algorithms.ssi import SSIcov
    from pyoma2.functions.plot import plot_mac_matrix, plot_mode_complexity
    from pyoma2.functions.gen import save_to_file, load_from_file

    # generate example data and results
    data, ground_truth = example_data()

    # Create an instance of the setup class
    simp_5dof = SingleSetup(data, fs=600)

Once we have created the instance of the setup class we have access to a series of methods (and attributes) that let us inspect the data:
- The ``data`` attribute stores the input data.
- The ``fs`` attribute stores the sampling frequency.
- The ``dt`` attribute stores the sampling interval.
- The ``algorithms`` attribute stores the algorithms associated with the setup.
- The ``plot_STFT()`` method is used to plot the Short-Time Fourier Transform (STFT) magnitude spectrogram for the specified channels, useful to get a graphical feedback regarding the time invariance of the data.
- The ``plot_ch_info()`` method plots information for the specified channels, including time history, normalized auto-correlation, power spectral density (PSD), probability density function, and normal probability plot, useful to get a feedback regarding the quality of the data.
- The ``plot_data()`` method plots the time histories of the data channels in a subplot format.

Moreover we have access to some methods useful to pre-process the data at hand:
- The ``decimate_data()`` method decimates the data.
- The ``detrend_data()`` method detrends the data.
- The ``filter_data()`` method applies a Butterworth filter to the data.
- The ``rollback()`` method restores the data and sampling frequency to their initial state.


.. code-block:: python

    # Decimate the data
    simp_5dof.decimate_data(q=30)

Once we´re done with the pre-processing we can start with the analysis.

The next step is the initialisation of the desired OMA algorithms that will be added to the setup instance.

The parameters required to run each algorithm can be passed one by one to the algorithm instance or in group through the ``run_params`` argument.

.. code-block:: python

    # Import FSDD run parameters (default values) and print out as dictionary
    fsdd_runpar = EFDDRunParams()
    dict(fsdd_runpar)

.. code-block:: python

    # Do the same for SSI, changing some of the default arguments
    ssi_runpar = SSIRunParams(br=50, ordmax=50, calc_unc=True)
    dict(ssi_runpar)

.. code-block:: python

    # Initialise the algorithms
    fsdd = FSDD(name="FSDD", run_params=fsdd_runpar)
    # Equivalent to
    fsdd1 = FSDD(name="FSDD1")

    ssicov = SSIcov(name="SSIcov", run_params=ssi_runpar)
    # Equivalent to
    ssicov1 = SSIcov(name="SSIcov1", br=50, ordmax=50, calc_unc=True)

The ``run_params`` attribute of the algorithm instance let us inspect the parameters passed and overwrite/update them if needed.

.. code-block:: python

    # Inspect the parameters passed
    print("SSI run parameters: ", ssicov.run_params)

    # Overwrite/update run parameters for an algorithm
    fsdd.run_params = FSDD.RunParamCls(nxseg=2048, method_SD="cor", pov=0.5)
    print("")
    print("FSDD run parameters: ", fsdd.run_params)

With the new release we have moved some of the parameters that were actually used for the ``mpe()`` and ``mpe_from_plot()`` methods to a specialised class ``MPEParams``.

Now the algorithms can be added to the setup instance and executed collectively or by name.

.. code-block:: python

    # Add algorithms to the class
    simp_5dof.add_algorithms(fsdd, ssicov)

    # to check which algorithms have been added, we can call the algorithms attribute
    simp_5dof.algorithms

.. code-block:: python

    # run all
    simp_5dof.run_all()
    # or run by name
    # simp_5dof.run_by_name("SSIcov", "FSDD")

Once the algorithms have been run, we gain access to plotting options such as:
- The ``plot_CMIF()`` method for the FDD family of classes, which shows the plot of the singular values of the Spectral Density matrix.
- The ``plot_stab()`` method for the SSI family of classes, which shows the stabilisation of the identified poles for increasing model order.

SSI algorithms have also access to the ``plot_freqvsdamp()`` method which shows the frequency-damping cluster diagram.

.. code-block:: python

    ssicov.plot_freqvsdamp()

The modal results can then be selected "manually" with the ``mpe()`` method or through an interactive version of the ``plot_CMIF()`` and  ``plot_stab()``, using the ``mpe_from_plot()`` method. As mentioned previously the arguments passed to these two methods will be stored in a specialised class accessible through the ``mpe_params`` attribute (after the method has been called)

In order to select a mode press the ``SHIFT`` button and left click on the desired peak/pole, to remove the last selected pole press ``SHIFT`` and right click, finally pressing ``SHIFT`` and the middle button will remove the closest selected peack/pole.

.. code-block:: python

    # get the modal parameters with the interactive plot
    simp_5dof.mpe_from_plot("FSDD", freqlim=(0, 8))
    simp_5dof.mpe_from_plot("SSIcov", freqlim=(0, 8))

Once the algorithms have been run and the modes extracted, we can access the results. We can inspect the whole dictionary of results at once, or access the single results one by one.

.. code-block:: python

    # check the mpe_params
    fsdd.mpe_params

.. code-block:: python

    # dict of results
    fsdd_res = dict(fsdd.result)
    fsdd.result.Fn
    # fsdd_res["Fn"]

.. code-block:: python

    ssicov_res = dict(ssicov.result)
    ssicov.result.Fn_std
    # ssicov_res["Fn_std"]

For the EFDD and FSDD algorithms the ``plot_EFDDfit()`` method generates a plot helping to visualise the quality and accuracy of modal identification.

There are also some useful functions in the ``plot`` module and in the ``gen`` module that can be used for further inspection of the results and saving/loading purposes:
- The ``plot_mac_matrix()`` function can be used to plot the MAC matrix, useful to compare different set of results.
- The ``plot_mode_complexity()`` function can be used to plot how "complex" a mode is.
- The ``load_from_file()`` function can be used to load a setup instance from a file.
- The ``save_to_file()`` function can be used to save a setup instance to a file.

.. code-block:: python

    # plot the mac matrix between the SSI and FSDD results
    plot_mac_matrix(ssicov_res['Phi'].real, fsdd_res['Phi'].real)
    # equivalent to
    # plot_mac_matrix(ssi.result.Phi.real, fsdd.result.Phi.real)

.. code-block:: python

    plot_mode_complexity(ssicov_res['Phi'][2])
