import math

from pyoma2.algorithms import SSIdat_MS, pLSCF_MS

from src.pyoma2.setup import MultiSetup_PreGER


def test_geo1(multi_setup_data_fixture, ms_preger: MultiSetup_PreGER) -> None:
    """
    Test the first geometric definition.
    """
    (
        _,
        _,
        _,
        Names,
        BG_nodes,
        BG_lines,
        sens_coord,
        sens_dir,
        _,
        _,
        _,
        _,
    ) = multi_setup_data_fixture

    # Test that the geometric is not defined
    assert ms_preger.geo1 is None

    ms_preger.def_geo1(
        sens_names=Names,
        sens_coord=sens_coord,
        sens_dir=sens_dir,
        bg_nodes=BG_nodes,
        bg_lines=BG_lines,
    )

    assert ms_preger.geo1 is not None


def test_geo2(multi_setup_data_fixture, ms_preger: MultiSetup_PreGER) -> None:
    """
    Test the second geometric definition.
    """
    (
        _,
        _,
        _,
        Names,
        BG_nodes,
        BG_lines,
        _,
        _,
        sens_lines,
        pts_coord,
        sens_map,
        sens_sign,
    ) = multi_setup_data_fixture

    # Test that the geometric is not defined
    assert ms_preger.geo2 is None

    ms_preger.def_geo2(
        sens_names=Names,
        pts_coord=pts_coord,
        sens_map=sens_map,
        sens_sign=sens_sign,
        sens_lines=sens_lines,
        bg_nodes=BG_nodes,
        bg_lines=BG_lines,
    )

    assert ms_preger.geo2 is not None


def test_plot_data(ms_preger: MultiSetup_PreGER) -> None:
    """
    Test the plotting and data manipulation methods of the MultiSetup_PreGER class.
    """
    # test DECIMATE_DATA method not inplace
    assert math.isclose(ms_preger.data[0]["ref"][0][0], -3.249758486587817e-05)
    assert ms_preger.fs == 100.0
    assert ms_preger.dt == 0.01
    decimation_factor = 4
    newdatasets, Y, fs, dt, Ndats, Ts = ms_preger.decimate_data(
        q=decimation_factor, inplace=False
    )
    assert math.isclose(Y[0]["ref"][0][0], -3.27248603574735e-05)
    assert fs == 25.0
    assert dt == 0.01

    # initial class data has not changed
    assert math.isclose(ms_preger.data[0]["ref"][0][0], -3.249758486587817e-05)
    assert ms_preger.fs == 100.0
    assert ms_preger.dt == 0.01

    # test DECIMATE_DATA method inplace
    ms_preger.decimate_data(q=decimation_factor, inplace=True)
    assert math.isclose(ms_preger.data[0]["ref"][0][0], -3.27248603574735e-05)
    assert ms_preger.fs == 25.0
    assert ms_preger.dt == 0.01

    # test DETREND_DATA method not inplace
    new_data = ms_preger.detrend_data(inplace=False)
    assert math.isclose(new_data[0]["ref"][0][0], -3.238227274628828e-05)

    # initial class data has not changed
    assert math.isclose(ms_preger.data[0]["ref"][0][0], -3.27248603574735e-05)

    # test DETREND_DATA method inplace
    ms_preger.detrend_data(inplace=True)
    assert math.isclose(ms_preger.data[0]["ref"][0][0], -3.238227274628828e-05)

    # test FILTER_DATA method not inplace
    new_data = ms_preger.filter_data(Wn=1, order=1, btype="lowpass", inplace=False)
    assert math.isclose(new_data[0]["ref"][0][0], -3.28116336523655e-05)

    # initial class data has not changed
    assert math.isclose(ms_preger.data[0]["ref"][0][0], -3.238227274628828e-05)

    # test FILTER_DATA method inplace
    ms_preger.filter_data(Wn=1, order=1, btype="lowpass", inplace=True)
    assert math.isclose(ms_preger.data[0]["ref"][0][0], -3.28116336523655e-05)

    # test PLOT_DATA method
    try:
        figs, axs = ms_preger.plot_data(data_idx=[0, 1, 2])
        assert isinstance(figs, list)
        assert isinstance(axs, list)
    except Exception as e:
        assert False, f"plot_data raised an exception {e}"

    # test PLOT_CH_INFO method
    try:
        figs, axs = ms_preger.plot_ch_info(data_idx=[0, 1, 2], ch_idx=[-1])
        assert isinstance(figs, list)
        assert isinstance(axs, list)
    except Exception as e:
        assert False, f"plot_ch_info raised an exception {e}"

    # test plot_STFT method
    try:
        figs, axs = ms_preger.plot_STFT(data_idx=[0, 1, 2])
        assert isinstance(figs, list)
        assert isinstance(axs, list)
    except Exception as e:
        assert False, f"plot_STFT raised an exception {e}"


def test_run(multi_setup_data_fixture, ms_preger: MultiSetup_PreGER) -> None:
    """
    Test the running of the algorithms in the MultiSetup_PreGER class.
    """
    (
        _,
        _,
        _,
        Names,
        BG_nodes,
        BG_lines,
        sens_coord,
        sens_dir,
        sens_lines,
        pts_coord,
        sens_map,
        sens_sign,
    ) = multi_setup_data_fixture

    # Define geometry1
    ms_preger.def_geo1(
        Names,  # Names of the channels
        sens_coord,  # coordinates of the sensors
        sens_dir,  # sensors' direction
        bg_nodes=BG_nodes,  # BG nodes
        bg_lines=BG_lines,
    )  # BG lines

    # Define geometry 2
    ms_preger.def_geo2(
        Names,  # Names of the channels
        pts_coord,
        sens_map,
        sens_sign=sens_sign,
        sens_lines=sens_lines,
        bg_nodes=BG_nodes,
        bg_lines=BG_lines,
    )

    # Initialise the algorithms
    ssidat = SSIdat_MS(name="SSIdat", br=5, ordmax=5)
    plscf = pLSCF_MS(name="pLSCF", ordmax=5, nxseg=64)

    ms_preger.decimate_data(q=50, inplace=True)

    # Add algorithms to the class
    ms_preger.add_algorithms(ssidat, plscf)

    # Results are None
    assert ms_preger["SSIdat"].result is None
    assert ms_preger["pLSCF"].result is None

    # Run all algorithms
    ms_preger.run_all()

    # Check the results
    assert ms_preger["SSIdat"].result is not None
    assert ms_preger["pLSCF"].result is not None
