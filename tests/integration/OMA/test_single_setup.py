import math
import typing

import numpy as np
import pandas as pd
import pytest
from scipy.signal import decimate, detrend

from src.pyoma2.algorithm import FDD_algo, FSDD_algo, SSIcov_algo
from src.pyoma2.OMA import BaseSetup, SingleSetup
from tests.factory import FakeAlgorithm, FakeAlgorithm2


def test_base_setup(single_setup_data_fixture, bs: BaseSetup) -> None:
    """Test BaseSetup utility functions."""
    (
        data,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = single_setup_data_fixture
    alg1 = FakeAlgorithm(name="fake_1")
    alg2 = FakeAlgorithm2(name="fake_2")

    # Test the INITIALIZATION of the FakeAlgorithm and FakeAlgorithm2
    assert getattr(alg1, "data", None) is None
    assert getattr(alg1, "fs", None) is None
    assert alg1.name == "fake_1"

    assert getattr(alg2, "data", None) is None
    assert getattr(alg2, "fs", None) is None
    assert alg2.name == "fake_2"

    # Test the INITIALIZATION of the BaseSetup
    assert bs.data is not None
    assert bs.fs == 100
    assert getattr(bs, "algorithms", None) is None

    # Test the ADD_ALGORITHMS method
    bs.add_algorithms(alg1, alg2)
    assert bs.algorithms["fake_1"].name == "fake_1"
    assert bs.algorithms["fake_2"].name == "fake_2"
    assert bs["fake_1"].name == "fake_1"
    assert bs["fake_2"].name == "fake_2"

    # Test the GET ALGORITHM method
    assert bs.get("fake_1").name == "fake_1"
    assert bs.get("fake_2").name == "fake_2"

    # Test the GET ALGORITHM method with an unknown algorithm
    with pytest.raises(KeyError):
        bs.algorithms["unknown"]
    with pytest.raises(KeyError):
        bs["unknown"]
    assert bs.get("unknown") is None

    # Test DECIMATE_DATA method
    data = np.array(np.arange(0, 30))
    fs = 100
    q = 2
    newdata, fs, dt, Ndat, T = BaseSetup._decimate_data(data=data, fs=fs, q=q, axis=0)
    assert np.array_equal(
        newdata, decimate(data, q)
    )  # Use scipy's decimate function for expected result
    assert fs == 50
    assert dt == 0.02
    assert Ndat == len(newdata)
    assert T == 0.15

    # Test DETREND_DATA method
    detrended_data = BaseSetup._detrend_data(data)
    assert np.array_equal(
        detrended_data, detrend(data)
    )  # Use scipy's detrend function for expected result

    # Test the FILTER_DATA method
    filtered_data = BaseSetup._filter_data(
        data=np.array(np.arange(0, 7)), fs=fs, Wn=1, order=1, btype="lowpass"
    )
    assert np.allclose(
        filtered_data,
        np.array(
            [
                0.20431932,
                0.76033598,
                1.31254294,
                1.85382214,
                2.37688175,
                2.87414037,
                3.33760645,
            ]
        ),
    )  # Use scipy's lfilter function for expected result


def test_geo1(single_setup_data_fixture, ss: SingleSetup) -> None:
    """
    Test the first geometry definition and plotting of the SingleSetup.
    """
    _, Names, Bg_nodes, Bg_lines, sens_coord, sens_dir, *_ = single_setup_data_fixture

    # Test that the geometry is not defined
    assert ss.Geo1 is None

    # plot without defining the geometry
    with pytest.raises(ValueError) as e:
        ss.plot_geo1()
    assert "Geo1 is not defined. cannot plot geometry" in str(e.value)

    assert np.array_equal(
        Bg_lines,
        np.array([[1, 5], [2, 6], [3, 7], [4, 8], [5, 6], [6, 7], [7, 8], [8, 5]]),
    )

    assert sens_coord.equals(
        pd.DataFrame(
            {
                "sName": ["ch5", "ch6", "ch1", "ch2", "ch3", "ch4"],
                "x": [5, 5, 2, 2, 11, 11],
                "y": [2, 2, 8, 8, 8, 8],
                "z": [20, 20, 20, 20, 20, 20],
            }
        )
    )

    # DEFINE THE GEOMETRY
    ss.def_geo1(
        sens_names=Names,
        sens_coord=sens_coord,
        sens_dir=sens_dir,
        bg_nodes=Bg_nodes,
        bg_lines=Bg_lines,
    )

    # Test the initialization of the Geometry
    assert ss.Geo1 is not None
    assert ss.Geo1.sens_names == Names
    # bg_lines are different because the first column is 0-indexed
    assert np.array_equal(
        ss.Geo1.bg_lines,
        np.array([[0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]]),
    )
    # sens_cord was reindexed
    assert ss.Geo1.sens_coord.equals(
        pd.DataFrame(
            {
                "sName": ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"],
                "x": [2, 2, 11, 11, 5, 5],
                "y": [8, 8, 8, 8, 2, 2],
                "z": [20, 20, 20, 20, 20, 20],
            },
            index=[2, 3, 4, 5, 0, 1],
        )
    )
    assert np.array_equal(
        ss.Geo1.sens_dir,
        np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )
    assert np.array_equal(ss.Geo1.bg_nodes, Bg_nodes)
    assert ss.Geo1.bg_surf is None

    # PLOT THE GEOMETRY
    # Call the plot_geo1 method and check that it doesn't raise an exception
    try:
        fig, ax = ss.plot_geo1()
    except Exception as e:
        assert False, f"plot_geo1 raised an exception {e}"

    # PLOT GEOMETRY WITH bg_surf
    # Define the bg_surf
    bg_surf = np.array([[0, 1, 2], [2, 3, 0]])
    ss.Geo1.bg_surf = bg_surf

    try:
        fig, ax = ss.plot_geo1()
    except Exception as e:
        assert False, f"plot_geo1 with bg_surf raised an exception: {e}"

    # PLOT GEOMETRY WITH sens_lines
    # Define the sens_lines
    sens_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    ss.Geo1.sens_lines = sens_lines

    try:
        fig, ax = ss.plot_geo1()
    except Exception as e:
        assert False, f"plot_geo1 with sens_lines raised an exception: {e}"

    # PLOT_MODE_GEO1
    try:
        f_al = FakeAlgorithm(name="fake1", run_params=FakeAlgorithm.RunParamCls())
        ss.add_algorithms(f_al)
        ss.run_all()
        fig, ax = ss["fake1"].plot_mode_g1(Geo1=ss.Geo1, mode_numb=2, view="3D", scaleF=2)
    except Exception as e:
        assert False, f"plot_mode_geo1 raised an exception {e}"


# @pytest.mark.xfail(reason="Issue #8")
@pytest.mark.parametrize(
    "input_order_red, input_sens_map, input_sens_sign",
    (
        (
            None,
            pd.DataFrame(
                {
                    "ptName": [1, 2, 3, 4, 5, 6],
                    "x": ["ch1", "ch3", "ch5", "ch1", "ch2", "ch4"],
                    "y": ["ch2", "ch4", "ch6", "ch3", "ch5", "ch6"],
                    "z": [0, 0, 0, 0, 0, 0],
                }
            ),
            pd.DataFrame(
                {
                    "ptName": [1, 2, 3, 4, 5, 6],
                    "x": [-1, -1, -1, 0, 0, 0],
                    "y": [-1, -1, 1, 0, 0, 0],
                    "z": [0, 0, 0, 0, 0, 0],
                }
            ),
        ),
        ("xy", None, None),  # use default sens map and sens sign
        pytest.param(
            "xz", None, None, marks=pytest.mark.xfail
        ),  # use default sens map and sens sign
        ("yz", None, None),  # use default sens map and sens sign
        (
            "x",
            pd.DataFrame(
                {
                    "ptName": [1, 2, 3, 4, 5, 6],
                    "x": ["ch1", "ch3", "ch5", "ch1", "ch2", "ch4"],
                }
            ),
            pd.DataFrame(
                {
                    "ptName": [1, 2, 3, 4, 5, 6],
                    "x": [-1, -1, -1, 0, 0, 0],
                }
            ),
        ),
        pytest.param(
            "y",
            pd.DataFrame(
                {
                    "ptName": [1, 2, 3, 4, 5, 6],
                    "x": ["ch1", "ch3", "ch5", "ch1", "ch2", "ch4"],
                }
            ),
            pd.DataFrame(
                {
                    "ptName": [1, 2, 3, 4, 5, 6],
                    "x": [-1, -1, -1, 0, 0, 0],
                }
            ),
            marks=pytest.mark.xfail,
        ),
        (
            "z",
            pd.DataFrame(
                {
                    "ptName": [1, 2, 3, 4, 5, 6],
                    "x": ["ch1", "ch3", "ch5", "ch1", "ch2", "ch4"],
                }
            ),
            pd.DataFrame(
                {
                    "ptName": [1, 2, 3, 4, 5, 6],
                    "x": [-1, -1, -1, 0, 0, 0],
                }
            ),
        ),
    ),
)
def test_geo2(
    single_setup_data_fixture,
    ss: SingleSetup,
    input_order_red: typing.Optional[str],
    input_sens_map: typing.Optional[pd.DataFrame],
    input_sens_sign: typing.Optional[pd.DataFrame],
) -> None:
    """
    Test the second geometry definition and plotting of the SingleSetup.
    """
    _, Names, Bg_nodes, Bg_lines, _, _, sens_lines, pts_coord, sens_map, sens_sign = (
        single_setup_data_fixture
    )

    if input_sens_map is not None:
        sens_map = input_sens_map
    if input_sens_sign is not None:
        sens_sign = input_sens_sign

    # Test the initialization of the SingleSetup
    assert ss.Geo2 is None

    # plot without defining the geometry
    with pytest.raises(ValueError) as e:
        ss.plot_geo2()
    assert "Geo2 is not defined. Cannot plot geometry" in str(e.value)

    # DEFINE THE GEOMETRY
    ss.def_geo2(
        sens_names=Names,
        pts_coord=pts_coord,
        sens_map=sens_map,
        order_red=input_order_red,
        sens_lines=sens_lines,
        sens_sign=sens_sign,
        bg_nodes=Bg_nodes,
        bg_lines=Bg_lines,
    )

    # Test the initialization of the Geometry
    assert ss.Geo2 is not None
    assert np.array_equal(
        ss.Geo2.bg_lines,
        np.array([[0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]]),
    )
    assert np.array_equal(
        ss.Geo2.sens_lines, np.array([[3, 0], [4, 1], [5, 2], [0, 1], [1, 2], [2, 0]])
    )
    assert ss.Geo2.pts_coord.equals(pts_coord)

    # PLOT THE GEOMETRY
    # Call the plot_geo2 method and check that it doesn't raise an exception
    try:
        fig, ax = ss.plot_geo2()
    except Exception as e:
        assert False, f"plot_geo2 raised an exception {e}"

    # Check the number of lines in the plot
    expected_number_of_lines = 0
    assert len(ax.lines) == expected_number_of_lines

    # PLOT GEOMETRY WITH bg_surf
    # Define the bg_surf
    bg_surf = np.array([[0, 1, 2], [2, 3, 0]])
    ss.Geo2.bg_surf = bg_surf

    try:
        fig, ax = ss.plot_geo2()
    except Exception as e:
        assert False, f"plot_geo2 with bg_surf raised an exception: {e}"


def test_plot_data(
    ss: SingleSetup,
) -> None:
    """
    Test the plotting and data manipulation methods of the SingleSetup.
    """
    # test DECIMATE_DATA method not inplace
    initial_shape = ss.data.shape
    decimation_factor = 4
    new_data, *_ = ss.decimate_data(q=decimation_factor, inplace=False)
    assert new_data.shape == (initial_shape[0] // decimation_factor, initial_shape[1])
    assert ss.data.shape == initial_shape

    # test DECIMATE_DATA method inplace
    initial_shape = ss.data.shape
    decimation_factor = 4
    ss.decimate_data(q=decimation_factor, inplace=True)
    assert ss.data.shape == (initial_shape[0] // decimation_factor, initial_shape[1])

    # test DETREND_DATA method not inplace
    initial_shape = ss.data.shape
    assert math.isclose(ss.data[0][0], 0.002033720017696059)
    new_data = ss.detrend_data(inplace=False)
    assert math.isclose(new_data[0][0], 0.002687724196559849)
    assert new_data.shape == initial_shape

    # test DETREND_DATA method inplace
    initial_shape = ss.data.shape
    assert math.isclose(ss.data[0][0], 0.002033720017696059)
    ss.detrend_data(inplace=True)
    assert math.isclose(ss.data[0][0], 0.002687724196559849)
    assert ss.data.shape == initial_shape

    # test FILTER_DATA method not inplace
    initial_shape = ss.data.shape
    assert math.isclose(ss.data[0][0], 0.002687724196559849)
    new_data = ss.filter_data(Wn=1, order=1, btype="lowpass", inplace=False)
    assert math.isclose(new_data[0][0], 0.002649116058096131)
    assert new_data.shape == initial_shape

    # test FILTER_DATA method inplace
    initial_shape = ss.data.shape
    assert math.isclose(ss.data[0][0], 0.002687724196559849)
    ss.filter_data(Wn=1, order=1, btype="lowpass", inplace=True)
    assert math.isclose(ss.data[0][0], 0.002649116058096131)
    assert ss.data.shape == initial_shape

    # test PLOT_DATA method
    try:
        fig, ax = ss.plot_data()
    except Exception as e:
        assert False, f"plot_data raised an exception {e}"

    # test PLOT_CH_INFO method
    try:
        fig, ax = ss.plot_ch_info(ch_idx=[-1])
        assert isinstance(ax, list)
    except Exception as e:
        assert False, f"plot_ch_info raised an exception {e}"


def test_run(single_setup_data_fixture, ss: SingleSetup) -> None:
    """
    Test the running of the algorithms in the SingleSetup.
    """
    # Initialize the geometries
    (
        _,
        Names,
        Bg_nodes,
        Bg_lines,
        sens_coord,
        sens_dir,
        sens_lines,
        pts_coord,
        sens_map,
        sens_sign,
    ) = single_setup_data_fixture

    # Define geometry1
    ss.def_geo1(
        Names,  # Names of the channels
        sens_coord,  # coordinates of the sensors
        sens_dir,  # sensors' direction
        bg_nodes=Bg_nodes,  # BG nodes
        bg_lines=Bg_lines,  # BG lines
    )

    # Define geometry 2
    ss.def_geo2(
        Names,  # Names of the channels
        pts_coord,  # name and coordinates of the points
        sens_map,  # mapping between points and sensors
        order_red="xy",  # argument to reduce the number of DOF per points
        sens_sign=sens_sign,  # sign of the sensor (respect to reference axis)
        sens_lines=sens_lines,  # lines connecting points
        bg_nodes=Bg_nodes,  # background nodes
        bg_lines=Bg_lines,  # background lines
    )

    # Initialise the algorithms
    fdd = FDD_algo(name="FDD")
    fsdd = FSDD_algo(name="FSDD", nxseg=2048, method_SD="per", pov=0.5)
    ssicov = SSIcov_algo(name="SSIcov", br=50, ordmax=80)

    # Overwrite/update run parameters for an algorithm
    fdd.run_params = FDD_algo.RunParamCls(nxseg=512, method_SD="cor")
    # Aggiungere esempio anche col metodo

    # Add algorithms to the single setup class
    ss.add_algorithms(ssicov, fsdd, fdd)

    # results are none
    assert ss["FDD"].result is None
    assert ss["FSDD"].result is None
    assert ss["SSIcov"].result is None

    # Run all or run by name
    ss.run_by_name("SSIcov")
    ss.run_by_name("FSDD")

    # Run all algorithms
    ss.run_all()

    # Check the result
    assert ss["FDD"].result is not None
    assert ss["FSDD"].result is not None
    assert ss["SSIcov"].result is not None

    # plot SINGULAR VALUES
    try:
        fig, ax = fsdd.plot_CMIF(freqlim=(1, 4))
    except Exception as e:
        assert False, f"plot_CMIF raised an exception {e}"

    # plot STABILISATION CHART for SSI
    try:
        fig4, ax4 = ssicov.plot_STDiag(freqlim=(1, 4), hide_poles=False)
    except Exception as e:
        assert False, f"plot_STDiag raised an exception {e}"

    # plot FREQUECY-DAMPING CLUSTERS for SSI
    try:
        fig4, ax4 = ssicov.plot_cluster(freqlim=(1, 4))
    except Exception as e:
        assert False, f"plot_cluster raised an exception {e}"

    # run MPE_FROMPLOT for algorithms
    try:
        ss.MPE_fromPlot("SSIcov", freqlim=(1, 4))
    except Exception as e:
        assert False, f"MPE_fromPlot raised an exception {e} for SSIcov"

    try:
        ss.MPE_fromPlot("FSDD", freqlim=(1, 4))
    except Exception as e:
        assert False, f"MPE_fromPlot raised an exception {e} for FSDD"

    try:
        ss.MPE_fromPlot("FDD", freqlim=(1, 4))
    except Exception as e:
        assert False, f"MPE_fromPlot raised an exception {e} for FDD"

    # run MPE for algorithms
    try:
        ss.MPE("SSIcov", sel_freq=[1.88, 2.42, 2.68], order=40)
    except Exception as e:
        assert False, f"MPE raised an exception {e} for SSIcov"

    try:
        ss.MPE("FSDD", sel_freq=[1.88, 2.42, 2.68], MAClim=0.95)
    except Exception as e:
        assert False, f"MPE raised an exception {e} for FSDD"

    try:
        ss.MPE("FDD", sel_freq=[1.88, 2.42, 2.68])
    except Exception as e:
        assert False, f"MPE raised an exception {e} for FDD"

    # plot_FIT for FSDD algorithms
    try:
        figs, axs = ss["FSDD"].plot_FIT(freqlim=(1, 4))
        assert isinstance(figs, list)
        assert isinstance(axs, list)
    except Exception as e:
        assert False, f"plot_fit raised an exception {e} for FDD"

    # PLOTE_MODE_G1
    try:
        fig, ax = ss["FDD"].plot_mode_g1(Geo1=ss.Geo1, mode_numb=2, view="3D", scaleF=2)
    except Exception as e:
        assert False, f"plot_mode_g1 raised an exception {e} for FDD"

    # PLOTE_MODE_G2
    try:
        fig, ax = ss["FSDD"].plot_mode_g2(Geo2=ss.Geo2, mode_numb=2, view="3D", scaleF=2)
    except Exception as e:
        assert False, f"plot_mode_g2 raised an exception {e} for FSDD"
