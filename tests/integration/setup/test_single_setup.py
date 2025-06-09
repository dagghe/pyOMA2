import math
import typing

import numpy as np
import pandas as pd
import pytest
from scipy.signal import decimate, detrend

from src.pyoma2.algorithms import FDD, FSDD, SSI
from src.pyoma2.setup import BaseSetup, SingleSetup
from tests.factory import FakeAlgorithm, FakeAlgorithm2


def test_base_setup(single_setup_data_fixture, bs: BaseSetup) -> None:
    """Test BaseSetup utility functions."""
    (
        data,
        *_,
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
    _, Names = single_setup_data_fixture

    # Test that the geometry is not defined
    assert ss.geo1 is None

    # plot without defining the geometry
    with pytest.raises(ValueError) as e:
        ss.plot_geo1()
    assert "geo1 is not defined. Call def_geo1 first." in str(e.value)

    # DEFINE THE GEOMETRY
    ss.def_geo1_by_file(
        path="./tests/test_data/palisaden/Geo1.xlsx",
    )

    # Test the initialization of the Geometry
    assert ss.geo1 is not None
    assert ss.geo1.sens_names == Names
    # bg_lines are different because the first column is 0-indexed
    assert np.array_equal(
        ss.geo1.bg_lines,
        np.array([[3, 0], [4, 1], [5, 2], [0, 1], [1, 2], [2, 0]]),
    )
    # sens_cord was reindexed
    assert ss.geo1.sens_coord.equals(
        pd.DataFrame(
            {
                "label": ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"],
                "x": [2, 11, 5, 2, 11, 5],
                "y": [8, 8, 2, 8, 8, 2],
                "z": [20, 20, 20, 20, 20, 20],
            },
        ).set_index("label")
    )
    assert np.array_equal(
        ss.geo1.sens_dir,
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
    assert np.array_equal(
        ss.geo1.bg_nodes,
        np.array(
            [[2, 8, 20], [11, 8, 20], [5, 2, 20], [2, 8, 0], [11, 8, 0], [5, 2, 0]],
            dtype=int,
        ),
    )
    assert ss.geo1.bg_surf is None

    # PLOT THE GEOMETRY
    # Call the plot_geo1 method and check that it doesn't raise an exception
    try:
        fig, ax = ss.plot_geo1()
    except Exception as e:
        assert False, f"plot_geo1 raised an exception {e}"

    # PLOT GEOMETRY WITH bg_surf
    # Define the bg_surf
    bg_surf = np.array([[0, 1, 2], [2, 3, 0]])
    ss.geo1.bg_surf = bg_surf

    try:
        fig, ax = ss.plot_geo1()
    except Exception as e:
        assert False, f"plot_geo1 with bg_surf raised an exception: {e}"

    # PLOT GEOMETRY WITH sens_lines
    # Define the sens_lines
    sens_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    ss.geo1.sens_lines = sens_lines

    try:
        fig, ax = ss.plot_geo1()
    except Exception as e:
        assert False, f"plot_geo1 with sens_lines raised an exception: {e}"

    # PLOT_MODE_GEO1
    try:
        f_al = FakeAlgorithm(name="fake1", run_params=FakeAlgorithm.RunParamCls())
        ss.add_algorithms(f_al)
        ss.run_all()
        fig, ax = ss.plot_mode_geo1(algo_res=f_al.result, mode_nr=2, view="3D", scaleF=2)
    except Exception as e:
        assert False, f"plot_mode_geo1 raised an exception {e}"


@pytest.mark.parametrize(
    "input_sens_map, input_sens_sign",
    (
        (
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
        (None, None),  # use default sens map and sens sign
        (None, None),  # use default sens map and sens sign
        (None, None),  # use default sens map and sens sign
        (
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
        (
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
        (
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
    ss: SingleSetup,
    input_sens_map: typing.Optional[pd.DataFrame],
    input_sens_sign: typing.Optional[pd.DataFrame],
) -> None:
    """
    Test the second geometry definition and plotting of the SingleSetup.
    """
    if input_sens_map is not None:
        _ = input_sens_map
    if input_sens_sign is not None:
        _ = input_sens_sign

    # Test the initialization of the SingleSetup
    assert ss.geo2 is None

    # plot without defining the geometry
    with pytest.raises(ValueError) as e:
        ss.plot_geo2()
    assert "geo2 is not defined. Call def_geo2 first." in str(e.value)

    # DEFINE THE GEOMETRY
    ss.def_geo2_by_file(path="./tests/test_data/palisaden/Geo2.xlsx")

    # Test the initialization of the Geometry
    assert ss.geo2 is not None
    assert np.array_equal(
        ss.geo2.bg_lines,
        np.array([[3, 0], [4, 1], [5, 2], [0, 1], [1, 2], [2, 0]]),
    )
    assert np.array_equal(
        ss.geo2.sens_lines, np.array([[3, 0], [4, 1], [5, 2], [0, 1], [1, 2], [2, 0]])
    )
    assert ss.geo2.pts_coord.equals(
        pd.DataFrame(
            {
                "ptName": [1, 2, 3, 4, 5, 6],
                "x": [2.0, 11.0, 5.0, 2.0, 11.0, 5.0],
                "y": [8.0, 8.0, 2.0, 8.0, 8.0, 2.0],
                "z": [20.0, 20.0, 20.0, 0.0, 0.0, 0.0],
            }
        ).set_index("ptName")
    )

    # PLOT THE GEOMETRY
    # Call the plot_geo2 method and check that it doesn't raise an exception
    try:
        _ = ss.plot_geo2()
    except Exception as e:
        assert False, f"plot_geo2 raised an exception {e}"

    # PLOT GEOMETRY WITH bg_surf
    # Define the bg_surf
    bg_surf = np.array([[0, 1, 2], [2, 3, 0]])
    ss.geo2.bg_surf = bg_surf

    try:
        _ = ss.plot_geo2()
    except Exception as e:
        assert False, f"plot_geo2 with bg_surf raised an exception: {e}"


def test_plot_data(
    ss: SingleSetup,
) -> None:
    """
    Test the plotting and data manipulation methods of the SingleSetup.
    """
    initial_data_first_el = ss.data[0][0]
    initial_fs = ss.fs
    initial_dt = ss.dt
    initial_shape = ss.data.shape
    decimation_factor = 4
    initial_T = ss.T

    # test DECIMATE_DATA method
    ss.decimate_data(q=decimation_factor)
    assert ss.data.shape == (initial_shape[0] // decimation_factor, initial_shape[1])
    assert ss.fs != initial_fs
    assert ss.data[0][0] != initial_data_first_el
    assert initial_T != ss.T
    # rollback the data
    ss.rollback()
    assert ss.data.shape == initial_shape
    assert ss.fs == initial_fs
    assert ss.dt == initial_dt
    assert ss.data[0][0] == initial_data_first_el
    assert initial_T == ss.T

    # test DETREND_DATA method
    initial_shape = ss.data.shape
    ss.detrend_data()
    assert math.isclose(ss.data[0][0], -0.12528615865785814)
    assert ss.data.shape == initial_shape
    # rollback the data
    ss.rollback()
    assert ss.data.shape == initial_shape
    assert ss.data[0][0] == initial_data_first_el
    assert ss.fs == initial_fs
    assert ss.dt == initial_dt

    # test FILTER_DATA method
    initial_shape = ss.data.shape
    ss.filter_data(Wn=1, order=1, btype="lowpass")
    assert math.isclose(ss.data[0][0], 0.41342428991383917)
    assert ss.data.shape == initial_shape
    # rollback the data
    ss.rollback()
    assert ss.data.shape == initial_shape
    assert ss.data[0][0] == initial_data_first_el
    assert ss.fs == initial_fs
    assert ss.dt == initial_dt

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


def test_run(ss: SingleSetup) -> None:
    """
    Test the running of the algorithms in the SingleSetup.
    """

    # Define geometry1
    ss.def_geo1_by_file(
        path="./tests/test_data/palisaden/Geo1.xlsx",
    )

    # Define geometry 2
    ss.def_geo2_by_file(path="./tests/test_data/palisaden/Geo2.xlsx")

    # Initialise the algorithms
    fdd = FDD(name="FDD")
    fsdd = FSDD(name="FSDD", nxseg=2048, method_SD="per", pov=0.5)
    ssicov = SSI(name="SSI", br=50, ordmax=80)

    # Overwrite/update run parameters for an algorithm
    fdd.run_params = FDD.RunParamCls(nxseg=512, method_SD="cor")
    # Aggiungere esempio anche col metodo

    # Add algorithms to the single setup class
    ss.add_algorithms(ssicov, fsdd, fdd)

    # results are none
    assert ss["FDD"].result is None
    assert ss["FSDD"].result is None
    assert ss["SSI"].result is None

    # Run all or run by name
    ss.run_by_name("SSI")
    ss.run_by_name("FSDD")

    # Run all algorithms
    ss.run_all()

    # Check the result
    assert ss["FDD"].result is not None
    assert ss["FSDD"].result is not None
    assert ss["SSI"].result is not None

    # plot SINGULAR VALUES
    try:
        fig, ax = fsdd.plot_CMIF(freqlim=(1, 4))
    except Exception as e:
        assert False, f"plot_CMIF raised an exception {e}"

    # plot STABILISATION CHART for SSI
    try:
        fig4, ax4 = ssicov.plot_stab(freqlim=(1, 4), hide_poles=False)
    except Exception as e:
        assert False, f"plot_stab raised an exception {e}"

    # plot FREQUECY-DAMPING CLUSTERS for SSI
    try:
        fig4, ax4 = ssicov.plot_freqvsdamp(freqlim=(1, 4))
    except Exception as e:
        assert False, f"plot_freqvsdamp raised an exception {e}"

    # run mpe_from_plot for algorithms
    try:
        ss.mpe_from_plot("SSI", freqlim=(1, 4))
    except Exception as e:
        assert False, f"mpe_from_plot raised an exception {e} for SSI"

    try:
        ss.mpe_from_plot("FSDD", freqlim=(1, 4))
    except Exception as e:
        assert False, f"mpe_from_plot raised an exception {e} for FSDD"

    try:
        ss.mpe_from_plot("FDD", freqlim=(1, 4))
    except Exception as e:
        assert False, f"mpe_from_plot raised an exception {e} for FDD"

    # run mpe for algorithms
    try:
        ss.mpe("SSI", sel_freq=[1.88, 2.42, 2.68], order_in=40)
    except Exception as e:
        assert False, f"mpe raised an exception {e} for SSI"

    try:
        ss.mpe("FSDD", sel_freq=[1.88, 2.42, 2.68], MAClim=0.95)
    except Exception as e:
        assert False, f"mpe raised an exception {e} for FSDD"

    try:
        ss.mpe("FDD", sel_freq=[1.88, 2.42, 2.68])
    except Exception as e:
        assert False, f"mpe raised an exception {e} for FDD"

    # plot_EFDDfit for FSDD algorithms
    try:
        figs, axs = ss["FSDD"].plot_EFDDfit(freqlim=(1, 4))
        assert isinstance(figs, list)
        assert isinstance(axs, list)
    except Exception as e:
        assert False, f"plot_fit raised an exception {e} for FDD"

    # PLOT_MODE_geo1
    try:
        _ = ss.plot_mode_geo1(algo_res=fdd.result, mode_nr=2, view="3D", scaleF=2)
    except Exception as e:
        assert False, f"plot_mode_geo1 raised an exception {e} for FDD"

    # PLOT_MODE_geo2
    try:
        _ = ss.plot_mode_geo2(algo_res=fsdd.result, mode_nr=2, scaleF=2)
    except Exception as e:
        assert False, f"plot_mode_geo2 raised an exception {e} for FSDD"
