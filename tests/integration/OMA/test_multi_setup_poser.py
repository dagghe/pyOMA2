import math

import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.pyoma2.OMA import MultiSetup_PoSER


@pytest.mark.xfail(
    reason="Can't instantiate MultiSetup_PoSER without Fn variable in result."
)
def test_geo1(multi_setup_data_fixture, ms_poser: MultiSetup_PoSER) -> None:
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
    assert ms_poser.Geo1 is None

    ms_poser.def_geo1(
        sens_names=Names,
        sens_coord=sens_coord,
        sens_dir=sens_dir,
        bg_nodes=BG_nodes,
        bg_lines=BG_lines,
    )

    assert ms_poser.Geo1 is not None


@pytest.mark.xfail(
    reason="Can't instantiate MultiSetup_PoSER without Fn variable in result."
)
def test_geo2(multi_setup_data_fixture, ms_poser: MultiSetup_PoSER) -> None:
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
    assert ms_poser.Geo2 is None

    ms_poser.def_geo2(
        sens_names=Names,
        pts_coord=pts_coord,
        sens_map=sens_map,
        order_red="xy",
        sens_sign=sens_sign,
        sens_lines=sens_lines,
        bg_nodes=BG_nodes,
        bg_lines=BG_lines,
    )

    assert ms_poser.Geo2 is not None

    # Merging results from single setups
    result = ms_poser.merge_results()
    # define results variable
    # algoRes = result[SSIcov_algo.__name__]
    algoRes = result["SSIcov_algo"]

    # PLOTE_MODE_G2
    try:
        fig, ax = ms_poser.plot_mode_g2(
            Algo_Res=algoRes, Geo2=ms_poser.Geo2, mode_numb=2, view="3D", scaleF=2
        )
        # assert isinstance(fig, Figure)
        # assert isinstance(ax, Axes)
    except Exception as e:
        assert False, f"plot_mode_g2 raised an exception {e} for FSDD"


@pytest.mark.xfail(
    reason="Can't instantiate MultiSetup_PoSER without Fn variable in result."
)
def test_plot_data(ms_poser: MultiSetup_PoSER) -> None:
    """
    Test the plotting and data manipulation methods of the MultiSetup_PoSER class.
    """
    # test DECIMATE_DATA method not inplace
    assert math.isclose(ms_poser.data[0]["ref"][0][0], -3.249758486587817e-05)
    assert ms_poser.fs == 100.0
    assert ms_poser.dt == 0.01
    decimation_factor = 4
    newdatasets, Y, fs, dt, Ndats, Ts = ms_poser.decimate_data(
        q=decimation_factor, inplace=False
    )
    assert math.isclose(Y[0]["ref"][0][0], -3.27248603574735e-05)
    assert fs == 25.0
    assert dt == 0.01

    # initial class data has not changed
    assert math.isclose(ms_poser.data[0]["ref"][0][0], -3.249758486587817e-05)
    assert ms_poser.fs == 100.0
    assert ms_poser.dt == 0.01

    # test DECIMATE_DATA method inplace
    ms_poser.decimate_data(q=decimation_factor, inplace=True)
    assert math.isclose(ms_poser.data[0]["ref"][0][0], -3.27248603574735e-05)
    assert ms_poser.fs == 25.0
    assert ms_poser.dt == 0.01

    # test DETREND_DATA method not inplace
    new_data = ms_poser.detrend_data(inplace=False)
    assert math.isclose(new_data[0]["ref"][0][0], -3.238227274628828e-05)

    # initial class data has not changed
    assert math.isclose(ms_poser.data[0]["ref"][0][0], -3.27248603574735e-05)

    # test DETREND_DATA method inplace
    ms_poser.detrend_data(inplace=True)
    assert math.isclose(ms_poser.data[0]["ref"][0][0], -3.238227274628828e-05)

    # test FILTER_DATA method not inplace
    new_data = ms_poser.filter_data(Wn=1, order=1, btype="lowpass", inplace=False)
    assert math.isclose(new_data[0]["ref"][0][0], -3.28116336523655e-05)

    # initial class data has not changed
    assert math.isclose(ms_poser.data[0]["ref"][0][0], -3.238227274628828e-05)

    # test FILTER_DATA method inplace
    ms_poser.filter_data(Wn=1, order=1, btype="lowpass", inplace=True)
    assert math.isclose(ms_poser.data[0]["ref"][0][0], -3.28116336523655e-05)

    # test PLOT_DATA method
    try:
        figs, axs = ms_poser.plot_data(data_idx=[0, 1, 2])
        assert isinstance(figs, list)
        assert isinstance(axs, list)
        # assert isinstance(figs[0], Figure)
        # assert isinstance(axs[0], Axes)
    except Exception as e:
        assert False, f"plot_data raised an exception {e}"

    # test PLOT_CH_INFO method
    try:
        figs, axs = ms_poser.plot_ch_info(data_idx=[0, 1, 2], ch_idx=[-1])
        assert isinstance(figs, list)
        assert isinstance(axs, list)
        # assert isinstance(figs[0], Figure)
        # assert isinstance(axs[0][0], Axes)
    except Exception as e:
        assert False, f"plot_ch_info raised an exception {e}"

    # test plot_STFT method
    try:
        figs, axs = ms_poser.plot_STFT(data_idx=[0, 1, 2])
        assert isinstance(figs, list)
        assert isinstance(axs, list)
        assert isinstance(figs[0][0], Figure)
        assert isinstance(axs[0][0], Axes)
    except Exception as e:
        assert False, f"plot_STFT raised an exception {e}"
