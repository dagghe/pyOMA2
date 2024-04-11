import pytest

from src.pyoma2.OMA import MultiSetup_PoSER


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

    # plot without defining the geometry
    with pytest.raises(ValueError) as e:
        ms_poser.plot_geo1()
    assert "Geo1 is not defined. cannot plot geometry" in str(e.value)

    ms_poser.def_geo1(
        sens_names=Names,
        sens_coord=sens_coord,
        sens_dir=sens_dir,
        bg_nodes=BG_nodes,
        bg_lines=BG_lines,
    )

    assert ms_poser.Geo1 is not None

    # Merging results from single setups
    result = ms_poser.merge_results()
    # define results variable
    algo_res = result["SSIcov_algo"]

    # PLOTE_MODE_G1
    try:
        fig, ax = ms_poser.plot_mode_g1(
            Algo_Res=algo_res, Geo1=ms_poser.Geo1, mode_numb=2
        )
    except Exception as e:
        assert False, f"plot_mode_g1 raised an exception {e} for MultiSetup_PoSER"

    # PLOT geo1
    try:
        fig, ax = ms_poser.plot_geo1()
    except Exception as e:
        assert False, f"plot_geo1 raised an exception {e} for MultiSetup_PoSER"


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

    # plot without defining the geometry
    with pytest.raises(ValueError) as e:
        ms_poser.plot_geo2()
    assert "Geo2 is not defined. cannot plot geometry" in str(e.value)

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
            Algo_Res=algoRes, Geo2=ms_poser.Geo2, mode_numb=2, view="xy", scaleF=2
        )
        # assert isinstance(fig, Figure)
        # assert isinstance(ax, Axes)
    except Exception as e:
        assert False, f"plot_mode_g2 raised an exception {e} for MultiSetup_PoSER"

    # PLOT GEO2
    try:
        fig, ax = ms_poser.plot_geo2()
    except Exception as e:
        assert False, f"plot_geo2 raised an exception {e} for MultiSetup_PoSER"
