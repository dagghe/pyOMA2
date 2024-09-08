import pytest

from src.pyoma2.setup import MultiSetup_PoSER


def test_geo1(ms_poser: MultiSetup_PoSER) -> None:
    """
    Test the first geometric definition.
    """

    # Test that the geometric is not defined
    assert ms_poser.geo1 is None

    # plot without defining the geometry
    with pytest.raises(ValueError) as e:
        ms_poser.plot_geo1()
    assert "geo1 is not defined. Call def_geo1 first." in str(e.value)

    ms_poser.def_geo1_by_file(path="./src/pyoma2/test_data/3SL/Geo1.xlsx")

    assert ms_poser.geo1 is not None

    # Merging results from single setups
    result = ms_poser.merge_results()
    # define results variable
    algo_res = result["SSIcov"]

    # PLOTE_MODE_G1
    try:
        _ = ms_poser.plot_mode_geo1(algo_res=algo_res, mode_nr=2)
    except Exception as e:
        assert False, f"plot_mode_geo1 raised an exception {e} for MultiSetup_PoSER"

    # PLOT geo1
    try:
        _ = ms_poser.plot_geo1()
    except Exception as e:
        assert False, f"plot_geo1 raised an exception {e} for MultiSetup_PoSER"


def test_geo2(ms_poser: MultiSetup_PoSER) -> None:
    """
    Test the second geometric definition.
    """
    # Test that the geometric is not defined
    assert ms_poser.geo2 is None

    # plot without defining the geometry
    with pytest.raises(ValueError) as e:
        ms_poser.plot_geo2()
    assert "geo2 is not defined. Call def_geo2 first." in str(e.value)

    ms_poser.def_geo2_by_file(path="./src/pyoma2/test_data/3SL/Geo2.xlsx")

    assert ms_poser.geo2 is not None

    # Merging results from single setups
    result = ms_poser.merge_results()
    # define results variable
    # algoRes = result[SSIcov.__name__]
    algoRes = result["SSIcov"]

    # PLOTE_MODE_G2
    try:
        _ = ms_poser.plot_mode_geo2(algo_res=algoRes, mode_nr=1, scaleF=3, notebook=True)
        # assert isinstance(fig, Figure)
        # assert isinstance(ax, Axes)
    except Exception as e:
        assert False, f"plot_mode_geo2 raised an exception {e} for MultiSetup_PoSER"

    # PLOT GEO2
    try:
        _ = ms_poser.plot_geo2()
    except Exception as e:
        assert False, f"plot_geo2 raised an exception {e} for MultiSetup_PoSER"
