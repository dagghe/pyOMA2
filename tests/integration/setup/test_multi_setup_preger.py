import math

from pyoma2.algorithms import SSI_MS, pLSCF_MS

from src.pyoma2.setup import MultiSetup_PreGER


def test_geo1(ms_preger: MultiSetup_PreGER) -> None:
    """
    Test the first geometric definition.
    """

    # Test that the geometric is not defined
    assert ms_preger.geo1 is None

    ms_preger.def_geo1_by_file(path="./tests/test_data/3SL/Geo1.xlsx")

    assert ms_preger.geo1 is not None


def test_geo2(ms_preger: MultiSetup_PreGER) -> None:
    """
    Test the second geometric definition.
    """

    # Test that the geometric is not defined
    assert ms_preger.geo2 is None

    ms_preger.def_geo2_by_file(path="./tests/test_data/3SL/Geo2.xlsx")

    assert ms_preger.geo2 is not None


def test_plot_data(ms_preger: MultiSetup_PreGER) -> None:
    """
    Test the plotting and data manipulation methods of the MultiSetup_PreGER class.
    """
    initial_data_first_ref = ms_preger.data[0]["ref"][0][0]
    initial_datasets_first_el = ms_preger.datasets[0][0][0]
    initial_fs = ms_preger.fs
    initial_dt = ms_preger.dt

    # test DECIMATE_DATA method
    decimation_factor = 4
    ms_preger.decimate_data(q=decimation_factor)
    # data has changed and is different from the initial data
    assert math.isclose(ms_preger.data[0]["ref"][0][0], -3.27248603574735e-05)
    assert not math.isclose(ms_preger.data[0]["ref"][0][0], initial_data_first_ref)
    # datasets has changed and is different from the initial datasets
    assert math.isclose(ms_preger.datasets[0][0][0], -3.272486035745707e-05)
    assert not math.isclose(ms_preger.datasets[0][0][0], initial_datasets_first_el)
    assert ms_preger.fs == 25.0
    assert ms_preger.dt == 0.01
    # rollback the data
    ms_preger.rollback()
    assert ms_preger.data[0]["ref"][0][0] == initial_data_first_ref
    assert ms_preger.datasets[0][0][0] == initial_datasets_first_el
    assert ms_preger.fs == initial_fs
    assert ms_preger.dt == initial_dt

    # test DETREND_DATA method
    ms_preger.detrend_data()
    # data has changed and is different from the initial data
    assert math.isclose(ms_preger.data[0]["ref"][0][0], -3.238227274628828e-05)
    assert not math.isclose(ms_preger.data[0]["ref"][0][0], initial_data_first_ref)
    assert ms_preger.fs == initial_fs
    assert ms_preger.dt == initial_dt
    # datasets has changed and is not different from the initial datasets
    assert math.isclose(ms_preger.datasets[0][0][0], -3.249758486587817e-05)
    assert math.isclose(ms_preger.datasets[0][0][0], initial_datasets_first_el)
    # rollback the data
    ms_preger.rollback()
    assert ms_preger.data[0]["ref"][0][0] == initial_data_first_ref
    assert ms_preger.datasets[0][0][0] == initial_datasets_first_el
    assert ms_preger.fs == initial_fs
    assert ms_preger.dt == initial_dt

    # test FILTER_DATA method
    ms_preger.filter_data(Wn=1, order=1, btype="lowpass")
    # data has changed and is different from the initial data
    assert math.isclose(ms_preger.data[0]["ref"][0][0], -3.4815804592448214e-05)
    assert not math.isclose(ms_preger.data[0]["ref"][0][0], initial_data_first_ref)
    assert ms_preger.fs == initial_fs
    assert ms_preger.dt == initial_dt
    # datasets has changed and is not different from the initial datasets
    assert math.isclose(ms_preger.datasets[0][0][0], -3.249758486587817e-05)
    assert math.isclose(ms_preger.datasets[0][0][0], initial_datasets_first_el)
    # rollback the data
    ms_preger.rollback()
    assert ms_preger.data[0]["ref"][0][0] == initial_data_first_ref
    assert ms_preger.datasets[0][0][0] == initial_datasets_first_el
    assert ms_preger.fs == initial_fs
    assert ms_preger.dt == initial_dt

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


def test_run(ms_preger: MultiSetup_PreGER) -> None:
    """
    Test the running of the algorithms in the MultiSetup_PreGER class.
    """
    # Define geometry1
    ms_preger.def_geo1_by_file(path="./tests/test_data/3SL/Geo1.xlsx")  # BG lines

    # Define geometry 2
    ms_preger.def_geo2_by_file(path="./tests/test_data/3SL/Geo2.xlsx")

    # Initialise the algorithms
    ssidat = SSI_MS(name="SSIdat", br=5, ordmax=5)
    plscf = pLSCF_MS(name="pLSCF", ordmax=5, nxseg=64)

    ms_preger.decimate_data(q=50)

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
