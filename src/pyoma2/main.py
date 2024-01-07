import numpy as np

from pyoma2.algorithm import EFDD_algo, FDD_algo, FSDD_algo, SSIcov_algo, SSIdat_algo
from pyoma2.OMA import MultiSetup, SingleSetup  # noqa: F401
from pyoma2.utils.utils import read_from_test_data

if __name__ == "__main__":
    """Running single setup"""
    print("**** Running single setup ****")

    # read data
    htc1 = read_from_test_data("src/pyoma2/test_data/3SL/3SL_set1.txt")
    palisaden1 = read_from_test_data(
        "src/pyoma2/test_data/palisaden/Palisaden_3_11082022_DAG.txt"
    )
    # ATTENZIONE
    # non funziona direttamento con dataframe
    palisaden1 = palisaden1.to_numpy()[:, 1:]

    # create single setup 1
    fs1 = 200
    Pali_ss = SingleSetup(palisaden1, fs1)

    Pali_ss.plot_data()

    # # INITIALIZE ALGORITHM WITHOUT RUN PARAMETERS # #
    fdd1 = FDD_algo(name="FDD1")
    # si possono settare dopo o così
    fdd1 = fdd1.set_run_params(
        run_param=FDD_algo.RunParamType(
            fs=fs1, nxseg=1024, method="per", pov=0.67, sel_freq=np.asarray([10, 20, 30])
        )
    )
    # o così
    fdd1.run_params = FDD_algo.RunParamType(
        fs=fs1, nxseg=1024, method="per", pov=0.67, sel_freq=np.asarray([10, 20, 30])
    )

    # # INITIALIZE ALGORITHM WITH RUN PARAMETERS # #
    efdd1 = EFDD_algo(
        name="EFDD1",
        run_params=EFDD_algo.RunParamType(
            fs=fs1, nxseg=1024, method="per", pov=0.67, sel_freq=np.asarray([10, 20, 30])
        ),
    )

    # # INITIALIZE OTHER ALG WITH RUN PARAMETERS # #
    fsdd1 = FSDD_algo(
        name="FSDD1",
        run_params=FSDD_algo.RunParamType(
            fs=fs1, nxseg=1024, method="per", pov=0.67, sel_freq=np.asarray([10, 20, 30])
        ),
    )
    ssicov1 = SSIcov_algo(
        name="SSIcov1",
        run_params=SSIcov_algo.RunParamType(
            fs=fs1,
            br=50,
            method_hank="cov",
            ref_ind=[0, 1, 2, 3],
            ordmin=0,
            ordmax=150,
            step=1,
            err_fn=0.1,
            err_xi=0.05,
            err_phi=0.03,
            xi_max=0.1,
        ),
    )
    ssidat1 = SSIdat_algo(
        name="SSIdat1",
        run_params=SSIdat_algo.RunParamType(
            fs=fs1,
            method_hank="dat",
            br=50,
            ordmax=150,
        ),
    )

    # # ADD ALGORITHMS TO SINGLE SETUP # #
    assert ssidat1.result is None
    assert ssicov1.result is None
    assert efdd1.result is None
    assert fsdd1.result is None
    assert fdd1.result is None

    # FIXME SSI si rompe al momento
    # Pali_ss.add_algorithms(ssidat1, ssicov1, efdd1, fsdd1, fdd1)
    Pali_ss.add_algorithms(efdd1, fsdd1, fdd1)
    Pali_ss.run_all()

    # assert ssidat1.result is not None
    # assert ssicov1.result is not None
    assert efdd1.result is not None
    assert fsdd1.result is not None
    assert fdd1.result is not None

    # oppure accedo agli algoritmi dal single setup (puoi usare il name come string o dalla variabile name)
    # assert Pali_ss[ssidat1.name].result is not None
    # assert Pali_ss[ssicov1.name].result is not None
    assert Pali_ss[efdd1.name].result is not None
    assert Pali_ss[fsdd1.name].result is not None
    assert Pali_ss[fdd1.name].result is not None
