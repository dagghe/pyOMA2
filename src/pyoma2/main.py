import numpy as np

from pyoma2.algorithm import FDD_algo, SSIcov_algo
from pyoma2.algorithm.run_params import FDDRunParams, SSIcovRunParams
from pyoma2.OMA import MultiSetup, SingleSetup
from pyoma2.utils.utils import read_from_test_data

if __name__ == "__main__":
    """Running single setup"""
    print("**** Running single setup ****")

    htc1 = read_from_test_data("src/pyoma2/test_data/3SL/3SL_set1.txt")
    palisaden1 = read_from_test_data(
        "src/pyoma2/test_data/palisaden/Palisaden_3_11082022_DAG.txt"
    )

    # create single setup 1
    data1 = np.asarray([1, 2, 3, 4, 5])
    fs1 = 10
    single_setup_1 = SingleSetup(data1, fs1)

    # create algorithms
    fdd = FDD_algo(run_params=FDDRunParams(df=0.01, pov=0.5, window="hann"))
    ssicov = SSIcov_algo(run_params=SSIcovRunParams(br=3, ordmin=0, ordmax="1"))

    # add algorithms to single setup 1
    single_setup_1.add_algorithms(fdd, ssicov)

    # run all algorithms in single setup 1
    single_setup_1.run_all()

    # define another setup
    data2 = [7, 8, 9, 10]
    fs2 = 20
    single_setup_2 = SingleSetup(data2, fs2)

    # add algorithms to single setup 2
    single_setup_2.add_algorithms(fdd, ssicov)

    print("\n**** Running multi setup ****")
    # define Multi Setup
    multi_setup = MultiSetup([data1, data2], [fs1, fs2])
    # add setups to multi setup
    multi_setup.add_setups(single_setup_1, single_setup_2)
    # run all algorithms in multi setup
    multi_setup.run_all()
