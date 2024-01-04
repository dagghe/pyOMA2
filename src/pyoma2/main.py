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
    fs1 = 200
    Pali_ss = SingleSetup(palisaden1, fs1)

    # Pali_ss.add_algorithms(FDD_algo())

    # create algorithms
    fdd1 = FDD_algo(name="FDD1")
    fdd2 = FDD_algo(name="FDD2",
                run_params=FDDRunParams(fs=fs1,
                                        nxseg=512, 
                                        method="per", 
                                        pov=0.67)
                )

    ssicov1 = SSIcov_algo(name="SSIcov1", 
                         run_params=SSIcovRunParams(br=50,
                         # not necessary but higly reccomended:
                                                    # ordmax=150,
                                                    )
                         )

    rp = SSIcovRunParams(br=50,
                         ref_id = [0,1,2,3],
                         ordmin = 0,
                         ordmax = 150,
                         step = 1,
                         err_fn = 0.1,
                         err_xi = 0.05,
                         err_phi = 0.03,
                         xi_max = 0.1,
        )

    ssicov2 = SSIcov_algo(name="SSIcov2", run_params=rp)

    # add algorithms to single setup 1
    Pali_ss.add_algorithms(fdd1, fdd2, ssicov1, ssicov2)

    # run all algorithms in single setup 1
    Pali_ss.run_all()

    # run by name
    Pali_ss.run_by_name("FDD1")
    Pali_ss.run_by_name(fdd1.name)


    # AGGIUNGERE METODI PER PLOT "STATICI" (dove pero?!)
    fig, ax = Pali_ss.Plot_MIF("FDD1") # .Plot_MIF(fdd1, *args)
    fig, ax = Pali_ss.Plot_StDiag("ssicov1") # .Plot_MIF(ssicov2, *args)


    # SISTEMARE CLASSE PER PLOT "INTERATTIVI"
    Pali_ss.MPE_fromPlot("FDD1")
    Pali_ss.MPE_fromPlot("SSIcov1")


    # exctract known freq
    sel_freq=[1.,2.,3.]
    Pali_ss.MPE("FDD1", sel_freq=sel_freq)
    Pali_ss.MPE("SSIcov1", sel_freq=sel_freq, )


    # # define another setup
    # data2 = [7, 8, 9, 10]
    # fs2 = 20
    # single_setup_2 = SingleSetup(data2, fs2)

    # # add algorithms to single setup 2
    # single_setup_2.add_algorithms(fdd, ssicov)

    # print("\n**** Running multi setup ****")
    # # define Multi Setup
    # multi_setup = MultiSetup([data1, data2], [fs1, fs2])
    # # add setups to multi setup
    # multi_setup.add_setups(single_setup_1, single_setup_2)
    # # run all algorithms in multi setup
    # multi_setup.run_all()
