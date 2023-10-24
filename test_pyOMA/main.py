import algorithm
import OMA

if __name__ == "__main__":
    """Running single setup"""
    print("**** Running single setup ****")

    # create single setup 1
    data1 = [1, 2, 3, 4, 5]
    fs1 = 10
    single_setup_1 = setup.SingleSetup(data1, fs1)

    # create algorithms
    fdd = algorithm.FDDAlgorithm(
        run_params=algorithm.run_params.FDDRunParams(df=0.01, pov=0.5, window="hann")
    )
    ssicov = algorithm.SSIcovAlgorithm(
        run_params=algorithm.run_params.SSIcovRunParams(br=3, ordmin=0, ordmax="1")
    )

    # add algorithms to single setup 1
    single_setup_1.add_algorithms(fdd, ssicov)

    # run all algorithms in single setup 1
    single_setup_1.run_all()

    # define another setup
    data2 = [7, 8, 9, 10]
    fs2 = 20
    single_setup_2 = setup.SingleSetup(data2, fs2)

    # add algorithms to single setup 2
    single_setup_2.add_algorithms(fdd, ssicov)

    print("\n**** Running multi setup ****")
    # define Multi Setup
    multi_setup = setup.MultiSetup([data1, data2], [fs1, fs2])
    # add setups to multi setup
    multi_setup.add_setups(single_setup_1, single_setup_2)
    # run all algorithms in multi setup
    multi_setup.run_all()
