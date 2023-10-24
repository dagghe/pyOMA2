import abc
import typing

import algorithm


class SingleSetup:
    def __init__(self, data: typing.Iterable[float], fs: float):
        self.data = data
        self.fs = fs # sampling frequency
        self.dt = 1/fs # sampling interval
        self.algorithms = []

    def add_algorithms(self, *algorithms: algorithm.algorithm.BaseAlgorithm):
        self.algorithms.extend(algorithms)


    def _find_algorithm(self, name: str) -> algorithm.algorithm.BaseAlgorithm:
        for algorithm in self.algorithms:
            if algorithm.name == name:
                return algorithm
        raise ValueError(f"Algorithm {name} not found")


    def run_all(self):
        for algorithm in self.algorithms:
            print(f"Running {algorithm.name} with params {algorithm.run_params}")
            algorithm.run()
        print("all done")

    def run_by_name(self, name: str):
        algorithm = self._find_algorithm(name)
        if algorithm:
            print(f"Running {algorithm.name} with params {algorithm.run_params}")
            return algorithm.run()
        print("run by name done")

    def MPE(self, name: str, *args, **kwargs):
        algorithm = self._find_algorithm(name)
        if algorithm:
            print(f"Getting modal parameters from {algorithm.name}")
            return algorithm.mpe(*args, **kwargs)
        print("***DONE***")

    def MPE_fromPlot(self, name: str, *args, **kwargs):
        algorithm = self._find_algorithm(name)
        if algorithm:
            print(f"Getting modal parameters from plot... {algorithm.name}")
            return algorithm.mpe_fromPlot(*args, **kwargs)
        print("***DONE***")



class MultiSetup:
    """Come il Single setup ma prende nell'init una lista degli stessi argomenti"""
    def __init__(self, data: list[typing.Iterable[float]], fs: list[float]):
        self.data = data
        self.fs = fs
        self.setups = []

    def add_setups(self, *setups: SingleSetup):
        self.setups.extend(setups)
    
    def run_all(self):
        for setup in self.setups:
            setup.run_all()
        print("all done")

    def merge_results(self):
        for setup in self.setups:
            for algorithm in setup.algorithms:
                if not algorithm.result:
                    raise ValueError(f"Run {algorithm.name} first")
                print(f"Merging {algorithm.name} results")