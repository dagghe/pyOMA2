import abc
import typing

import algorithm
import functions


class SingleSetup:
    def __init__(self, data: typing.Iterable[float], fs: float):
        self.data = data # data 
        self.fs = fs # sampling frequency
        self.dt = 1/fs # sampling interval
        self.algorithms = [] # set of algo 

    # add algorithm (method) to the set.
    def add_algorithms(self, *algorithms: algorithm.algorithm.BaseAlgorithm):
        self.algorithms.extend(algorithms)

    # find algorithm (method) from the available ones.
    def _find_algorithm(self, name: str) -> algorithm.algorithm.BaseAlgorithm:
        for alg in self.algorithms:
            if alg.name == name:
                return alg
        raise ValueError(f"Algorithm {name} not found")

    # run the whole set of algorithms (methods).
    def run_all(self):
        for alg in self.algorithms:
            print(f"Running {alg.name} with params {alg.run_params}")
            alg.run(data=self.data, fs=self.fs)
        print("all done")

    # run algorithm (method) by name. QUESTO è IL METODO 1
    def run_by_name(self, name: str):
        alg = self._find_algorithm(name)
        if alg:
            print(f"Running {alg.name} with params {alg.run_params}")
            return alg.run(data=self.data, fs=self.fs)
        print("run by name done")

    # get the modal properties (all results).
    #  QUESTO è IL METODO 2 (manuale)
    def MPE(self, name: str, *args, **kwargs):
        alg = self._find_algorithm(name)
        if alg:
            print(f"Getting modal parameters from {alg.name}")
            return alg.mpe(*args, **kwargs)
        print("***DONE***")

    # get the modal properties (all results) from the plots.
    # QUESTO è IL METODO 2 (grafico)
    def MPE_fromPlot(self, name: str, *args, **kwargs):
        alg = self._find_algorithm(name)
        if alg:
            print(f"Getting modal parameters from plot... {alg.name}")
            return alg.mpe_fromPlot(*args, **kwargs)
        print("***DONE***")
        
    # method to plot the time histories of the data channels.
    def plot_data(self, nc: int=1, names: None | list[str]=None, 
                  unit: str="unit", show_rms: bool=False, 
                  len_Wrms: None | int=None ):
        data = self.data
        dt = self.dt
        nc = nc # number of columns for subplot
        names = names # list of names (str) of the channnels
        unit = unit # str label for the y-axis (unit of measurement)
        show_rms = show_rms # wheter to show or not the rms acc in the plot
        len_Wrms = len_Wrms # lenght of window for rms calc
        # N.B. if len_Wrms is int then it's the lenght of the window to
        # to calculate the rms acc, if None then the window is taken as the
        # whole signal

        fig, ax = plt_data(data, dt, nc, names, unit, show_rms, len_Wrms)
        # possiamo salvarli questi ora?
    
    # metodo per definire geometria 1
    def def_geo1(self,):
        pass
    
    # metodo per definire geometria 2
    def def_geo2(self,):
        pass


# LA CLASSE MULTISETUP VA PROBABILMENTE RIVISTA...
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
            for alg in setup.algorithms:
                if not alg.result:
                    raise ValueError(f"Run {alg.name} first")
                print(f"Merging {alg.name} results")