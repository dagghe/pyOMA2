# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2024-02-29

### Added
- `plscf` module for polymax
- `plot_STFT()` to plot the Short Time Fourier Transform magnitude of a channel (time-frequency plot)
- `filter_data()` method to apply a Butterworth filter to the dataset
- origin and reference axes (xyz) in modeshape plots and animations

### Fixed
- axis argument for `detrend_data()` and `decimate_data()` methods
- minor fixes to `plot_data()` method

### Changed
- revised `plot_ch_info()` method to assess quality of data
- `Stab_plot()` and `Cluster_plot()` functions have been revised so that `plot_cluster()` `plot_STDiag()` methods work for both ssi and plscf

## [0.3.2] - 2024-02-17

### Fixed
- link to documentation in toml file
- pyOMA version in the requirements for documentation build
- small fixes in documentation

### Removed
- MAC function from SSI_funct module (restored import from Gen_funct)

## [0.3.1] - 2024-02-17

### Added
- docstring to all classes and functions
- option to save gif figure from animation of the mode shape
- documentation
- logo

### Removed
- old example files under main
- util.py as it was not used

### Changed
- the `freqlim` argument in all the plot function has been changed to a tuple, so to set both an upper and a lower limit to the x (frequency) axis
- moved info.svg under docs/img folder
- moved examples notebooks in Examples folder

### Fixed
- docstring fix for OMA.py
- default ax value to 0 for `detrend_data()` and `decimate_data()` methods
- links to moved items

## [0.3.0] - 2024-02-02

### Added
- methods: `plot_data()`, `plot_ch_info()`, `detrend_data()`, `decimate_data()` to Multisetup preGER class
- info.svg chart to README
- Multisetup PoSER and PreGER examples notebook
- several docstring to functions and classes

### Removed
- channels name from `plot_mode_g1()` method

### Changed
- default ax value to 0 for `detrend_data()` and `decimate_data()` methods
- name to single setup example notebook
- minor reorganisation of `pLSCF_funct.py`
- updated README.md with link to the example notebooks
- add `PYOMA_DISABLE_MATPLOTLIB_LOGGING` env variable to disable matplotlib logging, default to True

### Fixed
- error to `plot_geo2()` method for both SS and MS ***(WARNING posx,posy)***
- restored functions removed by mistake from `Gen_funct.py`

## [0.2.0] - 2024-01-30

### Fixed

- plot size
- small fixes in `OMA.py` and `plot_func.py`

### Added

- methods to plot channels info
- docstring on `SelFormPlot` class
- jupyter notebook for `SingleSetup`


## [0.1.0] - 2024-01-25

### Added

- Initial release of the project

### Removed

### Changed

### Fixed
