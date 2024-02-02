# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
