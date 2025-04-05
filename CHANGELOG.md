# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

* security dependencies patch

## [1.1.1] - 2025-01-24

- fix cluster_plt parameter name to Xi

## [1.1.0] - 2025-01-24

### Fixed

- uncertainty calculations for SSI algorithm
- animation problem in pyvista
- small fix (moved ax.grid()) in plt_data
- updated docs

### Changed

- Renamed `anim_mode_g2` to `anim_mode_geo2` in `GeometryMixin` class
- Updated hierarchy for results and run_params classes
- Renamed `plot_cluster()` method to `plot_freqvsdamp()`
- SSI functions and classes re-organization:
  - `cov_mm`method renamed to `cov`
  - removed `ac2mp` function
  - Hard criteria on MPC and MPD splitted
  - HC on damping and on complex conjugate included into `SSI_poles`function
  - order for run_param renamed to `order_in`
  - Renamed uncertanties component from `xxx_cov`to `xxx_std`
- updated tests

### Added

- pre commit in github workflow
- clustering plotting functions to plot.py


## [1.0.0] - 2024-09-12

**BREAKING CHANGES**

This version introduces a new project structure and a new way to handle the algorithms.
A script is available here https://gist.github.com/dfm88/2bd2a08bb8b5837103dd074033bd7710
to help running the migration to new version.

### Added

- `pyvista` optional dependency for 3D plots
- file `src/pyoma2/support/mpl_plotter.py` to handle matplotlib plots
- file `src/pyoma2/support/pyvista_plotter.py` to handle pyvista plots
- file `src/pyoma2/test_data/3SL/Geo1.xlsx`
- file `src/pyoma2/test_data/3SL/Geo2.xlsx`
- file `src/pyoma2/test_data/5dof/Geo1.xlsx`
- file `src/pyoma2/test_data/5dof/Geo2.xlsx`
- file `src/pyoma2/test_data/Geometry/htc_geo2.xlsx`
- file `src/pyoma2/test_data/Template_Geometry2.xlsx`
- file `src/pyoma2/test_data/palisaden/Geo1.xlsx`
- file `src/pyoma2/test_data/palisaden/Geo2.xlsx`
- platform specific installation in github workflow and requirements extraction in pre-commit


### Changed

- tests to support new project structure
- `OMA.py` removed `inplace` method from `SingleSetup` and `MultiSetup_PreGER` classes, add a copy of data on init with the possibility to `rollback` them
- moved `plot_mode_g1` and `plot_mode_g2` and `anim_mode_g2` methods from `SingleSetup` to `BaseSetup` class
- function `pyoma2/functions/fdd.py::SDOF_bellandMS` now have custom logic for algorithm methods `FSDD` and `EFDD`
- function `pyoma2/functions/plscf.py::pLSCF_poles` now return an additional element in the tuple
- function `pyoma2/functions/plscf.py::ac2mp_poly` now return an additional element in the tuple
- moved all geometry related methods to the `pyoma2/support/geometry/mixin.py` file where the following method are available `def_geo1`, `def_geo2`, `_def_geo_by_file`, `def_geo1_by_file`, `def_geo2_by_file`, `plot_geo1`, `plot_geo2`, `plot_geo2_mpl`, `plot_mode_geo1`, `plot_mode_geo2`, `plot_mode_geo2_mpl`, `anim_mode_geo2` and available to `BaseSetup` class and `MultiSetup_PoSER` class. The proxy to these method were removed from `BaseAlgorithm` class and moved to the mixin class `GeometryMixin` and so proxied by the `Setup` classes that implement the geometry mixin.

- Library re-organization:
  - file `pyoma2/OMA.py` split in `pyoma2/setup` package in the following files:
    - `base.py`: here we moved the following classes `BaseSetup`
    - `single.py`: here we moved the following classes `SingleSetup`
    - `multi.py`: here we moved the following classes `MultiSetup_PreGER`, `MultiSetup_PoSER`
  - file `pyoma2/support/utils/logging_handler.py` moved to `pyoma2/support/utils/logging_handler.py`
  - file `pyoma2/utils/typing.py` moved to `pyoma2/support/utils/typing.py`
  - package `pyoma2/algorithm` renamed to `pyoma2/algorithms`
  - file `pyoma2/algorithm/data/geometry.py` moved to `pyoma2/support/geometry.py`
  - file `pyoma2/plot/Sel_from_plot.py` moved to `pyoma2/support/Sel_from_plot.py`
  - variable `pyoma2/algorithms/data/result.py::SSIResult.xi_poles` renamed to `pyoma2/algorithms/data/result.py::SSIResult.Xi_poles`
  - variable `pyoma2/algorithms/data/result.py::xi_poles.xi_poles` renamed to `pyoma2/algorithms/data/result.py::xi_poles.Xi_poles`
  - variable `pyoma2/algorithms/data/run_params.py::EFDDRunParams.method` renamed to `pyoma2/algorithms/data/run_params.py::EFDDRunParams.method_hank`
  - class `pyoma2/algorithms/fdd.py::FDD_algo` renamed to `pyoma2/algorithms/fdd.py::FDD`
  - class `pyoma2/algorithms/fdd.py::EFDD_algo` renamed to `pyoma2/algorithms/fdd.py::EFDD`
  - class `pyoma2/algorithms/fdd.py::FSDD_algo` renamed to `pyoma2/algorithms/fdd.py::FSDD`
  - class `pyoma2/algorithms/fdd.py::FDD_algo_MS` renamed to `pyoma2/algorithms/fdd.py::FDD_MS`
  - class `pyoma2/algorithms/fdd.py::EFDD_algo_MS` renamed to `pyoma2/algorithms/fdd.py::EFDD_MS`
  - method `pyoma2/algorithms/fdd.py::EFDD.plot_FIT` renamed to `pyoma2/algorithms/fdd.py::EFDD.plot_EFDDfit`
  - function `pyoma2/functions/fdd.py::SD_Est` renamed to `pyoma2/functions/fdd.py::SD_est`
  - function `pyoma2/functions/plscf.py::pLSCF_Poles` renamed to `pyoma2/functions/plscf.py::pLSCF_poles`
  - function `pyoma2/functions/plscf.py::rmfd2AC` renamed to `pyoma2/functions/plscf.py::rmfd2ac`
  - function `pyoma2/functions/plscf.py::AC2MP_poly` renamed to `pyoma2/functions/plscf.py::ac2mp_poly`
  - function `pyoma2/functions/plscf.py::pLSCF_MPE` renamed to `pyoma2/functions/plscf.py::pLSCF_mpe`
  - function `pyoma2/functions/ssi.py::BuildHank` renamed to `pyoma2/functions/ssi.py::build_hank`
  - function `pyoma2/functions/ssi.py::AC2MP` renamed to `pyoma2/functions/ssi.py::ac2mp`
  - function `pyoma2/functions/ssi.py::SSI_FAST` renamed to `pyoma2/functions/ssi.py::SSI_fast`
  - function `pyoma2/functions/ssi.py::SSI_POLES` renamed to `pyoma2/functions/ssi.py::SSI_poles`
  - function `pyoma2/functions/ssi.py::SSI_MulSet` renamed to `pyoma2/functions/ssi.py::SSI_multi_setup`
  - function `pyoma2/functions/ssi.py::SSI_MPE` renamed to `pyoma2/functions/ssi.py::SSI_mpe`
  - file `pyoma2/functions/FDD_funct.py` renamed to `pyoma2/functions/fdd.py`
  - file `pyoma2/functions/plot_funct.py` renamed to `pyoma2/functions/plot.py`
  - file `pyoma2/functions/Gen_funct.py` renamed to `pyoma2/functions/gen.py`
  - file `pyoma2/functions/SSI_funct.py` renamed to `pyoma2/functions/ssi.py`
  - method `pyoma2/algorithms/base.py::BaseAlgorithm.MPE` renamed to `pyoma2/algorithms/base.py::BaseAlgorithm.mpe`
  - method `pyoma2/algorithms/base.py::BaseAlgorithm.mpe_fromPlot` renamed to `pyoma2/algorithms/base.py::BaseAlgorithm.mpe_from_plot`


### Fixed

- python 3.12 support ([akaszynsk](https://github.com/akaszynski))

### Removed

- file `src/pyoma2/plot/anim_mode.py` now handled by pyvista
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/3SL/BG_lines.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/3SL/BG_nodes.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/3SL/geom.xlsx`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/3SL/pts_coord.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/3SL/sens_coord.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/3SL/sens_dir.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/3SL/sens_lines.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/3SL/sens_map.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/3SL/sens_sign.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/palisaden/BG_lines.txt'`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/palisaden/BG_nodes.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/palisaden/geom_pali.xlsx`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/palisaden/pts_coord.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/palisaden/sens_dir.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/palisaden/sens_lines.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/palisaden/sens_map.txt`
- file `https://github.com/dagghe/pyOMA-test-data/tree/main/test_data/palisaden/sens_sign.txt`
- function `pyoma2/functions/ssi.py::Lab_stab_SSI`
- function `pyoma2/functions/gen.py::lab_stab`
- support in testing for `python3.8` and `macos` due to `vtk` dependency that is not compatible with platform using during tests action
- File for example are now retrieved from the online repo https://github.com/dagghe/pyOMA-test-data/tree/main/test_data and removed from the repo

## [0.6.0] - 2024-09-06

## Fixed

- python 3.12 support ([akaszynsk](https://github.com/akaszynski))

## Added

- tox.ini file to run tests locally on multiple python versions

## [0.5.2] - 2024-05-21

### Fixed

- type hints 3.8 compatibility

## [0.5.1] - 2024-04-16

## Fixed
- `multi_setup_poser` tests
- bug "SSI_Poles orders issue #11"
- various minor fixes

## Added
- python 3.12 support
- MPC and MPD criteria on stabilisation diagram for ssi and plscf
- colormap to mode shape animation
- method to save class to file


## [0.5.0] - 2024-04-09

### Added
- issue, feature, question templates

### Changed
- `pre-commit` default formatter to `ruff`
- `OMA.py` moved `decimate_data`, `detrend_data`, `filter_data` to BaseSetup and add `inplace` option default to false

### Fixed
- `mpe` in `FDD` algorithm

### Added
- tests
- workflow for tests on push and pull request

## [0.4.1] - 2024-03-05

### Added
- `pytest-cov` to qa dependencies
- first tests

### Changed
- evaluation types on BaseAlgorithm excluding itself
- more readable error when defining new algorithms
- `_pre_run` on algorithms is now called from setup classes

### Fixed
- `plscf.py` module name https://github.com/dagghe/pyOMA2/issues/5

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
