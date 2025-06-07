# pyOMA2

![pyoma2_logo_v2_COMPACT](https://github.com/dagghe/pyOMA2/assets/64746269/aa19bc05-d452-4749-a404-b702e6fe685d)

[![python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Test Pyoma2](https://github.com/dagghe/pyOMA2/actions/workflows/main.yml/badge.svg?branch=main&event=push)](https://github.com/dagghe/pyOMA2/actions/workflows/main.yml)
![downloads](https://img.shields.io/pepy/dt/pyOMA-2)
[![docs](https://readthedocs.org/projects/pyoma/badge/?version=main)](https://pyoma.readthedocs.io/en/main/)
_______________________

This is the new and updated version of pyOMA module, a Python module designed for conducting operational modal analysis.
With this update, we've transformed pyOMA from a basic collection of functions into a more sophisticated module that fully leverages the capabilities of Python classes.

Key Features & Enhancements:

- Support for single and multi-setup measurements, which includes handling multiple acquisitions with mixed reference and roving sensors.
- Interactive plots for intuitive mode selection, users can now extract desired modes directly from algorithm-generated plots.
- Structure geometry definition, enabling 3D visualization of mode shapes once modal results are obtained.
- Uncertainty estimation for modal properties in Stochastic Subspace Identification (SSI) algorithms.
- Specialized clustering classes for Automatic OMA using SSI, streamlining modal parameter extraction.
- New OMAX (OMA with Exogenous Input) functionality for SSI, expanding the module’s capabilities to handle forced excitation scenarios.

## Documentation

You can check the documentation at the following link:

https://pyoma.readthedocs.io/en/main/

## Quick start

Install the library with pip:

```shell
pip install pyOMA-2
```

or with conda/mamba:

```shell
conda install pyOMA-2
```

You'll probably need to install **tk** for the GUI on your system, here some instructions:

Windows:

https://www.pythonguis.com/installation/install-tkinter-windows/

Linux:

https://www.pythonguis.com/installation/install-tkinter-linux/

Mac:

https://www.pythonguis.com/installation/install-tkinter-mac/

_____

# Examples

To see how the module works please take a look at the jupyter notebook provided:

- [Example1 - Getting started.ipynb](Examples/Example1.ipynb)
- [Example2 - Real dataset.ipynb](Examples/Example2.ipynb)
- [Example3 - Multisetup PoSER.ipynb](Examples/Example3.ipynb)
- [Example4 - MultiSetup PreGER.ipynb](Examples/Example4.ipynb)
- [Example5 - Clustering for Automatic OMA.ipynb](Examples/Example5.ipynb)
- [Extra - Tips and Tricks 1.ipynb](Examples/Extra1.ipynb)

_____

# Schematic organisation of the module showing inheritance between classes

![](docs/img/info.png "")

____
