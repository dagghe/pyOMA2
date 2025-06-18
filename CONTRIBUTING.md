# Contribution Guide

This project runs on python>=3.8,<3.13

## Setup

We use PDM as a dependency manager. Check the updated installation instructions from here, or follow these steps:

Linux/MAC:

```shell
# Install PDM Linux/MAC

curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

Windows:

```powershell
# Install PDM Windows

(Invoke-WebRequest -Uri https://pdm-project.org/install-pdm.py -UseBasicParsing).Content | python -
```

Add PATH to environment manager and then run the appropriate command to install all the dependencies based on your Python version and operating system:

### For Python 3.8

Linux:
```shell
pdm install --lockfile=pdm-py38unix.lock
```

Windows:
```shell
pdm install --lockfile=pdm-py38win.lock
```

macOS:
If you are using macOS with Python 3.8, you need to manually install the `vtk` package due to compatibility issues. You can do this by running the following command:

```sh
pip install https://files.pythonhosted.org/packages/b3/15/40f8264f1b5379f12caf0e5153006a61c1f808937877c996e907610e8f23/vtk-9.3.1-cp38-cp38-macosx_10_10_x86_64.whl
```shell
pdm install --lockfile=pdm-py38macos.lock
```

### For Python 3.9 and above

Linux:
```shell
pdm install --lockfile=pdm-py39+unix.lock
```

Windows:
```shell
pdm install --lockfile=pdm-py39+win.lock
```

macOS:
```shell
pdm install --lockfile=pdm-py39+macos.lock
```

### Using requirements.txt files

The corresponding `requirements.txt` files are generated during pre-commit hooks and located in the `/requirements` folder.

### Adding new packages

When adding a new package, make sure to update the correct lock file(s). For example:

For Python 3.8 Windows
```shell
pdm add <package_name> --lockfile=pdm-py38win.lock
```

For Python 3.8 Linux
```shell
pdm add <package_name> --lockfile=pdm-py38unix.lock
```

For Python 3.8 (macOS):
```shell
pdm add <package_name> --lockfile=pdm-py38macos.lock
```

For Python 3.9+ Linux
```shell
pdm add <package_name> --lockfile=pdm-py39+unix.lock
```

For Python 3.9+ (macOS):
```shell
pdm add <package_name> --lockfile=pdm-py39+macos.lock
```

Remember to update all relevant lock files when adding or updating dependencies.

### Regenerate the lock file

If you need to regenerate the lock file, you can use the following command:

```shell
# Example for macos and Python 3.8
pdm lock --python="==3.8.*" --platform=macos --with pyvista --with openpyxl --lockfile=pdm-py38macos.lock
```

```shell
# Example for Windows and Python 3.9+
pdm lock --python="==3.8.*" --platform=windows --with pyvista --with openpyxl --lockfile=pdm-py39+win.lock
```

```shell
# Example for Linux and Python 3.9+
pdm lock --python="==3.8.*" --platform=linux --with pyvista --with openpyxl --lockfile=pdm-py39+unix.lock
```

### Install pre-commit

```shell
pdm run pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Run the project

Linux/MAC:
```shell
pdm run src/pyoma2/main.py
```

## Updating lock files

To update the lock files for different platforms and Python versions, use the following commands:

For Python 3.8 (Linux/Windows):
```shell
pdm lock --python="3.8" --lockfile=pdm-py38.lock
```

For Python 3.8 (macOS):
```shell
pdm lock --python="3.8" --platform=macos --lockfile=pdm-py38macos.lock
```

For Python 3.9+ (Linux/Windows):
```shell
pdm lock --python=">=3.9" --lockfile=pdm-py39+.lock
```

For Python 3.9+ (macOS):
```shell
pdm lock --python=">=3.9" --platform=macos --lockfile=pdm-py39+macos.lock
```

Make sure to update all relevant lock files when making changes to the project dependencies.

Windows

```powershell
pdm run .\src\pyoma2\main.py
```

You'll probably need to install **tk** for the GUI on your system, here some instructions:

Windows:

https://www.pythonguis.com/installation/install-tkinter-windows/


Linux:

https://www.pythonguis.com/installation/install-tkinter-linux/

Mac:

https://www.pythonguis.com/installation/install-tkinter-mac/

If using python with `pyenv`:

https://dev.to/xshapira/using-tkinter-with-pyenv-a-simple-two-step-guide-hh5

### Building the lock file

Due to [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html), Numpy drops support for active versions of Python before their support ends. Therefore, there are versions of numpy that cannot be installed for certain active versions of Python and this leads to PDM unable to resolve the dependencies, or attempting to install a version of numpy that does not have a wheel.

By following [Lock for specific platforms or Python versions](https://pdm-project.org/en/latest/usage/lock-targets/), you can generate a single lock file for multiple versions of Python with:

```
pdm lock --python=">=3.9" --with pyvista --with openpyxl
pdm lock --python="==3.8.*" --with pyvista --with openpyxl
```

When bumping the minimum supported version of Python in `pyproject` (`requires-python`), be sure to also bump the conditional numpy versions supported. For example, when Python 3.8 is dropped, you will have to modify:

```
    "numpy<1.25; python_version < '3.9'",
    "numpy>=1.25; python_version >= '3.9'",
```

to (this is just a guess; numpy versions will have to change):

```
    "numpy<2.0; python_version < '3.10'",
    "numpy>=2.0; python_version >= '3.10'",
```

## Running tests

```shell
make test
```

### Running tests with coverage

```shell
make test-coverage
```

### Running tests with tox on multiple python versions

```shell
make tox
```

## Docs

To build the documentation, you can install the required `docs` dependencies using PDM:

```shell
pdm install --with docs
```
Then, you can build the documentation using Sphinx:

```shell
cd docs
make html
```

An HTML version of the documentation will be generated in the `_build/html` directory. You can open the `index.html` file in a web browser to view the documentation.

### Running tests on

## Conventions

### Commits

Use conventional commits guidelines https://www.conventionalcommits.org/en/v1.0.0/
