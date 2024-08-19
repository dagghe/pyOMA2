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

Add PATH to environment manager and then run this command to install all the dependencies

Install dependencies

```shell
pdm install
```

Install pre-commit

```shell
pdm run pre-commit install --hook-type pre-commit --hook-type pre-push
```
 Run the project

 Linux/MAC

 ```shell
 pdm run src/pyoma2/main.py
 ```

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
pdm lock --python=">=3.9"
pdm lock --python="<3.9" --append
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

### Running tests on

## Conventions

### Commits

Use conventional commits guidelines https://www.conventionalcommits.org/en/v1.0.0/
