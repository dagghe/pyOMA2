# Contribution Guide

This project runs on python>=3.8,<3.11

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

## Conventions

### Commits

Use conventional commits guidelines https://www.conventionalcommits.org/en/v1.0.0/
