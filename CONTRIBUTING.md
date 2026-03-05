# Contributing to pyOMA2

## **Recent git history update at 205-03-18** ⚠️

Dear contributors,

We've successfully performed maintenance on the pyOMA2 Git repository to remove large binary files from the history, which was causing excessive repository size. This required a rewrite of the repository history.

If you have a local copy of the repository, you'll need to update it using one of these methods:

Option 1 (Recommended) - Fresh clone:

```shell
git clone https://github.com/dagghe/pyOMA2.git
```

Option 2 - Update existing repository:

```shell
# Make sure to commit or stash any local changes first
git fetch origin
git reset --hard origin/main
```
Please note that attempting to push or pull without performing the above steps will result in Git errors due to the diverged history.

If you have any questions or encounter any issues, please let us know.

____________________________________

Welcome! **pyOMA2** is a highly technical Python library focused on **Operational Modal Analysis (OMA)**. As such, contributions can come from both **programmers** (for bug fixes, best practices, and performance improvements) and **field experts** (for validation, methodology suggestions, and real-world applications). We appreciate any contributions that help improve the library!

## How to Get Involved

We welcome contributions from both developers and OMA experts in different ways:

- **Report Issues**: If you encounter a bug, have a feature request, or find something unclear in the documentation, please open a [GitHub Issue](https://github.com/dagghe/pyOMA2/issues).
- **Code Contributions**: If you are a developer, you can help improve the codebase, fix bugs, enhance performance, or improve best practices by submitting a Pull Request (PR).
- **Domain Expertise Contributions**: If you are an OMA specialist or engineer, we would love your input on methodology, validation of results, and possible improvements to the implemented techniques.
- **Documentation Improvements**: If you notice unclear explanations or missing information, feel free to suggest improvements.
- **Discussions & Questions**: If you're unsure about something, feel free to start a discussion in the Issues section.

## Contribution Guidelines

Before making a contribution, please ensure you follow these best practices:

- Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) guidelines.
- Ensure your changes do not break existing functionality.
- Write tests for new features where applicable.
- Be respectful and follow our community guidelines.

## Setup

We use [uv](https://docs.astral.sh/uv/) as a package manager. Install it following the [official instructions](https://docs.astral.sh/uv/getting-started/installation/).

Then install all dependencies:

```shell
uv sync --group qa
```

### Adding new packages

```shell
uv add <package_name>
```

### Regenerating the lock file

```shell
uv lock
```

### Install pre-commit

```shell
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
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

To build the documentation, install the required `docs` dependencies:

```shell
uv sync --group docs
```

Then build the documentation using Sphinx:

```shell
cd docs
make html
```

An HTML version of the documentation will be generated in the `_build/html` directory. You can open the `index.html` file in a web browser to view the documentation.

## Run the project

Linux/MAC:
```shell
uv run src/pyoma2/main.py
```

Windows:

```powershell
uv run .\src\pyoma2\main.py
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
