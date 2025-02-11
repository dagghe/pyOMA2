======================
Contributing to pyOMA2
======================

Welcome! **pyOMA2** is a highly technical Python library focused on **Operational Modal Analysis (OMA)**. As such, contributions can come from both **programmers** (for bug fixes, best practices, and performance improvements) and **field experts** (for validation, methodology suggestions, and real-world applications). We appreciate any contributions that help improve the library!

-------------------
How to Get Involved
-------------------

We welcome contributions from both developers and OMA experts in different ways:

- **Report Issues**: If you encounter a bug, have a feature request, or find something unclear in the documentation, please open a [GitHub Issue](https://github.com/dagghe/pyOMA2/issues).

- **Code Contributions**: If you are a developer, you can help improve the codebase, fix bugs, enhance performance, or improve best practices by submitting a Pull Request (PR).

- **Domain Expertise Contributions**: If you are an OMA specialist or engineer, we would love your input on methodology, validation of results, and possible improvements to the implemented techniques.

- **Documentation Improvements**: If you notice unclear explanations or missing information, feel free to suggest improvements.

- **Discussions & Questions**: If you're unsure about something, feel free to start a discussion in the Issues section.

-----------------------
Contribution Guidelines
-----------------------

Before making a contribution, please ensure you follow these best practices:

- Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) guidelines.
- Ensure your changes do not break existing functionality.
- Write tests for new features where applicable.
- Be respectful and follow our community guidelines.

-----
Setup
-----

We use PDM as a dependency manager. Check the updated installation instructions from here, or follow these steps:

----------------------
Install PDM Linux/MAC
----------------------

.. code-block:: shell

    curl -sSL https://pdm-project.org/install-pdm.py | python3 -

-------------------
Install PDM Windows
-------------------

.. code-block:: shell

    (Invoke-WebRequest -Uri https://pdm-project.org/install-pdm.py -UseBasicParsing).Content | python -

Add PATH to environment manager and then run the appropriate command to install all the dependencies based on your Python version and operating system:

^^^^^^^^^^^^^^
For Python 3.8
^^^^^^^^^^^^^^

**Linux**:

.. code-block:: shell

    pdm install --lockfile=pdm-py38unix.lock

**Windows**:

.. code-block:: shell

    pdm install --lockfile=pdm-py38win.lock

**macOS**:

If you are using macOS with Python 3.8, you need to manually install the `vtk` package due to compatibility issues. You can do this by running the following command:

.. code-block:: shell

    pip install https://files.pythonhosted.org/packages/b3/15/40f8264f1b5379f12caf0e5153006a61c1f808937877c996e907610e8f23/vtk-9.3.1-cp38-cp38-macosx_10_10_x86_64.whl
    pdm install --lockfile=pdm-py38macos.lock

^^^^^^^^^^^^^^^^^^^^^^^^
For Python 3.9 and above
^^^^^^^^^^^^^^^^^^^^^^^^

**Linux**:

.. code-block:: shell

    pdm install --lockfile=pdm-py39+unix.lock

**Windows**:

.. code-block:: shell

    pdm install --lockfile=pdm-py39+win.lock

**macOS**:

.. code-block:: shell

    pdm install --lockfile=pdm-py39+macos.lock

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using requirements.txt files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The corresponding `requirements.txt` files are generated during pre-commit hooks and located in the `/requirements` folder.

^^^^^^^^^^^^^^^^^^^
Adding new packages
^^^^^^^^^^^^^^^^^^^

When adding a new package, make sure to update the correct lock file(s). For example:

**For Python 3.8 Windows**:

.. code-block:: shell

    pdm add <package_name> --lockfile=pdm-py38win.lock

**For Python 3.8 Linux**:

.. code-block:: shell

    pdm add <package_name> --lockfile=pdm-py38unix.lock

**For Python 3.8 (macOS)**:

.. code-block:: shell

    pdm add <package_name> --lockfile=pdm-py38macos.lock

**For Python 3.9+ Linux**:

.. code-block:: shell

    pdm add <package_name> --lockfile=pdm-py39+unix.lock

**For Python 3.9+ (macOS)**:

.. code-block:: shell

    pdm add <package_name> --lockfile=pdm-py39+macos.lock

Remember to update all relevant lock files when adding or updating dependencies.

^^^^^^^^^^^^^^^^^^^^^^^^
Regenerate the lock file
^^^^^^^^^^^^^^^^^^^^^^^^

--------------------------------
Example for macos and Python 3.8
--------------------------------

.. code-block:: shell

    pdm lock --python="==3.8.*" --platform=macos --with pyvista --with openpyxl --lockfile=pdm-py38macos.lock

-----------------------------------
Example for Windows and Python 3.9+
-----------------------------------

.. code-block:: shell

    pdm lock --python="==3.8.*" --platform=windows --with pyvista --with openpyxl --lockfile=pdm-py39+win.lock

---------------------------------
Example for Linux and Python 3.9+
---------------------------------

.. code-block:: shell

    pdm lock --python="==3.8.*" --platform=linux --with pyvista --with openpyxl --lockfile=pdm-py39+unix.lock

^^^^^^^^^^^^^^^^^^
Install pre-commit
^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    pdm run pre-commit install --hook-type pre-commit --hook-type pre-push

---------------
Run the project
---------------

**Linux/MAC**:

.. code-block:: shell

    pdm run src/pyoma2/main.py

-------------------
Updating lock files
-------------------

To update the lock files for different platforms and Python versions, use the following commands:

**For Python 3.8 (Linux/Windows)**:

.. code-block:: shell

    pdm lock --python="3.8" --lockfile=pdm-py38.lock

**For Python 3.8 (macOS)**:

.. code-block:: shell

    pdm lock --python="3.8" --platform=macos --lockfile=pdm-py38macos.lock

**For Python 3.9+ (Linux/Windows)**:

.. code-block:: shell

    pdm lock --python=">=3.9" --lockfile=pdm-py39+.lock

**For Python 3.9+ (macOS)**:

.. code-block:: shell

    pdm lock --python=">=3.9" --platform=macos --lockfile=pdm-py39+macos.lock

Make sure to update all relevant lock files when making changes to the project dependencies.

**Windows Execution**:

.. code-block:: shell

    pdm run .\src\pyoma2\main.py

You'll probably need to install **tk** for the GUI on your system. Here are some instructions:

- **Windows**: https://www.pythonguis.com/installation/install-tkinter-windows/
- **Linux**: https://www.pythonguis.com/installation/install-tkinter-linux/
- **Mac**: https://www.pythonguis.com/installation/install-tkinter-mac/
- **Using python with `pyenv`**: https://dev.to/xshapira/using-tkinter-with-pyenv-a-simple-two-step-guide-hh5

^^^^^^^^^^^^^^^^^^^^^^
Building the lock file
^^^^^^^^^^^^^^^^^^^^^^

Due to `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_, Numpy drops support for active versions of Python before their support ends. Therefore, there are versions of numpy that cannot be installed for certain active versions of Python and this leads to PDM unable to resolve the dependencies, or attempting to install a version of numpy that does not have a wheel.

By following `Lock for specific platforms or Python versions <https://pdm-project.org/en/latest/usage/lock-targets/>`_, you can generate a single lock file for multiple versions of Python with:

.. code-block:: shell

    pdm lock --python=">=3.9" --with pyvista --with openpyxl
    pdm lock --python="==3.8.*" --with pyvista --with openpyxl

When bumping the minimum supported version of Python in `pyproject` (`requires-python`), be sure to also bump the conditional numpy versions supported. For example, when Python 3.8 is dropped, you will have to modify:

.. code-block:: shell

    "numpy<1.25; python_version < '3.9'",
    "numpy>=1.25; python_version >= '3.9'",

to (this is just a guess; numpy versions will have to change):

.. code-block:: shell

    "numpy<2.0; python_version < '3.10'",
    "numpy>=2.0; python_version >= '3.10'",

-------------
Running tests
-------------

.. code-block:: shell

    make test

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Running tests with coverage
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    make test-coverage

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Running tests with tox on multiple python versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    make tox

-----------
Conventions
-----------

^^^^^^^
Commits
^^^^^^^

Use conventional commits guidelines: https://www.conventionalcommits.org/en/v1.0.0/
