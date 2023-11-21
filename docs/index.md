# PyOMA2

Software for Structural Health Monitoring with Advanced Signal Processing Techniques

## Installation

```shell
pip install pyOMA2
```

## Project layout

    src/pyoma2/    # The root folder of the library
    algorithm/
        algorithm.py    # This module contains all available algorithm classes
        results.py      # This module contains the classes that model the results of the algorithms
        run_params.py   # This module contains the classes that model the parameters of the algorithms
    functions/
        ...             # all modules listed here implement functions used by the algorithms
    utils/
        ...             # all modules listed here implement utility functions
    main.py             # This module contains the main function of the library
    OMA.py              # This module contains the Single and Multiple Setup classes to run the algorithms
