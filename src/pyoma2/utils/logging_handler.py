"""
Logging handler for the pyOMA2 module.
"""
import logging
import os


def configure_logging() -> logging.Logger:
    """
    Configures and initializes logging for the pyOMA2 package.

    The function sets up a root logger specifically for the package with a logging level determined by
    an environment variable. It also configures a console handler to output log messages with a specific
    format, including timestamp, logger name, log level, message, module, and line number. Optionally,
    this function can disable logging from the 'matplotlib' library based on an environment variable.

    Environment Variables
    ---------------------
    PYOMA_LOG_LEVEL : str, optional
        Defines the logging level for the pyOMA2 logger. Acceptable values include 'DEBUG', 'INFO', 'WARNING',
        'ERROR', and 'CRITICAL'. Defaults to 'INFO' if not specified.
    PYOMA_DISABLE_MATPLOTLIB_LOGGING : str, optional
        If set to 'True' or '1', disables logging from the 'matplotlib' library. Defaults to 'True'.

    Returns
    -------
    logging.Logger
        The configured root logger for the pyOMA2 package.

    Notes
    -----
    - The logger's name is set to 'pyoma2'.
    - The logger outputs to the console.
    - The log format includes the timestamp, logger name, log level, message, and the module and line number
      where the log was generated.
    """
    # Read logging level from environment variable with a default of INFO
    log_level = os.getenv("PYOMA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    # Create a root logger
    logger = logging.getLogger(name="pyoma2")
    logger.setLevel(level)

    # Create a console handler and set its level
    ch = logging.StreamHandler()
    ch.setLevel(level)  # Set level from environment variable

    # log also module and line number and level
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(module)s:%(lineno)d)"
    )
    ch.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(ch)

    # disable logging from matplotlib
    if os.getenv("PYOMA_DISABLE_MATPLOTLIB_LOGGING", "True") in ["True", "true", "1"]:
        logging.getLogger("matplotlib").setLevel(logging.CRITICAL + 1)
    return logger
