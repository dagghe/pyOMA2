import logging
import os


def configure_logging() -> logging.Logger:
    """Configure logging for the package.

    Returns
    -------
    logging.Logger
        The root logger.
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
    return logger
