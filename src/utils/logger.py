import logging
import sys


def setup_logger() -> logging.Logger:
    """Configure and return a console logger for the project."""
    logger = logging.getLogger("type_correction")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger   

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
