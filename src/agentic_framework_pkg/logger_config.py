import logging
import sys

DEFAULT_LOG_LEVEL = logging.DEBUG
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(logger_name: str, level: int = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """
    Creates and configures a logger instance.

    Args:
        logger_name: The name for the logger (typically __name__).
        level: The logging level.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Add console handler if no handlers are configured
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
