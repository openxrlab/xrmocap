import logging
from typing import Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def setup_logger(logger_name: str = 'root',
                 logger_level: int = logging.INFO,
                 logger_path: str = None,
                 logger_format: str = None) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logger_level)
    handlers = [logging.StreamHandler()]
    if logger_path is not None:
        handler = logging.FileHandler(logger_path)
        handlers.append(handler)
    if logger_format is not None:
        formatter = logging.Formatter(logger_format)
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_logger(
        logger: Union[None, str, logging.Logger] = None) -> logging.Logger:
    """Get logger.

    Args:
        logger (Union[None, str, logging.Logger]):
            None for root logger. Besides, pass name of the
            logger or the logger itself.
            Defaults to None.

    Returns:
        logging.Logger
    """
    if logger is None or isinstance(logger, str):
        ret_logger = logging.getLogger(logger)
    else:
        ret_logger = logger
    return ret_logger
