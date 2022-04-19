import logging
from typing import Union


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
