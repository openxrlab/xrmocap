import time
from typing import Union
from xrprimer.utils.log_utils import get_logger, logging


class Timer:

    def __init__(self,
                 name: str,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Initialize Timer.

        Args:
            name (str): Id information of this class
            logger (Union[None, str, logging.Logger], optional):
                Defaults to None.
        """
        self.name = name
        self.logger = get_logger(logger)
        self.reset()

    def start(self) -> None:
        """Timer start and save current time."""
        self.start_time = time.time()
        self.end_time = None

    def stop(self) -> None:
        """Timer stop and save current time to calculate and save data of one
        forward process.

        Raises:
            ValueError: You have to start the timer first before stopping it
        """
        if self.start_time is None:
            self.logger.error(f'Timer {self.name} has not been started.' +
                              ' Please call start() before stop().')
            raise ValueError
        self.end_time = time.time()
        self.last_measure = self.end_time - self.start_time
        self.total_time += self.last_measure
        self.count += 1
        self.start_time = None

    def get_average(self, reset: bool = True) -> float:
        """Get average time consumption when some process finish. Determine
        whether to reset Timer according to reset.

        Args:
            reset (bool, optional): Whether to reset Timer. If true,
                records before will not impact the next average value.
                Defaults to True.

        Returns:
            float: Average time consumption for process already finished
        """
        avg_time = self.total_time / self.count

        if reset:
            self.reset()
        return avg_time

    def reset(self) -> None:
        """Restart Timer in get_average function."""
        self.count = 0.0
        self.total_time = 0.0
        self.start_time = None
        self.end_time = None
        self.last_measure = None

    def undo_last_stop(self) -> None:
        """Reverse the previous actions you have taken during this progress
        and restore the data to its previous state. Useful for occasions below:
        1. data for current process is not reliable
        2. current process crashes Unexpectedly.

        Raises:
            ValueError: You have to stop the timer first before restarting it
        """
        if self.end_time is None:
            self.start_time = None
            self.logger.error('Data for last stop is off-record.' +
                              ' Undo cannot be done.')
            raise ValueError
        else:
            self.total_time -= self.last_measure
            self.count -= 1
            self.last_measure = None
