import time
from xrprimer.utils.log_utils import get_logger, logging


class Timer:

    def __init__(self, name: str, logger: logging.Logger = None) -> None:
        """Timer initialization.

        Args:
            name (str): id information of this class
            logger (logging.Logger, optional): Save information for
            data recording and error reporting
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
            ValueError: start_time has not been assigned
        """
        if self.start_time is None:
            self.logger.error(f'Timer {self.name} has not been started.',
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
            reset (bool, optional): whether to reset Timer.
            Defaults to True.

        Returns:
            float: average time consumption for process already finished
        """
        avg_time = self.total_time / self.count

        if reset:
            self.reset()
        return avg_time

    def reset(self) -> None:
        """Timer initialization This function could be used to restared Timer
        in get_average function.

        count: number of process already finished.
        total_time: time consumption of process already finished.
        start_time: the specific time when one process starts.
        end_time:the specific time when one process finishses.
        last_measured: time consumption of running last process.
        """
        self.count = 0.0
        self.total_time = 0.0
        self.start_time = None
        self.end_time = None
        self.last_measure = None

    def undo_last_stop(self) -> None:
        """Go back to data which records data before this process starts. This
        function could be used when data for this process is not well or this
        process is interrupted Unexpectedly.

        Raises:
            ValueError: Only Timer has started but not ended yet
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
