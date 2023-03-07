import numpy as np
import pytest
import time

from xrmocap.utils.time_utils import Timer


def test_timer():
    # test Timer
    timer_test = Timer(name='test')

    timer_test.reset()
    low, high = 0, 7
    multi = 0.01

    # reset = True
    for count_num_true in range(low + 1, high - 1):
        for i in range(high):
            timer_test.start()
            time.sleep((i + 1) * multi)
            timer_test.stop()
            if i == count_num_true:
                aver_time = timer_test.get_average(reset=True)
                gt = (i + 2) * (i + 1) * multi / (2 * (i + 1))
                assert (aver_time <= gt * 1.05) & (aver_time >= gt * 0.95)
            if i == high - 1:
                aver_time = timer_test.get_average()
                gt = (count_num_true + 2 +
                      high) * (high - 1 - count_num_true) * multi / (
                          2 * (high - 1 - count_num_true))
                assert (aver_time <= gt * 1.05) & (aver_time >= gt * 0.95)

    timer_test.reset()
    # reset = False
    for count_num_False in range(low + 1, high - 1):
        for i in range(high):
            timer_test.start()
            time.sleep((i + 1) * multi)
            timer_test.stop()
            if i == count_num_False:
                aver_time = timer_test.get_average(reset=False)
                gt = (i + 2) * (i + 1) * multi / (2 * (i + 1))
                assert (aver_time <= gt * 1.05) & (aver_time >= gt * 0.95)
            if i == high - 1:
                aver_time = timer_test.get_average()
                gt = (low + high + 1) * high * multi / (2 * high)
                assert (aver_time <= gt * 1.05) & (aver_time >= gt * 0.95)

    # undo_last_stop "else" test
    timer_test.reset()

    for num_undo_last_stop in range(low, high):
        for i in range(high):
            timer_test.start()
            time.sleep((i + 1) * multi)
            timer_test.stop()
            if i == num_undo_last_stop:
                timer_test.undo_last_stop()
            if i == high - 1:
                aver_time = timer_test.get_average()
                gt = ((low + high + 1) * high * multi / 2 -
                      (num_undo_last_stop + 1) * multi) / (
                          high - 1)
                assert (aver_time <= gt * 1.05) & (aver_time >= gt * 0.95)

    # undo_last_stop "if" test
    timer_test.reset()
    num_undo_last_stop = np.random.randint(low, high)
    for num_undo_last_stop in range(low, high):
        for i in range(high):
            if i == num_undo_last_stop:
                with pytest.raises(ValueError):
                    timer_test.undo_last_stop()
