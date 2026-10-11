"""Sliding-window construction with explicit legacy tail exclusion."""
import operator


def iter_window_samples(data_x, data_y, window_size, reserved_tail=0):
    window_size = operator.index(window_size)
    reserved_tail = operator.index(reserved_tail)
    if window_size <= 0 or reserved_tail < 0:
        raise ValueError("window_size must be positive and reserved_tail nonnegative")
    if len(data_x) != len(data_y):
        raise ValueError("inputs and targets must contain the same number of rows")
    stop = len(data_x) - window_size - reserved_tail
    if stop <= 0:
        raise ValueError("not enough rows for the requested window and reserved tail")
    for index in range(stop):
        yield data_x[index:index + window_size], data_y[index + window_size]


def create_sliding_window(data_x, data_y, window_size, reserved_tail=0):
    from torch import stack
    X, y = [], []
    for inputs, target in iter_window_samples(data_x, data_y, window_size, reserved_tail):
        X.append(inputs)
        y.append(target)
    return stack(X), stack(y)
