import numpy as np
from time import time


class Profiler(object):

    def __init__(self, name=""):
        self.name = name
        self.start = []
        self.end = []

    def timeit(self, f, num_iter, *args, **kwargs):
        for _ in range(num_iter):
            self.tick()
            ret = f(*args, **kwargs)
            self.tock()
        return ret

    def tick(self):
        self.start.append(time())

    def tock(self):
        self.end.append(time())

    def del_last_tick(self):
        self.start = self.start[:-1]

    def trim_tick_tocks(self):
        tick_len, tock_len = len(self.start), len(self.end)
        if tick_len > tock_len:
            self.start = self.start[:tock_len]
        elif tick_len < tock_len:
            self.end = self.end[:tick_len]

    def sum(self):
        assert len(self.start) == len(self.end)
        return (np.array(self.end) - np.array(self.start)).sum()

    def mean(self):
        assert len(self.start) == len(self.end)
        return (np.array(self.end) - np.array(self.start)).mean()

    def median(self):
        assert len(self.start) == len(self.end)
        return np.median(np.array(self.end) - np.array(self.start))

    def min(self):
        assert len(self.start) == len(self.end)
        return (np.array(self.end) - np.array(self.start)).min()

    def max(self):
        assert len(self.start) == len(self.end)
        return (np.array(self.end) - np.array(self.start)).max()

    def __insert_middle_substr(self, size=50):
        buf_str, main_char = '\0', '#'
        substring = ('{:^' + str(size) + '}').format(buf_str + self.name.replace(' ', buf_str) + buf_str)
        return substring.replace(' ', main_char).replace(buf_str, ' ')

    def __str__(self):
        name = self.__insert_middle_substr(size=50)
        if not self.start or not self.end:
            return '\n'.join([name, "No tick-tocks!"])
        sum = "sum:    {:.6f}".format(self.sum())
        mean = "mean:   {:.6f}".format(self.mean())
        median = "median: {:.6f}".format(self.median())
        min = "min:    {:.6f}".format(self.min())
        max = "max:    {:.6f}".format(self.max())
        return '\n'.join([name, sum, mean, median, min, max])
