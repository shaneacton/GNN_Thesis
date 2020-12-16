import copy
import math


class Metric:

    def __init__(self, name: str, print_step=False, print_total=False, max_alpha=0.9995, beta=1):
        self.beta = beta
        self.max_alpha = max_alpha
        self.print_total = print_total
        self.print_step = print_step
        self.name = name
        self.values = []
        self.rolling_average = -1
        self.t = 0

    @property
    def alpha(self):
        return min(1 - 1/ (len(self.values) * self.beta + 1), self.max_alpha)

    @property
    def mean(self):
        """the average of the values since the last mean flash"""
        return sum(self.values[-self.t:]) / self.t

    def flash_mean(self):
        self.t = 0

    @property
    def total(self):
        return sum(self.values)

    @property
    def last(self):
        return self.values[-1]

    def __add__(self, other):
        clone = copy.deepcopy(self)
        if isinstance(other, int) or isinstance(other, float):
            clone.rolling_average += other
            for i in range(len(clone.values)):
                clone.values[i] += other
                return clone
        if isinstance(other, Metric):
            return self + other.rolling_average

        raise Exception()

    def __sub__(self, other):
        return self + (other * -1)

    def __mul__(self, other):
        clone = copy.deepcopy(self)
        clone.rolling_average *= other
        for i in range(len(clone.values)):
            clone.values[i] *= other
        return clone

    def report(self, value, step=1):
        if math.isnan(value):
            print("received", value, "val on", self.name, "metric")
            return 
        self.values.append(value)
        if self.rolling_average == -1:
            self.rolling_average = value
        else:
            a = self.alpha
            self.rolling_average = a * self.rolling_average + (1-a) * value
        self.t += step

    def __repr__(self):
        return self.name + " av: " + repr(self.rolling_average) \
                + (" total: " + repr(self.total) if self.print_total else "") \
                + (" step: " + repr(self.t) if self.print_step else "")

