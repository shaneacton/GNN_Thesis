
class Metric:

    def __init__(self, name: str, print_step=False):
        self.print_step = print_step
        self.name = name
        self.values = []
        self.rolling_average = -1
        self.t = 0

    @property
    def alpha(self):
        return 1 - 1/ (len(self.values) + 1)

    @property
    def mean(self):
        return self.total / len(self.values)

    @property
    def total(self):
        return sum(self.values)

    @property
    def last(self):
        return self.values[-1]

    def report(self, value, step=1):
        self.values.append(value)
        if self.rolling_average == -1:
            self.rolling_average = value
        else:
            a = self.alpha
            self.rolling_average = a * self.rolling_average + (1-a) * value
        self.t += step

    def __repr__(self):
        return self.name + " av: " + repr(self.rolling_average) + " total: " + repr(self.total) \
               + (" step: " + repr(self.t) if self.print_step else "")

