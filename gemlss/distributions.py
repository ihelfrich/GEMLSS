"""Basic probability distributions used by GEMLSS."""

import math


class Normal:
    """Normal distribution parameterized by mean and standard deviation."""

    def __init__(self, mean: float = 0.0, sd: float = 1.0):
        self.mean = mean
        self.sd = sd

    def logpdf(self, x: float) -> float:
        var = self.sd ** 2
        return -0.5 * math.log(2 * math.pi * var) - ((x - self.mean) ** 2) / (2 * var)

    def pdf(self, x: float) -> float:
        return math.exp(self.logpdf(x))
