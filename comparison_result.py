import math

import numpy


class ComparisonResult:
    MIN_COLOR = 0.001

    def __init__(self, p_value: float, model_1_metric_larger: bool):
        self.p_value = p_value
        self.model_1_metric_larger = model_1_metric_larger

    def table_color(self):
        """-1 is blue (model 2 better), 1 is red (model 1 better)"""
        v = min(self.p_value * 2, 1)
        v = math.log(v) / math.log(self.MIN_COLOR)
        if not self.model_1_metric_larger:
            v *= -1
        return v

    @classmethod
    def inverse_color(cls, c):
        c = abs(c)
        return math.exp(c * math.log(cls.MIN_COLOR) - math.log(2))

    @classmethod
    def example_color_ticks(cls):
        return numpy.linspace(-1, 1, 19)

    @classmethod
    def example_color_ticks_labels(cls):
        return [f'{cls.inverse_color(c):.3f}' for c in cls.example_color_ticks()]

    def flipped(self):
        return ComparisonResult(self.p_value, not self.model_1_metric_larger)
