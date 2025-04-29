import unittest

import numpy

from bootstrap_confidence_interval import bootstrap_confidence_interval, ValueWithConfidenceInterval


def accuracy(y_true, y_pred):
    return numpy.mean(y_true == y_pred)


class TestConfidenceIntervalTest(unittest.TestCase):
    def test_outputs_value_with_confidence_interval(self):
        result = bootstrap_confidence_interval([0, 0, 0, 1, 1], [0.3, 0.6, 0.2, 0.7, 0.8], accuracy)
        assert isinstance(result, ValueWithConfidenceInterval)

    def test_lower_bound_below_upper_bound(self):
        result = bootstrap_confidence_interval([0, 0, 0, 1, 1], [0.3, 0.6, 0.2, 0.7, 0.8], accuracy)
        assert result.lower_bound <= result.upper_bound