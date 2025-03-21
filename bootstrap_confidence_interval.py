import numpy


class ValueWithConfidenceInterval:
    def __init__(self, value, lower_bound, upper_bound):
        self.value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


def bootstrap_confidence_interval(y_true, y_pred, func, num_samples=1000, sample_size=None):
    y_true = numpy.array(y_true)
    y_pred = numpy.array(y_pred)
    if sample_size is None:
        sample_size = len(y_true)
    results = []
    for _ in range(num_samples):
        indices = pick_indices(sample_size, y_true)
        results.append(func(y_true[indices], y_pred[indices]))

    lower_bound, upper_bound = numpy.percentile(results, [2.5, 97.5])
    return ValueWithConfidenceInterval(numpy.mean(results), lower_bound, upper_bound)


def pick_indices(sample_size, y_true):
    return numpy.random.choice(len(y_true), sample_size, replace=True)
