import unittest

import numpy
from scipy.stats import binom
from test.test_hypothesis_test import AccuracyTest


class SanityCheckSimulation(unittest.TestCase):
    def test_bootstrap_distribution_shift(self):
        """
        This test shows that the distribution of the difference of two metrics is shifted towards zero after bootstrapping.
        That means that p-values are automatically miscalibrated if bootstrap is added into the mix
        """
        num_samples = 200
        n_bootstraps = 10000
        metric = AccuracyTest(test_set_size=10).metric
        y_true = numpy.ones(num_samples)
        inner_values = []
        outer_values = []
        larger_values = []

        for _ in range(n_bootstraps):
            samples_1 = numpy.random.randint(0, 2, size=num_samples)
            samples_2 = numpy.random.randint(0, 2, size=num_samples)
            difference_before_bootstrap = abs(metric(y_true, samples_1) - metric(y_true, samples_2))
            outer_values.append(difference_before_bootstrap)
            combined_sample = numpy.concatenate([samples_1, samples_2])
            bootstrap_indices = numpy.random.choice(numpy.arange(len(combined_sample)), size=num_samples, replace=True)
            bootstrap_sample_1 = combined_sample[bootstrap_indices]
            bootstrap_sample_2 = combined_sample[(bootstrap_indices + num_samples) % (2 * num_samples)]
            difference_after_bootstrap = abs(metric(y_true, bootstrap_sample_1) - metric(y_true, bootstrap_sample_2))
            inner_values.append(difference_after_bootstrap)
            larger_values.append(difference_before_bootstrap > difference_after_bootstrap)

        print(numpy.mean(outer_values), numpy.std(outer_values))
        print(numpy.mean(inner_values), numpy.std(inner_values))
        print(numpy.mean(larger_values))

        violation_p_value = binom.cdf(numpy.sum(larger_values), n_bootstraps, 0.5)
        print(violation_p_value)
        assert violation_p_value < 0.05

    def test_bootstrap_distribution_shift_without_permutation(self):
        """
        This test shows that the distribution of the difference of two metrics is shifted towards zero after bootstrapping.
        That means that p-values are automatically miscalibrated if bootstrap is added into the mix
        """
        num_samples = 200
        n_bootstraps = 10000
        metric = AccuracyTest(test_set_size=10).metric
        y_true = numpy.ones(num_samples)
        inner_values = []
        outer_values = []
        significant_values = []

        for _ in range(n_bootstraps):
            samples_1 = numpy.random.randint(0, 2, size=num_samples)
            samples_2 = numpy.random.randint(0, 2, size=num_samples)
            difference_before_bootstrap = metric(y_true, samples_1) - metric(y_true, samples_2)
            outer_values.append(difference_before_bootstrap)
            bootstrap_indices = numpy.random.choice(numpy.arange(len(samples_1)), size=num_samples, replace=True)
            bootstrap_sample_1 = samples_1[bootstrap_indices]
            bootstrap_indices = numpy.random.choice(numpy.arange(len(samples_2)), size=num_samples, replace=True)
            bootstrap_sample_2 = samples_2[bootstrap_indices]
            difference_after_bootstrap = metric(y_true, bootstrap_sample_1) - metric(y_true, bootstrap_sample_2)
            if difference_before_bootstrap < 0:
                difference_after_bootstrap = -difference_after_bootstrap
            inner_values.append(difference_after_bootstrap)
            significant_values.append(difference_after_bootstrap < 0)

        print(numpy.mean(outer_values), numpy.std(outer_values))
        print(numpy.mean(inner_values), numpy.std(inner_values))
        print(numpy.mean(significant_values))

        violation_p_value = binom.cdf(numpy.sum(significant_values), n_bootstraps, 0.5)
        print(violation_p_value)
        assert violation_p_value < 0.05
