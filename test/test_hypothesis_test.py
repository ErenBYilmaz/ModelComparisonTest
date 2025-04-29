import unittest
from typing import Type

import numpy
import parameterized
from parameterized import parameterized_class
from scipy.stats import binom
from tqdm import tqdm

from hypothesis_test import BootstrapModelComparison
from test.test_types import TestTest, make_unpaired_test, CIndexTest, AsymmetricCIndexTest, AvgLogLikelihoodTest, AsymmetricAverageLikelihoodTest, AccuracyTest, BinaryCETest, AsymmetricBinaryCETest, \
    SlightlyAsymmetricBinaryCETest, MSETest, AsymmetricMSETest, SlightlyAsymmetricMSETest, AsymmetricAccuracyTest, SlightlyAsymmetricAccuracyTest, SameModelAccuracyTest, LogLikelihoodTest
from utils.tuned_cache import TunedMemory

results_cache = TunedMemory('.cache')


def p_value_calibration_overview(p_values):
    values, counts = (numpy.unique(p_values, return_counts=True)[0], numpy.cumsum(numpy.unique(p_values, return_counts=True)[1] / len(p_values)))
    value_counts = {k: v for k, v in zip(values, counts)}
    p_values = {}
    for value, count in value_counts.items():
        p_values[value] = binom.cdf(count, len(p_values), value)
    return value_counts


def relevant_test_types():
    test_types = [
        BinaryCETest, AsymmetricBinaryCETest, SlightlyAsymmetricBinaryCETest, MSETest, AsymmetricMSETest, SlightlyAsymmetricMSETest, AccuracyTest, SameModelAccuracyTest, AsymmetricAccuracyTest,
        SlightlyAsymmetricAccuracyTest, CIndexTest, AsymmetricCIndexTest, AsymmetricAverageLikelihoodTest, AvgLogLikelihoodTest
    ]

    combinations = []
    for test_type in test_types:
        for permutation_only in [True]:
            for two_sided in [True, False]:
                for paired in [True, False]:
                    for skip_permutation in [False]:
                        if skip_permutation and (permutation_only or two_sided):
                            continue
                        combinations.append((test_type, permutation_only, two_sided, paired, skip_permutation))
                        if not paired and 'Same' not in test_type.__name__:
                            combinations.append((make_unpaired_test(test_type, 1, name_suffix='SameSize'), permutation_only, two_sided, paired, skip_permutation))
                            combinations.append((make_unpaired_test(test_type, 0.5, name_suffix='HalfSize'), permutation_only, two_sided, paired, skip_permutation))
                            combinations.append((make_unpaired_test(test_type, 2, name_suffix='DoubleSize'), permutation_only, two_sided, paired, skip_permutation))

    return combinations


class TestTestWithFixedParameters(unittest.TestCase):

    def test_some_example_computation(self):
        y_true = numpy.array([0, 0, 0, 1, 1, 1, 1])
        y_pred_1 = numpy.array([0.2, 0.2, 0.3, 0.52, 0.6, 0.7, 0.15])
        y_pred_2 = numpy.array([0.25, 0.3, 0.6, 0.4, 0.45, 0.6, 0.1])
        accuracy = lambda y_true, y_pred: numpy.mean((y_pred >= 0.5) == y_true).item()
        p_value = BootstrapModelComparison(999, accuracy).compare(y_true, y_pred_1, y_pred_2).p_value
        print(p_value)
        assert isinstance(p_value, float), p_value
        assert p_value < 0.55, p_value

    def test_accuracy_comparison_without_permutation(self):
        p_values_without = []
        p_values_with_permutation = []
        test_set_size = 100
        n_iterations = 999
        n_tests = 1000
        test_cls = LogLikelihoodTest
        bootstrap_only_test = BootstrapModelComparison(n_iterations=n_iterations, metric=test_cls.metric, permutation_only=False, skip_permutation=True, two_sided=False, skip_validation=True)
        permutation_only_test = BootstrapModelComparison(n_iterations=n_iterations, metric=test_cls.metric, permutation_only=True, skip_permutation=False, two_sided=False, skip_validation=True)
        for _ in tqdm(range(n_tests)):
            test = test_cls(test_set_size=test_set_size)
            p_value = bootstrap_only_test.compare(
                y_true=test.y_true,
                y_pred_1=test.model_outputs_1(),
                y_pred_2=test.model_outputs_2()
            ).p_value

            p_value_with_permutation = permutation_only_test.compare(
                y_true=test.y_true,
                y_pred_1=test.model_outputs_1(),
                y_pred_2=test.model_outputs_2()
            ).p_value

            p_values_without.append(p_value)
            p_values_with_permutation.append(p_value_with_permutation)
        for name, p_values in {'p_values_without': p_values_without, 'p_values_with_permutation': p_values_with_permutation}.items():
            print()
            print('#', name)
            print(f'n_iterations: {n_iterations}')
            print(f'n_tests: {n_tests}')
            print(f'test_set_size: {test_set_size}')
            print(f'median p-value: {numpy.median(p_values)}')
            cumulative_ratios = p_value_calibration_overview(p_values)

            # specifically check the 5 % threshold
            p_threshold = max([k for k in cumulative_ratios.keys() if k < 0.05])
            ratio = cumulative_ratios[p_threshold]
            violation_p_value = 1 - binom.cdf(ratio * n_tests - 1, n_tests, 0.05)
            # probability of having at least that many p-values below threshold if the test was well-calibrated
            print('Violation at 0.05 was', violation_p_value)

            print(f'ratio_p_values_at_most_{p_threshold}: {ratio}')

            violation_p_value = 1 - binom.cdf(ratio * n_tests - 1, n_tests, p_threshold)
            print(f'Violation at {p_threshold} was', violation_p_value)

        differences = numpy.array(p_values_with_permutation) - numpy.array(p_values_without)
        print('mean difference:', numpy.mean(differences), 'std:', numpy.std(differences))
        # t-test if difference is significantly different from 0
        from scipy import stats
        result = stats.ttest_1samp(differences, 0).pvalue
        print('t-test p-value for different p-values:', result)
        print('p-values were typically larger', 'with' if numpy.mean(differences) > 0 else 'without', 'permutation.')


def test_name(cls, idx, i):
    return (
            cls.__name__
            + f'{idx + 1:03d}_'
            + i['cls_test_data_generator'].__name__
            + ('_permutation_only' if i['permutation_only'] else 'skip_permutation' if i['skip_permutation'] else '_bootstrap')
            + ('_two_sided' if i['two_sided'] else '_one_sided')
            + ('_paired' if i['paired'] else '_unpaired')
    )


@parameterized_class(['cls_test_data_generator', 'permutation_only', 'two_sided', 'paired', 'skip_permutation'], relevant_test_types(), class_name_func=test_name)
class TestBootStrapTest(unittest.TestCase):
    cls_test_data_generator: Type[TestTest]
    permutation_only: bool
    two_sided: bool
    paired: bool
    skip_permutation: bool

    def bootstrap_test(self, n_bootstraps, y_true: numpy.ndarray, y_pred_1: numpy.ndarray, y_pred_2: numpy.ndarray, metric, y_true_2=None, verbose=0):
        return BootstrapModelComparison(
            n_iterations=n_bootstraps,
            metric=metric,
            two_sided=self.two_sided,
            verbose=verbose,
            paired=self.paired,
            permutation_only=self.permutation_only,
            skip_permutation=self.skip_permutation
        ).compare(y_true, y_pred_1, y_pred_2, y_true_2=y_true_2).p_value

    def test_some_example_computation(self):
        if self.paired and self.permutation_only and self.two_sided:
            # This test actually fails if we skip the bootstrap and only permute, because there is only one differently classified sample and that difference cant change by permutation
            return

        y_true = numpy.array([0, 0, 0, 1, 1, 1, 1])
        y_pred_1 = numpy.array([0.2, 0.2, 0.3, 0.52, 0.6, 0.7, 0.15])
        y_pred_2 = numpy.array([0.25, 0.3, 0.6, 0.4, 0.45, 0.6, 0.1])
        accuracy = lambda y_true, y_pred: numpy.mean((y_pred >= 0.5) == y_true)
        p_value = self.bootstrap_test(999, y_true, y_pred_1, y_pred_2, accuracy)
        print(p_value)
        assert isinstance(p_value, float), p_value
        assert p_value < 0.55, p_value

    def test_more_thorough_simulation(self):
        test = AccuracyTest(test_set_size=100)

        p_value = self.bootstrap_test(n_bootstraps=199,
                                      y_true=test.y_true,
                                      y_pred_1=test.model_outputs_1(),
                                      y_pred_2=test.model_outputs_2(),
                                      metric=test.metric,
                                      verbose=2)

        assert isinstance(p_value, float)
        assert p_value > 0.0001

    def test_c_index_comparison(self):
        test = CIndexTest(test_set_size=100)

        p_value = self.bootstrap_test(n_bootstraps=199,
                                      y_true=test.y_true,
                                      y_pred_1=test.model_outputs_1(),
                                      y_pred_2=test.model_outputs_2(),
                                      metric=test.metric,
                                      verbose=2)

        assert isinstance(p_value, float)
        assert p_value > 0.0001

    def test_avg_log_likelihood_comparison(self):
        test = AvgLogLikelihoodTest(test_set_size=100)

        p_value = self.bootstrap_test(n_bootstraps=199,
                                      y_true=test.y_true,
                                      y_pred_1=test.model_outputs_1(),
                                      y_pred_2=test.model_outputs_2(),
                                      metric=test.metric,
                                      verbose=2)

        assert isinstance(p_value, float)
        assert p_value > 0.0001

    @parameterized.parameterized.expand([
        (2,),
        (9,),
        (49,),
    ], name_func=lambda f, i, p: f'{f.__name__}_{p[0][0]:02d}')
    def test_with_multiple_bootstraps(self, n_bootstraps):
        worst_violation, p_values = self.run_repeated_permutation_tests(n_bootstraps)
        median_p_value = numpy.median(p_values)
        assert 0 < median_p_value <= 1
        if self.cls_test_data_generator.null_hypothesis_holds():
            assert worst_violation > 0.001, worst_violation
        else:
            assert worst_violation < 0.05, worst_violation

    def run_repeated_permutation_tests(self, n_bootstraps, n_tests=1000, test_set_size=100):
        print(f'n_bootstraps is relatively low at {n_bootstraps}. This function will only return multiples of 1 / (n_bootstraps + 1) = {1 / (n_bootstraps + 1):.3g} '
              f'so it you can use those as significance level α and then check p <= α or p < α + {1 / (n_bootstraps + 1):.3g} instead of p < α')
        p_values = []
        for _ in tqdm(range(n_tests)):
            test = self.cls_test_data_generator(test_set_size)

            p_value = self.bootstrap_test(n_bootstraps,
                                          y_true=test.ground_truth(),
                                          y_true_2=test.ground_truth_2(),
                                          y_pred_1=test.model_outputs_1(),
                                          y_pred_2=test.model_outputs_2(),
                                          metric=test.metric)

            p_values.append(p_value)
        print()
        print(f'n_bootstraps: {n_bootstraps}')
        print(f'n_tests: {n_tests}')
        print(f'test_set_size: {test_set_size}')
        print(f'median p-value: {numpy.median(p_values)}')
        cumulative_ratios = p_value_calibration_overview(p_values)
        worst_violation = 1
        for p_threshold, ratio in cumulative_ratios.items():
            print(f'ratio_p_values_at_most_{p_threshold}: {ratio}')
            if p_threshold > 0.5:
                continue

            # I expect the p value to be below 0.05 in 5% roughly 5 % of the cases, following a binomial distribution (assuming the null hypothesis is true)
            # otherwise it should be below 0.05 more often
            # this is the probability of getting at least that many p-values below threshold if the test was well-calibrated and the null hypothesis holds
            probability_of_having_less_positives = binom.cdf(ratio * n_tests - 1, n_tests, p_threshold)
            probability_if_well_calibrated = 1 - probability_of_having_less_positives
            worst_violation = min(worst_violation, probability_if_well_calibrated)
        print('Worst violation was', worst_violation)
        return worst_violation, p_values
