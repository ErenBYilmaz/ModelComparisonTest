import unittest
from typing import Type, List

import numpy
from parameterized import parameterized_class
from scipy.stats import binom
from tqdm import tqdm

from hypothesis_test import BootstrapModelComparisonPaired, PermutationModelComparisonPaired, ResamplingBasedModelComparison, BootstrapModelComparisonUnpaired, \
    PermutationModelComparisonUnpaired, HypothesisTest, LikelihoodRatioTestForBinaryModels
from test.test_hypothesis_test import p_value_calibration_overview
from test.test_types import TestTest, BinaryCETest, AsymmetricBinaryCETest, SlightlyAsymmetricBinaryCETest
from utils.tuned_cache import TunedMemory

results_cache = TunedMemory('.cache')


def relevant_test_types():
    test_data_generator_types: List[Type[TestTest]] = [
        BinaryCETest,
        AsymmetricBinaryCETest,
        SlightlyAsymmetricBinaryCETest,
    ]
    hypothesis_test: List[HypothesisTest] = [
        LikelihoodRatioTestForBinaryModels(degrees_of_freedom=1),
        BootstrapModelComparisonPaired(n_iterations=99, metric=BinaryCETest.metric, two_sided=False, verbose=0, skip_validation=True),
        PermutationModelComparisonPaired(n_iterations=99, metric=BinaryCETest.metric, two_sided=False, verbose=0, skip_validation=True),
        BootstrapModelComparisonUnpaired(n_iterations=99, metric=BinaryCETest.metric, two_sided=False, verbose=0, skip_validation=True),
        PermutationModelComparisonUnpaired(n_iterations=99, metric=BinaryCETest.metric, two_sided=False, verbose=0, skip_validation=True),
    ]

    combinations = []
    for test_data_generator_type in test_data_generator_types:
        for hypothesis_test_type in hypothesis_test:
            combinations.append((test_data_generator_type, hypothesis_test_type))

    return combinations


def test_name(cls, idx, i):
    return (
            cls.__name__
            + f'{idx + 1:03d}_'
            + i['cls_test_data_generator'].__name__
            + '_'
            + type(i['hypothesis_test']).__name__
    )


@parameterized_class(['cls_test_data_generator', 'hypothesis_test'], relevant_test_types(), class_name_func=test_name)
class TestAgainstLikelihoodRatioTest(unittest.TestCase):
    cls_test_data_generator: Type[TestTest]
    hypothesis_test: Type[ResamplingBasedModelComparison]

    def run_test(self, y_true: numpy.ndarray, y_pred_1: numpy.ndarray, y_pred_2: numpy.ndarray, y_true_2=None):
        return self.hypothesis_test.compare(y_true, y_pred_1, y_pred_2, y_true_2=y_true_2).p_value

    def test_with_multiple_iterations(self):
        worst_violation, p_values = self.run_repeated_tests()
        median_p_value = numpy.median(p_values)
        assert 0 < median_p_value <= 1
        if self.cls_test_data_generator.null_hypothesis_holds():
            assert worst_violation > 0.001, worst_violation
        else:
            assert worst_violation < 0.05, worst_violation

    def run_repeated_tests(self, n_tests=1000, test_set_size=100):
        p_values = []
        for _ in tqdm(range(n_tests)):
            test = self.cls_test_data_generator(test_set_size)

            p_value = self.run_test(y_true=test.ground_truth(),
                                    y_true_2=test.ground_truth_2(),
                                    y_pred_1=test.model_outputs_1(),
                                    y_pred_2=test.model_outputs_2())

            p_values.append(p_value)
        print()
        print(f'Setting: {type(self.cls_test_data_generator).__name__}')
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
