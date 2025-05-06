import unittest
from typing import Type, List

import numpy
from parameterized import parameterized_class
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from hypothesis_test import PermutationModelComparisonPaired, ResamplingBasedModelComparison, PermutationModelComparisonUnpaired, HypothesisTest, LikelihoodRatioTestForBinaryModels, \
    BootstrapModelComparisonPaired
from test.test_hypothesis_test import p_value_calibration_overview
from test.test_types import TestTest, BinaryCETest
from utils.tuned_cache import TunedMemory

results_cache = TunedMemory('.cache')


class OutputsOfLogisticRegressionModels(TestTest):
    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size)
        y_true = numpy.random.randint(0, 2, size=test_set_size)

        x_1, x_2 = self.generate_input_columns(test_set_size)
        combined_xs = numpy.concatenate((x_1, x_2), axis=-1)

        self.nested_model = LogisticRegression(penalty=None)
        self.nested_model.fit(x_1, y_true)

        self.super_model = LogisticRegression(penalty=None)
        self.super_model.fit(combined_xs, y_true)

        self.y_pred_1 = self.super_model.predict_proba(combined_xs)[:, 1]
        self.y_pred_2 = self.nested_model.predict_proba(x_1)[:, 1]
        self.y_true = y_true

        self.degree_of_freedom_difference = self.super_model.n_features_in_ - self.nested_model.n_features_in_
        assert self.degree_of_freedom_difference == x_2.shape[-1]
        assert self.metric(self.y_true, self.y_pred_1) >= self.metric(self.y_true, self.y_pred_2)

    def sigmoid(self, z):
        return 1 / (1 + numpy.exp(-z))

    def generate_input_columns(self, test_set_size):
        x_1 = self.sigmoid(numpy.random.normal(size=(test_set_size, 1)))
        x_2 = self.sigmoid(numpy.random.normal(size=(test_set_size, 1)))
        return x_1, x_2

    def ground_truth(self):
        return self.y_true

    def model_outputs_1(self):
        return self.y_pred_1

    def model_outputs_2(self):
        return self.y_pred_2

    @classmethod
    def metric(cls, y_true, y_pred, epsilon=1e-7):
        return numpy.sum(y_true * numpy.log(y_pred + epsilon) + (1 - y_true) * numpy.log(1 - y_pred + epsilon))

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return True


class InputsToLogisticRegressionModels(OutputsOfLogisticRegressionModels):
    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size)
        y_true = numpy.random.randint(0, 2, size=test_set_size)

        x_1, x_2 = self.generate_input_columns(test_set_size)

        self.y_pred_1 = x_1.mean(axis=-1)
        self.y_pred_2 = x_2.mean(axis=-1)
        self.y_true = y_true


class TwoRandomVariables(OutputsOfLogisticRegressionModels):
    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size)
        y_true = numpy.random.randint(0, 2, size=test_set_size)

        x_1, x_2 = self.generate_input_columns(test_set_size)

        self.y_pred_1 = x_2.mean(axis=-1)
        self.y_pred_2 = x_1.mean(axis=-1)
        self.y_true = y_true


class OutputsOfLogisticRegressionModelsOnTestData(OutputsOfLogisticRegressionModels):
    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size)
        y_true = numpy.random.randint(0, 2, size=test_set_size)

        x_1, x_2 = self.generate_input_columns(test_set_size)
        combined_xs = numpy.concatenate((x_1, x_2), axis=-1)

        self.y_pred_1 = self.super_model.predict_proba(combined_xs)[:, 1]
        self.y_pred_2 = self.nested_model.predict_proba(x_1)[:, 1]
        self.y_true = y_true


def relevant_test_types():
    test_data_generator_types: List[Type[OutputsOfLogisticRegressionModels]] = [
        OutputsOfLogisticRegressionModelsOnTestData,
        TwoRandomVariables,
        InputsToLogisticRegressionModels,
        OutputsOfLogisticRegressionModels,
    ]
    hypothesis_tests: List[HypothesisTest] = [
        LikelihoodRatioTestForBinaryModels(degrees_of_freedom=1),
        PermutationModelComparisonPaired(n_iterations=99, metric=BinaryCETest.metric, two_sided=True, verbose=0, skip_validation=True),
        # PermutationModelComparisonUnpaired(n_iterations=99, metric=BinaryCETest.metric, two_sided=True, verbose=0, skip_validation=True),
        PermutationModelComparisonPaired(n_iterations=99, metric=BinaryCETest.metric, two_sided=False, verbose=0, skip_validation=True),
        # PermutationModelComparisonUnpaired(n_iterations=99, metric=BinaryCETest.metric, two_sided=False, verbose=0, skip_validation=True),
        # BootstrapModelComparisonPaired(n_iterations=99, metric=BinaryCETest.metric, two_sided=True, verbose=0, skip_validation=True),
        # BootstrapModelComparisonUnpaired(n_iterations=99, metric=BinaryCETest.metric, two_sided=True, verbose=0, skip_validation=True),
        BootstrapModelComparisonPaired(n_iterations=99, metric=BinaryCETest.metric, two_sided=False, verbose=0, skip_validation=True),
        # BootstrapModelComparisonUnpaired(n_iterations=99, metric=BinaryCETest.metric, two_sided=False, verbose=0, skip_validation=True),
    ]

    combinations = []
    for test_data_generator_type in test_data_generator_types:
        for hypothesis_test in hypothesis_tests:
            estimate_dof = test_data_generator_type(100).degree_of_freedom_difference
            if isinstance(hypothesis_test, LikelihoodRatioTestForBinaryModels) and hypothesis_test.degrees_of_freedom != estimate_dof:
                continue
            combinations.append((test_data_generator_type, hypothesis_test))

    return combinations


def test_name(cls, idx, i):
    test = i['hypothesis_test']
    return (
            cls.__name__
            + f'{idx + 1:03d}_'
            + i['cls_test_data_generator'].__name__
            + '_'
            + type(test).__name__
            + ('_2sided' if isinstance(test, ResamplingBasedModelComparison) and test.two_sided else '_1sided')
    )


# @skip('This does not work yet...')
@parameterized_class(['cls_test_data_generator', 'hypothesis_test'], relevant_test_types(), class_name_func=test_name)
class TestAgainstLRTest(unittest.TestCase):
    cls_test_data_generator: Type[TestTest]
    hypothesis_test: Type[HypothesisTest]

    def run_test(self, y_true: numpy.ndarray, y_pred_1: numpy.ndarray, y_pred_2: numpy.ndarray, y_true_2=None):
        return self.hypothesis_test.compare(y_true, y_pred_1, y_pred_2, y_true_2=y_true_2).p_value

    def test_with_multiple_iterations(self):
        try:
            worst_violation, p_values = self.run_repeated_tests()
        except ValueError as e:
            if 'Log likelihood of model 1 is smaller than log likelihood of model 2' in str(e):
                return self.skipTest('Skipping test because of log likelihood error')
            raise
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
