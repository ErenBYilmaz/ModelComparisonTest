import math
import unittest
from typing import Type

import numpy
import pandas
import parameterized
from parameterized import parameterized_class
from scipy.stats import binom
from tqdm import tqdm

from hypothesis_test import BootstrapModelComparison
from utils.tuned_cache import TunedMemory

results_cache = TunedMemory('.cache')


class TestTest:
    def __init__(self, test_set_size: int):
        self.test_set_size = test_set_size

    def ground_truth(self):
        raise NotImplementedError('Abstract method')

    def ground_truth_2(self):
        return self.ground_truth()

    def unpaired(self):
        return False

    def model_outputs_1(self):
        raise NotImplementedError('Abstract method')

    def model_outputs_2(self):
        raise NotImplementedError('Abstract method')

    def metric(self, y_true: numpy.ndarray, y_pred: numpy.ndarray) -> float:
        raise NotImplementedError('Abstract method')

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        raise NotImplementedError('Abstract method')


def make_unpaired_test(t: Type[TestTest], test_set_size_ratio: float, name_suffix: str = None) -> Type[TestTest]:
    class UnpairedTest(TestTest):
        def __init__(self, test_set_size: int):
            super().__init__(test_set_size=test_set_size)
            self.base_1 = t(test_set_size)
            self.base_2 = t(round(test_set_size * test_set_size_ratio))

        def ground_truth(self):
            return self.base_1.ground_truth()

        def ground_truth_2(self):
            return self.base_2.ground_truth()

        def model_outputs_1(self):
            return self.base_1.model_outputs_1()

        def model_outputs_2(self):
            return self.base_2.model_outputs_2()

        def unpaired(self):
            return True

        def metric(self, y_true, y_pred):
            return self.base_1.metric(y_true, y_pred)

        @classmethod
        def null_hypothesis_holds(cls) -> bool:
            return t.null_hypothesis_holds()

    name = f'Unpaired{t.__name__}'
    if name_suffix:
        name += name_suffix
    UnpairedTest.__name__ = name

    return UnpairedTest


class CIndexTest(TestTest):
    def __init__(self, test_set_size: int):
        durations = numpy.random.random(test_set_size)
        event_observed = (numpy.random.random(test_set_size) < 0.2).astype(int)
        self.y_true = numpy.stack([durations, event_observed], axis=1)

        self.y_pred_1 = numpy.random.random(test_set_size) + 0.5 * durations + 0.3 * event_observed
        self.y_pred_2 = numpy.random.random(test_set_size) + 0.5 * durations + 0.3 * event_observed
        self.y_pred_2[:20] = self.y_pred_1[:20]

        super().__init__(test_set_size=test_set_size)

    def ground_truth(self):
        return self.y_true

    def model_outputs_1(self):
        return self.y_pred_1

    def model_outputs_2(self):
        return self.y_pred_2

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return True

    def metric(self, y_true, y_pred):
        from lifelines.utils import concordance_index
        try:
            return concordance_index(y_true[:, 0], -y_pred, y_true[:, 1])
        except ZeroDivisionError as e:
            if 'No admissable pairs in the dataset.' in str(e):
                return 0.5
            raise e


class AsymmetricCIndexTest(CIndexTest):
    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return False

    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size)
        durations = numpy.random.random(test_set_size)
        event_observed = (numpy.random.random(test_set_size) < 0.4).astype(int)
        self.y_true = numpy.stack([durations, event_observed], axis=1)

        self.y_pred_1 = numpy.random.random(test_set_size) - 0.5 * durations + 1 * event_observed
        self.y_pred_2 = numpy.random.random(test_set_size) - 0.1 * durations + 0.1 * event_observed
        self.y_pred_2[:test_set_size // 5] = self.y_pred_1[:test_set_size // 5]
        self.metric_1 = self.metric(self.y_true, self.y_pred_1)
        self.metric_2 = self.metric(self.y_true, self.y_pred_2)


class SlightlyAsymmetricCIndexTest(AsymmetricCIndexTest):
    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size)
        durations = numpy.random.random(test_set_size)
        event_observed = (numpy.random.random(test_set_size) < 0.4).astype(int)
        self.y_true = numpy.stack([durations, event_observed], axis=1)

        self.y_pred_1 = numpy.random.random(test_set_size) - 0.4 * durations + 0.6 * event_observed
        self.y_pred_2 = numpy.random.random(test_set_size) - 0.2 * durations + 0.3 * event_observed
        self.y_pred_2[:test_set_size // 5] = self.y_pred_1[:test_set_size // 5]
        self.metric_1 = self.metric(self.y_true, self.y_pred_1)
        self.metric_2 = self.metric(self.y_true, self.y_pred_2)


def avg_likelihood(y_true, y_pred):
    """
    Copied and modified from SemiParametricPHFitter._get_efron_values_single
    y_pred should be (x * beta), i.e. without exponentiation
    """
    from numpy import log, arange
    order = numpy.lexsort((y_true[:, 1], y_true[:, 0]))
    y_true = y_true[order]
    y_pred = y_pred[order]

    T = y_true[:, 0]
    E = y_true[:, 1]
    n = y_true.shape[0]

    log_lik = 0

    # Init risk and tie sums to zero
    y_pred_sum = 0
    risk_phi, tie_phi = 0, 0

    # Init number of ties and weights
    weight_count = 0.0
    tied_death_counts = 0
    scores = numpy.exp(y_pred)

    # Iterate backwards to utilize recursive relationship
    for i in range(n - 1, -1, -1):
        # Doing it like this to preserve shape
        ti = T[i]
        ei = E[i]
        phi_i = scores[i]

        # Calculate sums of Risk set
        risk_phi = risk_phi + phi_i

        # Calculate sums of Ties, if this is an event
        if ei:
            y_pred_sum = y_pred_sum + y_pred[i]
            tie_phi = tie_phi + phi_i

            # Keep track of count
            tied_death_counts += 1
            weight_count += 1

        if i > 0 and T[i - 1] == ti:
            # There are more ties/members of the risk set
            continue
        elif tied_death_counts == 0:
            # Only censored with current time, move on
            continue

        # There was at least one event and no more ties remain. Time to sum.
        # This code is near identical to the _batch algorithm below. In fact, see _batch for comments.
        weighted_average = weight_count / tied_death_counts

        if tied_death_counts > 1:
            increasing_proportion = arange(tied_death_counts) / tied_death_counts
            denominator = (risk_phi - increasing_proportion * tie_phi)
        else:
            denominator = numpy.array([risk_phi])

        log_lik = log_lik + y_pred_sum - weighted_average * log(denominator).sum()

        # reset tie values
        tied_death_counts = 0
        weight_count = 0.0
        y_pred_sum = 0
        tie_phi = 0

    return log_lik / n


class AvgLogLikelihoodTest(CIndexTest):
    def metric(self, y_true, y_pred):
        return avg_likelihood(y_true, y_pred)


class AsymmetricAverageLikelihoodTest(AsymmetricCIndexTest, AvgLogLikelihoodTest):
    pass


class SlightlyAsymmetricAverageLikelihoodTest(SlightlyAsymmetricCIndexTest, AvgLogLikelihoodTest):
    pass


class AccuracyTest(TestTest):

    def __init__(self, test_set_size: int, accuracy_1=0.74, accuracy_2=0.74):
        y_true = numpy.random.randint(0, 2, size=test_set_size)
        y_pred_1_correct = numpy.random.random(test_set_size) <= accuracy_1
        y_pred_1 = numpy.zeros_like(y_true)
        y_pred_1[y_pred_1_correct] = y_true[y_pred_1_correct]
        y_pred_1[~y_pred_1_correct] = 1 - y_true[~y_pred_1_correct]
        y_pred_2_correct = numpy.random.random(test_set_size) <= accuracy_2
        y_pred_2 = numpy.zeros_like(y_true)
        y_pred_2[y_pred_2_correct] = y_true[y_pred_2_correct]
        y_pred_2[~y_pred_2_correct] = 1 - y_true[~y_pred_2_correct]
        self.y_pred_1 = y_pred_1
        self.y_pred_2 = y_pred_2
        self.y_true = y_true

        super().__init__(test_set_size=test_set_size)

    def ground_truth(self):
        return self.y_true

    def model_outputs_1(self):
        return self.y_pred_1

    def model_outputs_2(self):
        return self.y_pred_2

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return True

    def metric(self, y_true, y_pred):
        return numpy.mean((y_pred >= 0.5) == y_true)


class CorrelatedAccuracyTest(AccuracyTest):
    def __init__(self, test_set_size: int, accuracy_1=0.74, accuracy_2=0.74):
        super().__init__(test_set_size=test_set_size, accuracy_1=accuracy_1, accuracy_2=accuracy_2)
        self.y_pred_2[:round(test_set_size / 2)] = self.y_pred_1[:round(test_set_size / 2)]


class BinaryCETest(TestTest):

    def __init__(self, test_set_size: int, accuracy_1=0.74, accuracy_2=0.74):
        y_true = numpy.random.randint(0, 2, size=test_set_size)

        y_pred_1_correct = numpy.random.random(test_set_size) <= accuracy_1
        y_pred_1 = numpy.zeros_like(y_true, dtype='float32')
        y_pred_1[y_pred_1_correct] = y_true[y_pred_1_correct] * accuracy_1 + (1 - y_true[y_pred_1_correct]) * (1 - accuracy_1)
        y_pred_1[~y_pred_1_correct] = y_true[~y_pred_1_correct] * (1 - accuracy_1) + (1 - y_true[~y_pred_1_correct]) * accuracy_1

        y_pred_2_correct = numpy.random.random(test_set_size) <= accuracy_2
        y_pred_2 = numpy.zeros_like(y_true, dtype='float32')
        y_pred_2[y_pred_2_correct] = y_true[y_pred_2_correct] * accuracy_2 + (1 - y_true[y_pred_2_correct]) * (1 - accuracy_2)
        y_pred_2[~y_pred_2_correct] = y_true[~y_pred_2_correct] * (1 - accuracy_2) + (1 - y_true[~y_pred_2_correct]) * accuracy_2

        self.y_pred_1 = y_pred_1
        self.y_pred_2 = y_pred_2
        self.y_true = y_true

        super().__init__(test_set_size=test_set_size)

    def ground_truth(self):
        return self.y_true

    def model_outputs_1(self):
        return self.y_pred_1

    def model_outputs_2(self):
        return self.y_pred_2

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return True

    def metric(self, y_true, y_pred, epsilon=1e-7):
        return numpy.mean(y_true * numpy.log(y_pred + epsilon) + (1 - y_true) * numpy.log(1 - y_pred + epsilon))


class AsymmetricBinaryCETest(BinaryCETest):
    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size, accuracy_1=0.80, accuracy_2=0.70)

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return False


class SlightlyAsymmetricBinaryCETest(BinaryCETest):
    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size, accuracy_1=0.78, accuracy_2=0.74)

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return False


class MSETest(TestTest):
    def __init__(self, test_set_size: int, uniform_width_1=2.0, uniform_width_2=2.0):
        y_true = numpy.zeros(test_set_size)
        y_pred_1 = numpy.random.uniform(size=test_set_size, low=-0.5, high=0.5) * uniform_width_1
        y_pred_2 = numpy.random.uniform(size=test_set_size, low=-0.5, high=0.5) * uniform_width_2
        self.y_pred_1 = y_pred_1
        self.y_pred_2 = y_pred_2
        self.y_true = y_true

        super().__init__(test_set_size=test_set_size)

    def ground_truth(self):
        return self.y_true

    def model_outputs_1(self):
        return self.y_pred_1

    def model_outputs_2(self):
        return self.y_pred_2

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return True

    def metric(self, y_true, y_pred):
        return numpy.mean((y_pred - y_true) ** 2)


class AsymmetricMSETest(MSETest):
    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size, uniform_width_1=4, uniform_width_2=2)

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return False


class SlightlyAsymmetricMSETest(MSETest):
    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size, uniform_width_1=2.5, uniform_width_2=2)

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return False


class AsymmetricAccuracyTest(AccuracyTest):

    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size, accuracy_1=0.90, accuracy_2=0.74)

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return False


class SlightlyAsymmetricAccuracyTest(AccuracyTest):

    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size, accuracy_1=0.80, accuracy_2=0.74)

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return False


class SameModelAccuracyTest(AccuracyTest):
    def __init__(self, test_set_size: int):
        super().__init__(test_set_size=test_set_size, accuracy_1=0.74, accuracy_2=0.74)
        self.y_pred_2 = self.y_pred_1

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return True


def p_value_calibration_overview(p_values):
    values, counts = (numpy.unique(p_values, return_counts=True)[0], numpy.cumsum(numpy.unique(p_values, return_counts=True)[1] / len(p_values)))
    value_counts = {k: v for k, v in zip(values, counts)}
    p_values = {}
    for value, count in value_counts.items():
        p_values[value] = binom.cdf(count, len(p_values), value)
    return value_counts


class LogLikelihoodTest(TestTest):
    def __init__(self, test_set_size: int):
        y_true = numpy.random.randint(0, 2, size=test_set_size)
        y_pred_1_correct = numpy.random.random(test_set_size) <= 0.74
        y_pred_1 = numpy.zeros_like(y_true, dtype=float)
        y_pred_1[y_pred_1_correct] = y_true[y_pred_1_correct]
        y_pred_1[~y_pred_1_correct] = 1 - y_true[~y_pred_1_correct]
        y_pred_1 = numpy.clip(y_pred_1, 1e-15, 1 - 1e-15)

        y_pred_2_correct = numpy.random.random(test_set_size) <= 0.74
        y_pred_2 = numpy.zeros_like(y_true, dtype=float)
        y_pred_2[y_pred_2_correct] = y_true[y_pred_2_correct]
        y_pred_2[~y_pred_2_correct] = 1 - y_true[~y_pred_2_correct]
        y_pred_2 = numpy.clip(y_pred_2, 1e-15, 1 - 1e-15)

        self.y_pred_1 = y_pred_1
        self.y_pred_2 = y_pred_2
        self.y_true = y_true

        super().__init__(test_set_size=test_set_size)

    def ground_truth(self):
        return self.y_true

    def model_outputs_1(self):
        return self.y_pred_1

    def model_outputs_2(self):
        return self.y_pred_2

    @classmethod
    def null_hypothesis_holds(cls) -> bool:
        return True

    @classmethod
    def metric(cls, y_true: numpy.ndarray, y_pred: numpy.ndarray) -> float:
        # Calculate log-likelihood
        log_likelihood = y_true * numpy.log(y_pred) + (1 - y_true) * numpy.log(1 - y_pred)
        return numpy.mean(log_likelihood)


def test_name(cls, idx, i):
    return (
            cls.__name__
            + f'{idx + 1:03d}_'
            + i['cls_test_data_generator'].__name__
            + ('_permutation_only' if i['permutation_only'] else 'skip_permutation' if i['skip_permutation'] else '_bootstrap')
            + ('_two_sided' if i['two_sided'] else '_one_sided')
            + ('_paired' if i['paired'] else '_unpaired')
    )


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


@parameterized_class(['cls_test_data_generator', 'permutation_only', 'two_sided', 'paired', 'skip_permutation'], relevant_test_types(), class_name_func=test_name)
class TestBootStrapTest(unittest.TestCase):
    cls_test_data_generator: Type[TestTest]
    permutation_only: bool
    two_sided: bool
    paired: bool
    skip_permutation: bool

    def bootstrap_test(self, n_bootstraps, y_true: numpy.ndarray, y_pred_1: numpy.ndarray, y_pred_2: numpy.ndarray, metric, y_true_2=None, verbose=0):
        return BootstrapModelComparison(n_bootstraps=n_bootstraps,
                                        y_true=y_true,
                                        y_true_2=y_true_2,
                                        y_pred_1=y_pred_1,
                                        y_pred_2=y_pred_2,
                                        metric=metric,
                                        permutation_only=self.permutation_only,
                                        two_sided=self.two_sided,
                                        paired=self.paired,
                                        verbose=verbose,
                                        skip_permutation=self.skip_permutation)().p_value

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


class TestLikelihoodFunction(unittest.TestCase):
    def test_likelihood_function(self):
        from lifelines import CoxPHFitter
        import pandas
        rossi = pandas.read_csv('rossi.csv')

        rossi = rossi.sample(frac=1.0, random_state=25)  # ensures the reproducibility of the example
        train_rossi = rossi.iloc[:400]
        test_rossi = rossi.iloc[400:]

        time_column = 'week'
        event_column = 'arrest'
        cph_l1 = CoxPHFitter(penalizer=0.1, l1_ratio=1.).fit(train_rossi, time_column, event_column)
        cph_l2 = CoxPHFitter(penalizer=0.1, l1_ratio=0.).fit(train_rossi, time_column, event_column)

        y_true = test_rossi[[time_column, event_column]].values
        y_pred_1 = numpy.array(cph_l1.predict_log_partial_hazard(test_rossi))
        y_pred_2 = numpy.array(cph_l2.predict_log_partial_hazard(test_rossi))

        print(cph_l1.score(test_rossi))
        print(avg_likelihood(y_true, y_pred_1))
        print(cph_l2.score(test_rossi))
        print(avg_likelihood(y_true, y_pred_2))

        assert math.isclose(avg_likelihood(y_true, y_pred_1), cph_l1.score(test_rossi))
        assert math.isclose(avg_likelihood(y_true, y_pred_2), cph_l2.score(test_rossi))


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