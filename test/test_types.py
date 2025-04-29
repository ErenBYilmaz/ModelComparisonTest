from typing import Type

import numpy

from test.test_likelihood_function import avg_likelihood


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

    @classmethod
    def metric(cls, y_true: numpy.ndarray, y_pred: numpy.ndarray) -> float:
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

        @classmethod
        def metric(self, y_true, y_pred):
            return t.metric(y_true, y_pred)

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

    @classmethod
    def metric(cls, y_true, y_pred):
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


class AvgLogLikelihoodTest(CIndexTest):
    @classmethod
    def metric(cls, y_true, y_pred):
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

    @classmethod
    def metric(cls, y_true, y_pred):
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

    @classmethod
    def metric(cls, y_true, y_pred, epsilon=1e-7):
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

    @classmethod
    def metric(cls, y_true, y_pred):
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
        return numpy.mean(log_likelihood).item()


