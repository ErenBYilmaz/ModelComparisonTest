"""
This snippet includes two important methods for comparing statistical models.
The method `permutation_based_model_comparison()` allows to compare two models outputs regarding a specificied metric and test under the null hypothesis that they come from the same distribution.
The class `PairwisePermutationModelComparison` allows automatically conducting and plotting multiple such comparisons for the use case where more than two models need to be compared.
See the respective docstrings for more information.

Author: Eren Bora Yilmaz, 2025
"""
from abc import abstractmethod, ABC
from typing import Callable, Union, Optional

import numpy
import seaborn
from matplotlib import pyplot

from comparison_result import ComparisonResult


class HypothesisTest:
    def compare(self,
                y_true: numpy.ndarray,
                y_pred_1: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                y_pred_2: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                y_true_2=None, ) -> ComparisonResult:
        """
        :param y_true: array shape (n_samples,) with ground truth labels or callable returning such an array based on sample indices
        :param y_pred_1: array shape (n_samples,) with outputs of model 1 or callable returning such an array based on sample indices
        :param y_pred_2: array shape (n_samples,) with outputs of model 2 or callable returning such an array based on sample indices
        :param y_true_2: array shape (n_samples,) with ground truth values corresponding to y_pred_2, to be used in case of an unpaired test.
        If omitted, y_true_2 is set to y_true.
        """
        raise NotImplementedError('Abstract method')

    @classmethod
    def allow_two_sided(cls):
        return True


class ResamplingBasedModelComparison(HypothesisTest):
    def __init__(self,
                 n_iterations: int,
                 metric: Callable[[numpy.ndarray, numpy.ndarray], float],  # argument order y_true, y_pred
                 two_sided=True,
                 verbose=0,
                 plot_histogram_to_file=None,
                 skip_validation=False):
        """
        Compare two models based on their outputs and respective ground truth.
        Both models need to be evaluated on the same test set, but no assumptions regarding training data or nestedness are required.
        :param n_iterations: number of samples to draw
        :param metric: callable taking two arrays of shape (n_samples,) and returning a float. The first argument is the ground truth, the second are the model outputs.
        :param two_sided: if True, the test is conducted in a two-sided way, otherwise one-sided
        :param verbose: verbosity level
        :param paired: if True, the test is conducted in a paired way, otherwise unpaired
        :param permutation_only: if True, the test is conducted by permuting the model outputs instead of bootstrapping
        :param skip_permutation: if True, the test is conducted by bootstrapping without permutation
        :param plot_histogram_to_file: if not None, a histogram of the differences in metric for the bootstrap iterations is plotted and saved to this file
        """
        self.n_iterations = n_iterations
        self.metric = metric
        self.two_sided = two_sided
        self.verbose = verbose
        self.plot_histogram_to_file = plot_histogram_to_file
        self.running = False
        # None as long as not running
        self.y_pred_1: Optional[Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]]] = None
        self.y_pred_2: Optional[Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]]] = None
        self.y_true: Optional[numpy.ndarray] = None
        self.y_true_2: Optional[numpy.ndarray] = None
        self.skip_validation = skip_validation

    def get_y_pred_array(self, y_pred: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]], num_samples: int):
        return y_pred(numpy.arange(num_samples)) if callable(y_pred) else y_pred

    def compare(self,
                y_true: numpy.ndarray,
                y_pred_1: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                y_pred_2: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                y_true_2=None, ) -> ComparisonResult:
        if y_true_2 is None:
            y_true_2 = y_true
        self.y_pred_1 = y_pred_1
        self.y_pred_2 = y_pred_2
        self.y_true = y_true
        self.y_true_2 = y_true_2
        if not self.skip_validation:
            self.validate_parameters(y_true=y_true, y_pred_1=y_pred_1, y_pred_2=y_pred_2, y_true_2=y_true_2, )

        y_pred_1_array = self.get_y_pred_array(self.y_pred_1, len(self.y_true))
        y_pred_2_array = self.get_y_pred_array(self.y_pred_2, len(self.y_true_2))
        observed_metric_1 = self.metric(self.y_true, y_pred_1_array)
        observed_metric_2 = self.metric(self.y_true_2, y_pred_2_array)

        bootstrap_metrics_1 = []
        bootstrap_metrics_2 = []
        for _ in range(self.n_iterations):
            bootstrap_indices_1, bootstrap_indices_2, from_dataset_1_1, from_dataset_1_2 = self.pick_samples()
            y_pred_1_array = self.get_y_pred_array(self.y_pred_1, len(self.y_true))
            y_pred_2_array = self.get_y_pred_array(self.y_pred_2, len(self.y_true_2))
            assert len(self.y_true) == len(y_pred_1_array)
            assert len(self.y_true_2) == len(y_pred_2_array)
            y_pred_b, y_true_b = self.get_sample(y_pred_1_array, y_pred_2_array, bootstrap_indices_1, from_dataset_1_1)
            bootstrap_metrics_1.append(self.metric(y_true_b, y_pred_b))

            y_pred_b, y_true_b = self.get_sample(y_pred_1_array, y_pred_2_array, bootstrap_indices_2, from_dataset_1_2)
            bootstrap_metrics_2.append(self.metric(y_true_b, y_pred_b))

        bootstrap_metrics_1 = numpy.array(bootstrap_metrics_1)
        bootstrap_metrics_2 = numpy.array(bootstrap_metrics_2)

        test_set_difference = observed_metric_1 - observed_metric_2

        if self.two_sided:
            transform = lambda x: numpy.abs(x)
        else:
            transform = lambda x: x

        threshold = transform(test_set_difference)
        differences = transform(bootstrap_metrics_1 - bootstrap_metrics_2)
        larger_equal_differences = self.count_outliers(differences, threshold)
        p_value_bootstrap = (larger_equal_differences + 1) / (self.n_iterations + 1)
        # p_value_bootstrap = (num_larger_differences + numpy.random.binomial(num_equal_differences, 0.5) + 1) / (n_bootstraps + 1)
        assert 0 < p_value_bootstrap <= 1
        if self.verbose >= 2:
            print()
            print(f'n_iterations: {self.n_iterations}')
            print(f'Observed metric 1: {observed_metric_1}')
            print(f'Observed metric 2: {observed_metric_2}')
            print(f'Observed difference: {test_set_difference}')
            print(f'p-value: {p_value_bootstrap}')

        if self.plot_histogram_to_file is not None:
            fig, ax = pyplot.subplots()
            seaborn.histplot(bootstrap_metrics_1 - bootstrap_metrics_2, ax=ax, bins=50)
            ax.axvline(test_set_difference, color='red')
            ax.axvline(-test_set_difference, color='red')
            ax.set_title(f'Difference in metric between model 1 and model 2.\nRed: test set difference. p = {p_value_bootstrap:.3f}')
            pyplot.savefig(self.plot_histogram_to_file)
            print('Created', self.plot_histogram_to_file)
            pyplot.close()

        return ComparisonResult(p_value_bootstrap, test_set_difference > 0)

    def count_outliers(self, differences: numpy.ndarray, threshold: float) -> int:
        raise NotImplementedError('Abstract method')

    def validate_parameters(self,
                            y_true: numpy.ndarray,
                            y_pred_1: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                            y_pred_2: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                            y_true_2=None, ):
        if self.verbose >= 1 and self.n_iterations < 99:
            print(f'n_iterations is relatively low at {self.n_iterations}. This function will only return multiples of 1 / (2 * (n_bootstraps + 1)) = {1 / (self.n_iterations + 1):.3g} so '
                  f'it is recommended to use those as significance level α and then check p <= α or p < α + {1 / (self.n_iterations + 1):.3g} instead of p < α. '
                  f'Alternatively, increase the number of iterations to get more fine-grained p-values.')

    def get_sample(self, y_pred_1_array, y_pred_2_array, sample_indices_1, from_dataset_1):
        combined_array = numpy.concatenate([y_pred_1_array, y_pred_2_array])
        combined_y_true = numpy.concatenate([self.y_true, self.y_true_2])
        adjusted_indices = sample_indices_1 + (~from_dataset_1) * len(y_pred_1_array)
        return combined_array[adjusted_indices], combined_y_true[adjusted_indices]

    @abstractmethod
    def pick_samples(self):
        pass


class PairedTestMixin(ResamplingBasedModelComparison, ABC):
    def validate_parameters(self,
                            y_true: numpy.ndarray,
                            y_pred_1: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                            y_pred_2: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                            y_true_2=None, ):
        if self.verbose >= 1 and self.n_iterations < 99:
            print(f'n_bootstraps is relatively low at {self.n_iterations}. This function will only return multiples of 1 / (2 * (n_bootstraps + 1)) = {1 / (self.n_iterations + 1):.3g} so '
                  f'it is recommended to use those as significance level α and then check p <= α or p < α + {1 / (self.n_iterations + 1):.3g} instead of p < α. '
                  f'Alternatively, increase the number of bootstraps to get more fine-grained p-values.')
        if (y_true_2 is not y_true):
            raise NotImplementedError('Paired tests with two different ground truths are highly questionable. Might even crash. Are you sure this is a good idea?')
        super().validate_parameters(y_true=y_true, y_pred_1=y_pred_1, y_pred_2=y_pred_2, y_true_2=y_true_2, )


class BootstrapModelComparisonUnpaired(ResamplingBasedModelComparison):
    def validate_parameters(self,
                            y_true: numpy.ndarray,
                            y_pred_1: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                            y_pred_2: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                            y_true_2=None, ):
        if self.two_sided:
            raise NotImplementedError('I have no idea how to do a bootstrap-only test in a two-sided way.')
        super().validate_parameters(y_true=y_true, y_pred_1=y_pred_1, y_pred_2=y_pred_2, y_true_2=y_true_2, )

    @classmethod
    def allow_two_sided(cls):
        return False

    def pick_samples(self):
        # sample_indices if they were all from one large array y_true+y_true_2
        combined_indices = numpy.arange(len(self.y_true) + len(self.y_true_2))
        combined_indices_1 = combined_indices[:len(self.y_true)]
        combined_indices_2 = combined_indices[len(self.y_true):]
        combined_indices_1 = numpy.random.choice(combined_indices_1, len(combined_indices_1), replace=True)
        combined_indices_2 = numpy.random.choice(combined_indices_2, len(combined_indices_2), replace=True)
        separated_indices_1 = combined_indices_1 - (combined_indices_1 >= len(self.y_true)) * len(self.y_true)
        from_dataset_1_1 = combined_indices_1 < len(self.y_true)
        separated_indices_2 = combined_indices_2 - (combined_indices_2 >= len(self.y_true)) * len(self.y_true)
        from_dataset_1_2 = combined_indices_2 < len(self.y_true)
        return separated_indices_1, separated_indices_2, from_dataset_1_1, from_dataset_1_2

    def count_outliers(self, differences: numpy.ndarray, threshold: float) -> int:
        return numpy.count_nonzero(differences <= 0)


class PermutationModelComparisonUnpaired(ResamplingBasedModelComparison):
    def pick_samples(self):
        combined_indices = numpy.arange(len(self.y_true) + len(self.y_true_2))
        numpy.random.shuffle(combined_indices)
        combined_indices_1 = combined_indices[:len(self.y_true)]
        combined_indices_2 = combined_indices[len(self.y_true):]
        separated_indices_1 = combined_indices_1 - (combined_indices_1 >= len(self.y_true)) * len(self.y_true)
        from_dataset_1_1 = combined_indices_1 < len(self.y_true)
        separated_indices_2 = combined_indices_2 - (combined_indices_2 >= len(self.y_true)) * len(self.y_true)
        from_dataset_1_2 = combined_indices_2 < len(self.y_true)
        return separated_indices_1, separated_indices_2, from_dataset_1_1, from_dataset_1_2

    def count_outliers(self, differences: numpy.ndarray, threshold: float) -> int:
        return numpy.count_nonzero(differences >= threshold)


class PermutationModelComparisonPaired(PairedTestMixin, PermutationModelComparisonUnpaired):
    def pick_samples(self):
        combined_indices = numpy.arange(len(self.y_true) + len(self.y_true_2))
        flip_at = numpy.random.random(len(combined_indices)) < 0.5
        combined_indices[flip_at] = (combined_indices[flip_at] + len(self.y_true)) % len(combined_indices)
        combined_indices_1 = combined_indices[:len(self.y_true)]
        separated_indices_1 = combined_indices_1 - (combined_indices_1 >= len(self.y_true)) * len(self.y_true)
        from_dataset_1_1 = combined_indices_1 < len(self.y_true)
        from_dataset_1_2 = ~from_dataset_1_1
        separated_indices_2 = separated_indices_1
        return separated_indices_1, separated_indices_2, from_dataset_1_1, from_dataset_1_2


class BootstrapModelComparisonPaired(PairedTestMixin, BootstrapModelComparisonUnpaired):
    def pick_samples(self):
        combined_indices = numpy.arange(len(self.y_true) + len(self.y_true_2))
        combined_indices_1 = combined_indices[:len(self.y_true)]
        combined_indices_1 = numpy.random.choice(combined_indices_1, len(combined_indices_1), replace=True)
        separated_indices_1 = combined_indices_1 - (combined_indices_1 >= len(self.y_true)) * len(self.y_true)
        from_dataset_1_1 = combined_indices_1 < len(self.y_true)
        from_dataset_1_2 = ~from_dataset_1_1
        separated_indices_2 = separated_indices_1
        return separated_indices_1, separated_indices_2, from_dataset_1_1, from_dataset_1_2


class BootstrapPlusPermutationComparisonUnpaired(ResamplingBasedModelComparison):
    def pick_samples(self):
        combined_indices = numpy.arange(len(self.y_true) + len(self.y_true_2))
        combined_indices = numpy.random.choice(combined_indices, len(combined_indices), replace=False)
        combined_indices_1 = combined_indices[:len(self.y_true)]
        combined_indices_2 = combined_indices[len(self.y_true):]
        separated_indices_1 = combined_indices_1 - (combined_indices_1 >= len(self.y_true)) * len(self.y_true)
        from_dataset_1_1 = combined_indices_1 < len(self.y_true)
        separated_indices_2 = combined_indices_2 - (combined_indices_2 >= len(self.y_true)) * len(self.y_true)
        from_dataset_1_2 = combined_indices_2 < len(self.y_true)
        return separated_indices_1, separated_indices_2, from_dataset_1_1, from_dataset_1_2

    def count_outliers(self, differences: numpy.ndarray, threshold: float) -> int:
        return numpy.count_nonzero(differences >= threshold)

    def validate_parameters(self,
                            y_true: numpy.ndarray,
                            y_pred_1: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                            y_pred_2: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                            y_true_2=None, ):
        print('Warning: I noticed that bootstrap breaks p-values, by a tiny bit, but enough to make them not work. Use at your own risk or switch to permutation test.')
        super().validate_parameters(y_true=y_true, y_pred_1=y_pred_1, y_pred_2=y_pred_2, y_true_2=y_true_2, )


class BootstrapPlusPermutationComparisonPaired(PairedTestMixin, BootstrapPlusPermutationComparisonUnpaired):
    def pick_samples(self):
        combined_indices = numpy.arange(len(self.y_true) + len(self.y_true_2))
        combined_indices = numpy.random.choice(combined_indices, len(combined_indices), replace=False)
        combined_indices_1 = combined_indices[:len(self.y_true)]
        separated_indices_1 = combined_indices_1 - (combined_indices_1 >= len(self.y_true)) * len(self.y_true)
        from_dataset_1_1 = combined_indices_1 < len(self.y_true)
        from_dataset_1_2 = ~from_dataset_1_1
        separated_indices_2 = separated_indices_1
        return separated_indices_1, separated_indices_2, from_dataset_1_1, from_dataset_1_2


class LikelihoodRatioTestForBinaryModels(HypothesisTest):
    def __init__(self, degrees_of_freedom: int):
        """
        :param degrees_of_freedom: degrees of freedom for the likelihood ratio test
        """
        self.degrees_of_freedom = degrees_of_freedom

    @classmethod
    def allow_two_sided(cls):
        return False

    def compare(self,
                y_true: numpy.ndarray,
                y_pred_1: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                y_pred_2: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                y_true_2=None, ) -> ComparisonResult:
        from scipy.stats import chi2
        l1 = self.log_likelihood(y_true, y_pred_1)
        if y_true_2 is None:
            y_true_2 = y_true
        l2 = self.log_likelihood(y_true_2, y_pred_2)
        if l1 < l2:
            raise ValueError('Log likelihood of model 1 is smaller than log likelihood of model 2. This is not possible for a likelihood ratio test.')
        log_ratio = 2 * (l1 - l2)
        degrees_of_freedom = self.degrees_of_freedom
        assert self.degrees_of_freedom >= 0
        p_value = 1 - chi2.cdf(log_ratio, degrees_of_freedom)
        return ComparisonResult(p_value, l1 > l2)

    def log_likelihood(self, y_true: numpy.ndarray, y_pred: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]], epsilon=1e-7) -> float:
        """
        :param y_true: array shape (n_samples,) with ground truth labels
        :param y_pred: array shape (n_samples,) with outputs of the model
        :param epsilon: small value to avoid division by zero
        """
        return numpy.sum(y_true * numpy.log(y_pred + epsilon) + (1 - y_true) * numpy.log(1 - y_pred + epsilon))
