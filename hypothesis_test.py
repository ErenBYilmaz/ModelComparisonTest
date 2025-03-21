"""
This snippet includes two important methods for comparing statistical models.
The method `permutation_based_model_comparison()` allows to compare two models outputs regarding a specificied metric and test under the null hypothesis that they come from the same distribution.
The class `PairwisePermutationModelComparison` allows automatically conducting and plotting multiple such comparisons for the use case where more than two models need to be compared.
See the respective docstrings for more information.

Author: Eren Bora Yilmaz, 2025
"""

import itertools
import math
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Union

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from tqdm import tqdm

from utils.tuned_cache import TunedMemory

results_cache = TunedMemory('.cache', verbose=0)


def permutation_based_model_comparison(n_permutations: int,
                                       y_true: numpy.ndarray,
                                       y_pred_1: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                                       y_pred_2: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                                       metric: Callable[[numpy.ndarray, numpy.ndarray], float],  # argument order y_true, y_pred
                                       two_sided=True,
                                       verbose=0,
                                       plot_histogram_to_file=None):
    """
    Compare two models based on their outputs and respective ground truth.
    Both models need to be evaluated on the same test set, but no assumptions regarding training data or nestedness are required.
    See also the documentation of PairwisePermutationModelComparison.
    :param n_permutations: number of iterations.
    :param y_true: array shape (n_samples,) with ground truth labels or callable returning such an array based on sample indices
    :param y_pred_1: array shape (n_samples,) with outputs of model 1 or callable returning such an array based on sample indices
    :param y_pred_2: array shape (n_samples,) with outputs of model 2 or callable returning such an array based on sample indices
    :param metric: callable taking two arrays of shape (n_samples,) and returning a float. The first argument is the ground truth, the second are the model outputs.
    :param two_sided: if True, the test is conducted in a two-sided way, otherwise one-sided
    :param verbose: verbosity level
    :param plot_histogram_to_file: if not None, a histogram of the differences in metric for the bootstrap iterations is plotted and saved to this file
    """
    return bootstrap_based_model_comparison(n_permutations,
                                            y_true,
                                            y_pred_1,
                                            y_pred_2,
                                            metric,
                                            two_sided=two_sided,
                                            verbose=verbose,
                                            paired=True,
                                            permutation_only=True,
                                            skip_permutation=False,
                                            plot_histogram_to_file=plot_histogram_to_file)


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



def bootstrap_based_model_comparison(n_bootstraps: int,
                                     y_true: numpy.ndarray,
                                     y_pred_1: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                                     y_pred_2: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                                     metric: Callable[[numpy.ndarray, numpy.ndarray], float],  # argument order y_true, y_pred
                                     two_sided=True,
                                     verbose=0,
                                     paired=True,
                                     permutation_only=False,
                                     skip_permutation=False,
                                     plot_histogram_to_file=None):
    """
    Compare two models based on their outputs and respective ground truth.
    Both models need to be evaluated on the same test set, but no assumptions regarding training data or nestedness are required.
    :param n_bootstraps: number of bootstrap samples to draw
    :param y_true: array shape (n_samples,) with ground truth labels or callable returning such an array based on sample indices
    :param y_pred_1: array shape (n_samples,) with outputs of model 1 or callable returning such an array based on sample indices
    :param y_pred_2: array shape (n_samples,) with outputs of model 2 or callable returning such an array based on sample indices
    :param metric: callable taking two arrays of shape (n_samples,) and returning a float. The first argument is the ground truth, the second are the model outputs.
    :param two_sided: if True, the test is conducted in a two-sided way, otherwise one-sided
    :param verbose: verbosity level
    :param paired: if True, the test is conducted in a paired way, otherwise unpaired
    :param permutation_only: if True, the test is conducted by permuting the model outputs instead of bootstrapping
    :param skip_permutation: if True, the test is conducted by bootstrapping without permutation
    :param plot_histogram_to_file: if not None, a histogram of the differences in metric for the bootstrap iterations is plotted and saved to this file
    """
    if permutation_only and not paired:
        raise NotImplementedError('This seems contradictory. I have to think a bit more about this')
    if permutation_only and skip_permutation:
        raise ValueError('Both permutation_only and skip_permutation are set to True. This is contradictory.')
    if skip_permutation and two_sided:
        raise NotImplementedError('I have not idea how to do a bootstrap-only test in a two-sided way.')

    if verbose >= 1 and n_bootstraps < 100:
        print(f'n_bootstraps is relatively low at {n_bootstraps}. This function will only return multiples of 1 / (2 * (n_bootstraps + 1)) = {1 / (n_bootstraps + 1):.3g} so '
              f'it is recommended to use those as significance level α and then check p <= α or p < α + {1 / (n_bootstraps + 1):.3g} instead of p < α. '
              f'Alternatively, increase the number of bootstraps to get more fine-grained p-values.')
    double_y_true = numpy.concatenate([y_true, y_true])

    y_pred_1_array = y_pred_1(numpy.arange(len(y_true))) if callable(y_pred_1) else y_pred_1
    y_pred_2_array = y_pred_2(numpy.arange(len(y_true))) if callable(y_pred_2) else y_pred_2
    observed_metric_1 = metric(y_true, y_pred_1_array)
    observed_metric_2 = metric(y_true, y_pred_2_array)

    bootstrap_metrics_1 = []
    bootstrap_metrics_2 = []
    for _ in range(n_bootstraps):
        if permutation_only:
            bootstrap_indices = numpy.arange(len(y_true))
            bootstrap_indices[numpy.random.random(len(y_true)) < 0.5] += len(y_true)
        elif skip_permutation:
            bootstrap_indices = numpy.random.choice(len(y_true), len(y_true), replace=True)
        else:
            bootstrap_indices = numpy.random.choice(2 * len(y_true), len(y_true), replace=True)  # TODO what about outputs of model 1 and model 2 for the same patient in a single bootstrap
        y_pred_1_array = y_pred_1(bootstrap_indices % len(y_true)) if callable(y_pred_1) else y_pred_1
        y_pred_2_array = y_pred_2(bootstrap_indices % len(y_true)) if callable(y_pred_2) else y_pred_2
        assert len(y_true) == len(y_pred_1_array) == len(y_pred_2_array)
        combined_y_pred = numpy.concatenate([y_pred_1_array, y_pred_2_array])
        combined_y_pred_flipped = numpy.concatenate([y_pred_2_array, y_pred_1_array])

        y_pred_b = combined_y_pred[bootstrap_indices]
        y_true_b = double_y_true[bootstrap_indices]
        bootstrap_metrics_1.append(metric(y_true_b, y_pred_b))

        if not paired:
            bootstrap_indices = numpy.random.choice(len(combined_y_pred), len(y_pred_2), replace=True)
        y_pred_b = numpy.array(combined_y_pred_flipped)[bootstrap_indices]
        y_true_b = numpy.array(double_y_true)[bootstrap_indices]
        bootstrap_metrics_2.append(metric(y_true_b, y_pred_b))

    bootstrap_metrics_1 = numpy.array(bootstrap_metrics_1)
    bootstrap_metrics_2 = numpy.array(bootstrap_metrics_2)

    test_set_difference = observed_metric_1 - observed_metric_2

    if two_sided:
        transform = lambda x: numpy.abs(x)
    else:
        transform = lambda x: x

    threshold = transform(test_set_difference)
    if skip_permutation:
        larger_difference: numpy.ndarray = transform(bootstrap_metrics_1 - bootstrap_metrics_2) < 0
    else:
        larger_difference: numpy.ndarray = transform(bootstrap_metrics_1 - bootstrap_metrics_2) > threshold
    num_larger_differences = numpy.count_nonzero(larger_difference)
    equal_difference: numpy.ndarray = transform(bootstrap_metrics_1 - bootstrap_metrics_2) == threshold
    num_equal_differences = numpy.count_nonzero(equal_difference)
    p_value_bootstrap = (num_larger_differences + num_equal_differences + 1) / (n_bootstraps + 1)
    # p_value_bootstrap = (num_larger_differences + numpy.random.binomial(num_equal_differences, 0.5) + 1) / (n_bootstraps + 1)
    if verbose >= 2:
        print()
        print(f'n_bootstraps: {n_bootstraps}')
        print(f'Observed difference: {test_set_difference}')
        print(f'p-value from bootstraps: {p_value_bootstrap}')

    if plot_histogram_to_file is not None:
        fig, ax = pyplot.subplots()
        seaborn.histplot(bootstrap_metrics_1 - bootstrap_metrics_2, ax=ax, bins=50)
        ax.axvline(test_set_difference, color='red')
        ax.axvline(-test_set_difference, color='red')
        ax.set_title(f'Difference in metric between model 1 and model 2.\nRed: test set difference. p = {p_value_bootstrap:.3f}')
        pyplot.savefig(plot_histogram_to_file)
        print('Created', plot_histogram_to_file)
        pyplot.close()

    return ComparisonResult(p_value_bootstrap, test_set_difference > 0)


def _compare_models_cache_key(metric, model_1: str, model_2: str, n_bootstraps, verbose, y_preds, y_true, two_sided, permutation_only, _extra_cache_key):
    result = (len(y_true), metric.__name__, model_1, model_2, n_bootstraps, two_sided, _extra_cache_key)
    if permutation_only:
        result = result + (permutation_only,)
    return result


@results_cache.cache(cache_key=_compare_models_cache_key)
def cached_compare_models(metric, model_1: str, model_2: str, n_bootstraps, verbose, y_preds, y_true, two_sided, permutation_only, _extra_cache_key):
    return bootstrap_based_model_comparison(n_bootstraps, y_true, y_preds[model_1], y_preds[model_2], metric, permutation_only=permutation_only, verbose=verbose, two_sided=two_sided)


class PairwiseModelComparison(metaclass=ABCMeta):
    def __init__(self,
                 to_file: str,
                 model_names: List[str],
                 verbose=0,
                 plot_title_addition: str = None, ):
        self.verbose = verbose
        self.plot_title_addition = plot_title_addition
        self.to_file = to_file
        self.model_names = model_names

    def plot_pairwise_comparisons(self, symmetric=True):
        results_table = []
        results_table_rows = self.model_names
        results_table_columns = self.model_names
        if symmetric:
            pairs = list((model_1, model_2)
                         for model_1, model_2 in itertools.product(results_table_rows, results_table_columns)
                         if results_table_rows.index(model_1) <= results_table_rows.index(model_2))
        else:
            pairs = list(itertools.product(results_table_rows, results_table_columns))
        results = {}
        for model_1, model_2 in tqdm(pairs):
            result = self.compare_models(model_1, model_2)
            larger_metric_name = model_1 if result.model_1_metric_larger else model_2
            if self.verbose:
                print(f'{model_1} vs {model_2}: p = {result.p_value}, avg larger metric for "{larger_metric_name}"')
            results[(model_1, model_2)] = result
        for row in results_table_rows:
            results_table.append([])
            for column in results_table_columns:
                if (row, column) in results:
                    results_table[-1].append(results[(row, column)])
                elif (column, row) in results:
                    results_table[-1].append(results[(column, row)].flipped())
                else:
                    results_table[-1].append(ComparisonResult(math.nan, False))
        p_values_table = pandas.DataFrame([[result.p_value for result in row] for row in results_table],
                                          index=results_table_rows, columns=results_table_columns)
        colors_table = pandas.DataFrame([[result.table_color() for result in row] for row in results_table],
                                        index=results_table_rows, columns=results_table_columns)

        fig, ax = pyplot.subplots(figsize=(24 * 1.2, 12 * 1.2))

        # Generate the heatmap including the mask
        seaborn.heatmap(colors_table,
                        annot=p_values_table,
                        annot_kws={"fontsize": 10},
                        cbar_kws={'ticks': ComparisonResult.example_color_ticks()},
                        fmt='.3f',
                        linewidths=0.5,
                        vmin=-1,
                        vmax=1,
                        cmap='RdBu',
                        ax=ax)
        p_values_table.to_csv(self.to_file + '_p.csv', index=True)
        colors_table.to_csv(self.to_file + '_c.csv', index=True)
        ax.collections[0].colorbar.set_ticklabels(ComparisonResult.example_color_ticks_labels())
        title = f'Pairwise model comparisons by {self.metric_name()}. Blue: Row larger, red: Column larger'
        if self.plot_title_addition is not None:
            title = title + f' ({self.plot_title_addition})'
        pyplot.title(title)
        pyplot.tight_layout()
        pyplot.savefig(self.to_file)
        pyplot.close()
        print('Wrote', self.to_file)

    @abstractmethod
    def metric_name(self):
        pass

    @abstractmethod
    def compare_models(self, model_1: str, model_2: str) -> ComparisonResult:
        pass


class PairwiseBootstrapModelComparison(PairwiseModelComparison):
    def __init__(self,
                 to_file: str,
                 n_bootstraps: int,
                 metric: Callable[[numpy.ndarray, numpy.ndarray], float],  # argument order y_true, y_pred
                 y_true: numpy.ndarray,
                 y_preds: Dict[str, numpy.ndarray],
                 two_sided=True,
                 permutation_only=False,
                 verbose=0,
                 extra_cache_key=None,
                 plot_title_addition: str = None,
                 ):
        super().__init__(to_file, list(y_preds.keys()), verbose, plot_title_addition)
        self.n_bootstraps = n_bootstraps
        self.extra_cache_key = extra_cache_key
        self.y_true = y_true
        self.y_preds = y_preds
        self.metric = metric
        self.permutation_only = permutation_only
        self.two_sided = two_sided

    def metric_name(self):
        return self.metric.__name__

    def compare_models(self, model_1, model_2):
        return cached_compare_models(self.metric, model_1, model_2, self.n_bootstraps, self.verbose, self.y_preds, self.y_true,
                                     permutation_only=self.permutation_only,
                                     two_sided=self.two_sided, _extra_cache_key=self.extra_cache_key)


class PairwisePermutationModelComparison(PairwiseBootstrapModelComparison):
    """
    Allows easy comparison of multiple models based on their outputs and respective ground truth.
    All models need to be evaluated on the same test set, but no assumptions regarding training data or nestedness are required.
    Example usage:
    ```
    PairwisePermutationModelComparison(
        to_file='results.png',
        n_permutations=1000,
        metric=roc_auc_score,
        y_true=y_true,
        y_preds = {
            'model1': y_pred_1,
            'model2': y_pred_2,
            'model3': y_pred_3,
            'model4': y_pred_4,
        }
    )
    """

    def __init__(self,
                 to_file: str,
                 n_permutations: int,
                 metric: Callable[[numpy.ndarray, numpy.ndarray], float],  # argument order y_true, y_pred
                 y_true: numpy.ndarray,
                 y_preds: Dict[str, numpy.ndarray],
                 two_sided=True,
                 verbose=0,
                 extra_cache_key=None,
                 plot_title_addition: str = None,
                 ):
        super().__init__(to_file,
                         n_permutations,
                         metric,
                         y_true,
                         y_preds,
                         two_sided,
                         permutation_only=True,
                         verbose=verbose,
                         extra_cache_key=extra_cache_key,
                         plot_title_addition=plot_title_addition)


def non_permutation_bootstrap_based_model_comparison(n_bootstraps: int,
                                                     y_true: numpy.ndarray,
                                                     y_pred_1: numpy.ndarray,
                                                     y_pred_2: numpy.ndarray,
                                                     metric: Callable[[numpy.ndarray, numpy.ndarray], float],  # argument order y_true, y_pred
                                                     verbose=0):
    raise DeprecationWarning('This function is not recommended for use. Unittests fail and p values are wrong. Use bootstrap_based_model_comparison instead.')
    if verbose >= 1 and n_bootstraps < 100:
        print(f'n_bootstraps is relatively low at {n_bootstraps}. This function will only return multiples of 1 / (2 * (n_bootstraps + 1)) = {1 / (n_bootstraps + 1):.3g} so '
              f'it is recommended to use those as significance level α and then check p <= α or p < α + {1 / (n_bootstraps + 1):.3g} instead of p < α. '
              f'Alternatively, increase the number of bootstraps to get more fine-grained p-values.')
    assert len(y_true) == len(y_pred_1) == len(y_pred_2)

    observed_metric_1 = metric(y_true, y_pred_1)
    observed_metric_2 = metric(y_true, y_pred_2)

    bootstrap_metrics_1 = []
    bootstrap_metrics_2 = []
    for _ in range(n_bootstraps):
        bootstrap_indices = numpy.random.choice(len(y_true), len(y_true), replace=True)
        y_pred_1_b = y_pred_1[bootstrap_indices]
        y_pred_2_b = y_pred_2[bootstrap_indices]
        y_true_b = y_true[bootstrap_indices]
        bootstrap_metrics_1.append(metric(y_true_b, y_pred_1_b))
        bootstrap_metrics_2.append(metric(y_true_b, y_pred_2_b))

    bootstrap_metrics_1 = numpy.array(bootstrap_metrics_1)
    bootstrap_metrics_2 = numpy.array(bootstrap_metrics_2)
    bootstrap_differences = bootstrap_metrics_1 - bootstrap_metrics_2

    test_set_difference = observed_metric_1 - observed_metric_2

    threshold = 0
    above_zero: numpy.ndarray = numpy.abs(bootstrap_differences) > threshold
    num_above_zero = numpy.count_nonzero(above_zero)
    below_zero: numpy.ndarray = numpy.abs(bootstrap_differences) < threshold
    num_below_zero = numpy.count_nonzero(below_zero)
    equal_zero = numpy.abs(bootstrap_differences) == threshold
    num_equal_zero = numpy.count_nonzero(equal_zero)

    if num_above_zero < num_below_zero:
        num_above_zero, num_below_zero = num_below_zero, num_above_zero  # two-sided test

    # p_value_one_sided = (num_above_zero + num_equal_zero * 0.5 + 1) / (n_bootstraps + 1)

    # equal_difference: numpy.ndarray = numpy.abs(bootstrap_metrics_1 - bootstrap_metrics_2) == 0
    # num_equal_differences = numpy.count_nonzero(equal_difference)
    # p_value_bootstrap = (num_above_zero + numpy.random.binomial(num_equal_differences, 0.5) + 1) / (n_bootstraps + 1)
    p_value_bootstrap = (num_below_zero + num_equal_zero * 0.5) / (n_bootstraps)
    p_value_bootstrap *= 2  # two-sided test
    if verbose >= 2:
        print()
        print(f'n_bootstraps: {n_bootstraps}')
        print(f'Observed difference: {test_set_difference}')
        print(f'p-value from bootstraps: {p_value_bootstrap}')

    return p_value_bootstrap
