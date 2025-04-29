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
from typing import Callable, Dict, List, Union, Optional

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from tqdm import tqdm

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


def permutation_based_model_comparison(n_permutations: int,
                                       y_true: numpy.ndarray,
                                       y_pred_1: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                                       y_pred_2: Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]],
                                       metric: Callable[[numpy.ndarray, numpy.ndarray], float],  # argument order y_true, y_pred
                                       two_sided=True,
                                       paired=True,
                                       y_true_2=None,
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
    :param paired: if True, the test is conducted in a paired way, otherwise unpaired
    :param verbose: verbosity level
    :param plot_histogram_to_file: if not None, a histogram of the differences in metric for the bootstrap iterations is plotted and saved to this file
    """
    return BootstrapModelComparison(
        n_permutations,
        metric,
        two_sided=two_sided,
        verbose=verbose,
        paired=paired,
        permutation_only=True,
        skip_permutation=False,
        plot_histogram_to_file=plot_histogram_to_file
    ).compare(y_true, y_pred_1, y_pred_2, y_true_2=y_true_2)


class BootstrapModelComparison(HypothesisTest):
    def __init__(self,
                 n_iterations: int,
                 metric: Callable[[numpy.ndarray, numpy.ndarray], float],  # argument order y_true, y_pred
                 two_sided=True,
                 verbose=0,
                 paired=True,
                 permutation_only=True,
                 skip_permutation=False,
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
        self.n_bootstraps = n_iterations
        self.metric = metric
        self.two_sided = two_sided
        self.verbose = verbose
        self.paired = paired
        self.permutation_only = permutation_only
        self.skip_permutation = skip_permutation
        self.plot_histogram_to_file = plot_histogram_to_file
        self.running = False
        # None as long as not running
        self.y_pred_1: Optional[Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]]] = None
        self.y_pred_2: Optional[Union[numpy.ndarray, Callable[[numpy.ndarray], numpy.ndarray]]] = None
        self.y_true: Optional[numpy.ndarray] = None
        self.y_true_2: Optional[numpy.ndarray] = None
        if not skip_validation:
            self.validate_parameters()

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
        if (y_true_2 is not y_true) and self.paired:
            raise NotImplementedError('Paired tests with two different ground truths are highly questionable. Might even crash. Are you sure this is a good idea?')

        y_pred_1_array = self.get_y_pred_array(self.y_pred_1, len(self.y_true))
        y_pred_2_array = self.get_y_pred_array(self.y_pred_2, len(self.y_true_2))
        observed_metric_1 = self.metric(self.y_true, y_pred_1_array)
        observed_metric_2 = self.metric(self.y_true_2, y_pred_2_array)

        bootstrap_metrics_1 = []
        bootstrap_metrics_2 = []
        for _ in range(self.n_bootstraps):
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
        if self.skip_permutation:
            threshold = 0
            larger_difference: numpy.ndarray = transform(bootstrap_metrics_1 - bootstrap_metrics_2) < 0
        else:
            larger_difference: numpy.ndarray = transform(bootstrap_metrics_1 - bootstrap_metrics_2) > threshold
        num_larger_differences = numpy.count_nonzero(larger_difference)
        equal_difference: numpy.ndarray = transform(bootstrap_metrics_1 - bootstrap_metrics_2) == threshold
        num_equal_differences = numpy.count_nonzero(equal_difference)
        p_value_bootstrap = (num_larger_differences + num_equal_differences + 1) / (self.n_bootstraps + 1)
        # p_value_bootstrap = (num_larger_differences + numpy.random.binomial(num_equal_differences, 0.5) + 1) / (n_bootstraps + 1)
        assert 0 < p_value_bootstrap <= 1
        if self.verbose >= 2:
            print()
            print(f'n_iterations: {self.n_bootstraps}')
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

    def validate_parameters(self):
        if self.permutation_only and self.skip_permutation:
            raise ValueError('Both permutation_only and skip_permutation are set to True. This is contradictory.')
        if self.skip_permutation and self.two_sided:
            raise NotImplementedError('I have no idea how to do a bootstrap-only test in a two-sided way.')
        if self.verbose >= 1 and self.n_bootstraps < 99:
            print(f'n_bootstraps is relatively low at {self.n_bootstraps}. This function will only return multiples of 1 / (2 * (n_bootstraps + 1)) = {1 / (self.n_bootstraps + 1):.3g} so '
                  f'it is recommended to use those as significance level α and then check p <= α or p < α + {1 / (self.n_bootstraps + 1):.3g} instead of p < α. '
                  f'Alternatively, increase the number of bootstraps to get more fine-grained p-values.')
        if self.use_bootstrap():
            print('Warning: I noticed that bootstrap breaks p-values, by a tiny bit, but enough to make them not work. Use at your own risk or set permutation_only=True.')

    def get_sample(self, y_pred_1_array, y_pred_2_array, bootstrap_indices_1, from_dataset_1):
        combined_array = numpy.concatenate([y_pred_1_array, y_pred_2_array])
        combined_y_true = numpy.concatenate([self.y_true, self.y_true_2])
        adjusted_indices = bootstrap_indices_1 + (~from_dataset_1) * len(y_pred_1_array)
        return combined_array[adjusted_indices], combined_y_true[adjusted_indices]

    def pick_samples(self):
        combined_indices = numpy.arange(len(self.y_true) + len(self.y_true_2))
        if self.use_permutation() and self.use_bootstrap():
            combined_indices = numpy.random.choice(combined_indices, len(combined_indices), replace=False)
            combined_indices_1 = combined_indices[:len(self.y_true)]
            combined_indices_2 = combined_indices[len(self.y_true):]
        elif self.use_permutation() and not self.use_bootstrap():
            if self.paired:
                flip_at = numpy.random.random(len(combined_indices)) < 0.5
                combined_indices[flip_at] = (combined_indices[flip_at] + len(self.y_true)) % len(combined_indices)
            else:
                numpy.random.shuffle(combined_indices)
            combined_indices_1 = combined_indices[:len(self.y_true)]
            combined_indices_2 = combined_indices[len(self.y_true):]
        elif not self.use_permutation() and self.use_bootstrap():  # sample_indices if they were all from one large array y_true+y_true_2
            combined_indices_1 = combined_indices[:len(self.y_true)]
            combined_indices_2 = combined_indices[len(self.y_true):]
            combined_indices_1 = numpy.random.choice(combined_indices_1, len(combined_indices_1), replace=True)
            combined_indices_2 = numpy.random.choice(combined_indices_2, len(combined_indices_2), replace=True)
        elif not self.use_permutation() and not self.use_bootstrap():
            raise RuntimeError('This should not happen')
        else:
            raise RuntimeError('This should not happen')

        separated_indices_1 = combined_indices_1 - (combined_indices_1 >= len(self.y_true)) * len(self.y_true)
        from_dataset_1_1 = combined_indices_1 < len(self.y_true)
        if self.paired:
            from_dataset_1_2 = ~from_dataset_1_1
            separated_indices_2 = separated_indices_1
        else:
            separated_indices_2 = combined_indices_2 - (combined_indices_2 >= len(self.y_true)) * len(self.y_true)
            from_dataset_1_2 = combined_indices_2 < len(self.y_true)
        return separated_indices_1, separated_indices_2, from_dataset_1_1, from_dataset_1_2

    def use_permutation(self):
        return not self.skip_permutation

    def use_bootstrap(self):
        return not self.permutation_only


def _compare_models_cache_key(metric, model_1: str, model_2: str, n_bootstraps, verbose, y_preds, y_true, two_sided, permutation_only, _extra_cache_key):
    result = (len(y_true), metric.__name__, model_1, model_2, n_bootstraps, two_sided, _extra_cache_key)
    if permutation_only:
        result = result + (permutation_only,)
    return result


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
                 permutation_only=True,
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
        return BootstrapModelComparison(
            self.n_bootstraps,
            self.metric,
            two_sided=self.two_sided,
            verbose=self.verbose,
            paired=True,
            permutation_only=self.permutation_only
        ).compare(self.y_true, self.y_preds[model_1], self.y_preds[model_2])


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
