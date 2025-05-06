import itertools
import math
from typing import List, Callable, Dict

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from tqdm import tqdm

from comparison_result import ComparisonResult
from hypothesis_test import HypothesisTest


class PairwiseModelComparison:
    def __init__(self,
                 to_file: str,
                 model_names: List[str],
                 verbose=0,
                 y_true: numpy.ndarray = None,
                 y_preds: Dict[str, numpy.ndarray] = None,
                 test: HypothesisTest = None,
                 plot_title_addition: str = None, ):
        self.verbose = verbose
        self.plot_title_addition = plot_title_addition
        self.to_file = to_file
        self.model_names = model_names
        self.y_true = y_true
        self.y_preds = y_preds
        self.test = test

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

    def metric_name(self):
        return type(self.test).__name__

    def compare_models(self, model_1, model_2):
        return self.test.compare(self.y_true, self.y_preds[model_1], self.y_preds[model_2])

