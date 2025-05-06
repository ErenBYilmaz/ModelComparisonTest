# Comparing (statistical) models

In machine learning and related fields, we often encounter the issue of having trained two models but being unsure if our test set is large enough to show that one model is better than the other.
This repository implements a methods of hypothesis testing to check if two models achieve different results according to some specified metric, against the null hypothesis that the two models' outputs come from the same distribution.

If you want to compare a single pair of trained models evaluated on the same test set, use `PermutationModelComparisonPaired()`: 
A standard way of conducting this test, accepted by statisticians and [commonly used.](https://aclanthology.org/2022.naacl-main.360.pdf)
We also have `PairwiseModelComparison`: An extension to plot many pairwise comparisons (of more than two models) to get an overview of the data.
If you are instead interested in estimating confidence intervals for the metric value of a single model, I recommend `bootstrap_confidence_interval()`.

# Example

### Comparing two model outputs:
```python
import numpy, scipy

# simulate some model outputs and ground truth
y_true = numpy.array(numpy.random.randint(0, 2, size=200))
y_pred_1 = scipy.special.expit(numpy.random.normal(size=(200, 1)))
y_pred_2 = scipy.special.expit(numpy.random.normal(size=(200, 1)))

# define metric
def average_log_likelihood(y_true, y_pred, epsilon=1e-7):
    return numpy.mean(y_true * numpy.log(y_pred + epsilon) + (1 - y_true) * numpy.log(1 - y_pred + epsilon))

# create the test object
from hypothesis_test import PermutationModelComparisonPaired
test = PermutationModelComparisonPaired(
    n_iterations=9999,
    metric=average_log_likelihood,
    two_sided=True,
    plot_histogram_to_file='out.png'
)

# run the test
result = test.compare(y_true, y_pred_1, y_pred_2)
print(result.p_value)
print(result.model_1_metric_larger)
```

Other available tests: `PermutationModelComparisonPaired`, `PermutationModelComparisonUnpaired`, `BootstrapModelComparisonUnpaired`, `BootstrapModelComparisonUnpaired`.

### Comparing many models outputs in a pairwise way (beware of [multiple testing](https://en.wikipedia.org/wiki/Multiple_comparisons_problem)):
```python
import numpy, scipy

# simulate some model outputs and ground truth
y_true = numpy.array(numpy.random.randint(0, 2, size=200))
y_preds = {
    f'model_{i}': scipy.special.expit(numpy.random.normal(size=(200, 1)))
    for i in range(10)
}

# define metric
def average_log_likelihood(y_true, y_pred, epsilon=1e-7):
    return numpy.mean(y_true * numpy.log(y_pred + epsilon) + (1 - y_true) * numpy.log(1 - y_pred + epsilon))

# create the test object
from hypothesis_test import PermutationModelComparisonPaired
from pairwise_model_comparison import PairwiseModelComparison
test = PermutationModelComparisonPaired(
    n_iterations=9999,
    metric=average_log_likelihood,
    two_sided=True,
)
pairwise_test = PairwiseModelComparison(to_file = 'out.png',
                                        model_names=list(y_preds),
                                        y_true=y_true,
                                        y_preds=y_preds,
                                        test=test)

# run the tests
pairwise_test.plot_pairwise_comparisons()
```


# recommended workflow:
0. Build, train, and optimize the models that you want to compare.
1. Collect ground truth for test data as a numpy array or pandas DataFrame, with the first axis as the sample axis (one row per observation).
2. Collect model outputs on test data (e.g., segmentation masks, class probabilities, etc.), ordering rows the same way as the ground truth.
3. (Optional) If using cross-validation, combine outputs on the respective validation sets into one large ground truth array and do the same for model outputs.
4. Define a Python function as a metric, taking the ground truth as the first argument and the predictions as the second argument.
5. Determine the number of iterations (more is better, but with diminishing returns. It also makes the test take longer. I recommend 999, more if you have the computation time, less if you want to do thousands of comparisons or if computing the metric is expensive).
6. Run a `PermutationModelComparisonPaired` to get a p-value for comparing the models.

# Examples and Simulations with artificial data
Examples and simulations with artificial data can be found in the unit tests in `test_hypothesis_test.py`.

# Common pitfalls to avoid
- If performing many tests, a p-value of 0.05 is not significant. Look up the Bonferroni correction.
- Do not repeat the same observation multiple times in the dataset. For example, if classifying cancer outcomes, you should never have two rows for the same patient in one array, as this will systematically reduce p-values even if the model remains the same.
- The pairwise comparison classes cache some results on disk to speed up computations. In rare cases, you may need to clear the cache by deleting the `.cache` directory.
- All the usual pitfalls for hypothesis testing apply here, obviously: [Check out this paper](https://www.ohri.ca/newsroom/seminars/SeminarUploads/1829%5CSuggested%20Reading%20-%20Nov%203,%202014.pdf)