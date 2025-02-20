# Comparing (statistical) models

In machine learning and related fields, we often have the issue that we trained two models, but are unsure if our test set is large enough to show that one model is better than the other.
This repository implements two methods of hypothesis testing to check if two models achieve different results according to some specified metric, against the null hypothesis that the two model's outputs come from the same distribution.

- `permutation_based_model_comparison()`: A standard way of conducting this test, accepted by statisticians and commonly used.
- `bootstrap_based_model_comparison()`: A bootstrap-based extension that works better for small sample-sizes then the permutation test in my simulations, but is not widely accepted in the literature (yet?).

We also have `PairwisePermutationModelComparison` and `PairwiseBootstrapModelComparison`: Extensions to plot many pairwise comparisons (of more than 2 models) to get an overview of the data.

# recommended workflow:
0. build/train/optimize the models that you want to compare.
1. Collect ground truth for test data as numpy array or pandas dataframe, first axis should be the sample axis (one row per observation)
2. Collect model outputs on test data (e.g. segmentation masks, class probabilities, ...), order rows the same way as the ground truth
3. (optional) if using cross-validation, combine outputs on the respective validation sets into one large ground truth array and do the same for model outputs
4. Define a python function as metric taking the ground truth as the first argument and the predictions as the second argument
5. Think of a number of iterations (more is better, but with diminishing returns. also makes the test take longer. I recommend 999, more if you have the computation time, less if you want to do thousands of comparisons or if computing the metric is expensive)
6. Run one of the methods mentioned above (e.g. `permutation_based_model_comparison()`)

# Examples and Simulations with artificial data
Examples and simulations with artificial data can be found by looking at the unit tests in `test_hypothesis_test.py`.

# Common pitfalls to avoid
- If doing many tests, a p-value of 0.05 is not significant. Look up Bonferroni correction.
- Do not repeat the same observation multiple times in the dataset. For example if classifying cancer outcomes, you should never have two rows for the same patient in one array, as this will systematically reduce p-values even if the model stays the same.
- The pairwise comparison classes do some caching of results on disk to speed up computations, in some rare cases you may need to clear the cache by deleting the `.cache` directory.
- All the usual pitfalls for hypothesis testing apply here, obviously: [Check out this paper](https://www.ohri.ca/newsroom/seminars/SeminarUploads/1829%5CSuggested%20Reading%20-%20Nov%203,%202014.pdf)
