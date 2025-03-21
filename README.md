# Comparing (statistical) models

In machine learning and related fields, we often encounter the issue of having trained two models but being unsure if our test set is large enough to show that one model is better than the other.
This repository implements a methods of hypothesis testing to check if two models achieve different results according to some specified metric, against the null hypothesis that the two models' outputs come from the same distribution.

If you want to compare a single pair of trained models, use `permutation_based_model_comparison()`: A standard way of conducting this test, accepted by statisticians and [commonly used.](https://aclanthology.org/2022.naacl-main.360.pdf)
We also have `PairwisePermutationModelComparison`: Extensions to plot many pairwise comparisons (of more than two models) to get an overview of the data.
If you are instead interested in estimating confidence intervals for the metric value of a single model, I recommend `bootstrap_confidence_interval()`.

# recommended workflow:
0. Build, train, and optimize the models that you want to compare.
1. Collect ground truth for test data as a numpy array or pandas DataFrame, with the first axis as the sample axis (one row per observation).
2. Collect model outputs on test data (e.g., segmentation masks, class probabilities, etc.), ordering rows the same way as the ground truth.
3. (Optional) If using cross-validation, combine outputs on the respective validation sets into one large ground truth array and do the same for model outputs.
4. Define a Python function as a metric, taking the ground truth as the first argument and the predictions as the second argument.
5. Determine the number of iterations (more is better, but with diminishing returns. It also makes the test take longer. I recommend 999, more if you have the computation time, less if you want to do thousands of comparisons or if computing the metric is expensive).
6. Run `permutation_based_model_comparison()` to get a p-value for comparing the models.

# Examples and Simulations with artificial data
Examples and simulations with artificial data can be found in the unit tests in `test_hypothesis_test.py`.

# Common pitfalls to avoid
- If performing many tests, a p-value of 0.05 is not significant. Look up the Bonferroni correction.
- Do not repeat the same observation multiple times in the dataset. For example, if classifying cancer outcomes, you should never have two rows for the same patient in one array, as this will systematically reduce p-values even if the model remains the same.
- The pairwise comparison classes cache some results on disk to speed up computations. In rare cases, you may need to clear the cache by deleting the `.cache` directory.
- All the usual pitfalls for hypothesis testing apply here, obviously: [Check out this paper](https://www.ohri.ca/newsroom/seminars/SeminarUploads/1829%5CSuggested%20Reading%20-%20Nov%203,%202014.pdf)