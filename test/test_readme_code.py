import unittest


class TestReadmeCode(unittest.TestCase):
    def test_comparing_two_models(self):
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
        assert isinstance(result.p_value, float)

    def test_pairwise_comparisons(self):
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
            n_iterations=999,
            metric=average_log_likelihood,
            two_sided=True,
        )
        pairwise_test = PairwiseModelComparison(to_file='out.png',
                                                model_names=list(y_preds),
                                                y_true=y_true,
                                                y_preds=y_preds,
                                                test=test)

        # run the tests
        pairwise_test.plot_pairwise_comparisons()