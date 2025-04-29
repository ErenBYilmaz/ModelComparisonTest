import math
import os
import unittest

import numpy


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


class TestLikelihoodFunction(unittest.TestCase):
    def test_likelihood_function(self):
        import lifelines
        import pandas
        rossi_path = os.path.join(os.path.dirname(lifelines.__file__), 'datasets', 'rossi.csv')
        rossi = pandas.read_csv(rossi_path)

        rossi = rossi.sample(frac=1.0, random_state=25)  # ensures the reproducibility of the example
        train_rossi = rossi.iloc[:400]
        test_rossi = rossi.iloc[400:]

        time_column = 'week'
        event_column = 'arrest'
        cph_l1 = lifelines.CoxPHFitter(penalizer=0.1, l1_ratio=1.).fit(train_rossi, time_column, event_column)
        cph_l2 = lifelines.CoxPHFitter(penalizer=0.1, l1_ratio=0.).fit(train_rossi, time_column, event_column)

        y_true = test_rossi[[time_column, event_column]].values
        y_pred_1 = numpy.array(cph_l1.predict_log_partial_hazard(test_rossi))
        y_pred_2 = numpy.array(cph_l2.predict_log_partial_hazard(test_rossi))

        print(cph_l1.score(test_rossi))
        print(avg_likelihood(y_true, y_pred_1))
        print(cph_l2.score(test_rossi))
        print(avg_likelihood(y_true, y_pred_2))

        assert math.isclose(avg_likelihood(y_true, y_pred_1), cph_l1.score(test_rossi))
        assert math.isclose(avg_likelihood(y_true, y_pred_2), cph_l2.score(test_rossi))
