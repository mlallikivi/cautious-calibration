import numpy as np
import reliable_isotonic
from utils import get_CP_estimate, get_isotonic_scores


### ISOBINS+CP ###

# helper function to get groups from isotonic scores so we can find lower bounds for them later
def get_isotonic_groups(iso_scores, sequence):
    score_groups = []
    sequence_groups = []
    current_score_group = [iso_scores[0]]
    current_sequence_group = [sequence[0]]

    for i in range(1, len(iso_scores)):
        if iso_scores[i] == iso_scores[i-1]:
            current_score_group.append(iso_scores[i])
            current_sequence_group.append(sequence[i])
        else:
            score_groups.append(current_score_group)
            sequence_groups.append(current_sequence_group)
            current_score_group = [iso_scores[i]]
            current_sequence_group = [sequence[i]]

    score_groups.append(current_score_group)
    sequence_groups.append(current_sequence_group)

    return {"score_groups": score_groups, "sequence_groups": sequence_groups}


# This function calculates isobins+CP cautious calibration map. It first finds the isotonic calibration bins and the calculates the lower bound
# for each bin using the Clopper-Pearson method.
def isobins_cp(sequence, conf=0.99):

    # get iso scores, can use any increasing score vectors, because exact values of scores don't matter
    iso_scores = get_isotonic_scores(np.linspace(0, 1, len(sequence)), sequence)
    iso_groups = get_isotonic_groups(iso_scores, sequence)["sequence_groups"]

    lower_bound_map = []

    for group in iso_groups:
        ones = sum(group)
        zeros = len(group) - ones
        lb = get_CP_estimate(zeros, ones, confidence=conf)

        # repeat the chosen bound len(group) times
        lower_bound_map += [lb]*len(group)

    return lower_bound_map


### RCIR+CP ###

# This method is very similar to the isobins+CP, but uses RCIR algorithm described here https://link.springer.com/chapter/10.1007/978-3-030-75762-5_46
# to find potentially larger and more stable bins where the lower bound estimates would be more reliable.
def rcir_cp(probs, sequence, d=0.7, credible_level=.99):
    rcir_model = reliable_isotonic.train_rcir(sequence, probs,
               credible_level=credible_level, d=d, y_min=0,
               y_max=1, merge_criterion='auc_roc')

    bin_sizes = rcir_model["bin summary"][1]

    sequence_in_bins = []

    start = 0
    for size in bin_sizes:
        sequence_in_bins.append(sequence[start:start+size])
        start += size

    lb_values = []
    for i in range(len(sequence_in_bins)):
        nr_of_ones = sum(sequence_in_bins[i])
        nr_of_zeros = len(sequence_in_bins[i]) - nr_of_ones
        lb = get_CP_estimate(nr_of_zeros, nr_of_ones, confidence=credible_level)
        lb_values += [lb]*len(sequence_in_bins[i])

    return lb_values