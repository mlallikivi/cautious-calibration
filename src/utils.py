from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression

# calculates lower bounds for a given binary subsequence (represented as #0 and #1) with
# Clopper-Pearson (cp) method that has prior Beta(~0, 1)
def get_CP_estimate(nr_of_zeros, nr_of_ones, confidence=0.99):
    return beta.ppf(1-confidence, nr_of_ones, nr_of_zeros + 1) if nr_of_ones > 0 else 0

def get_isotonic_scores(scores, sequence):
    ir = IsotonicRegression()
    ir.fit(scores, sequence)
    return ir.transform(scores)