

# calculates lower bounds for a given binary subsequence (represented as #0 and #1) with either
# Clopper-Pearson (cp) method that has prior Beta(~0, 1)
def get_HTLB_CP_estimate(nr_of_zeros, nr_of_ones, confidence=0.99):
    return beta.ppf(1-confidence, nr_of_ones, nr_of_zeros + 1) if nr_of_ones > 0 else 0



# method that calculates the full HTLB cautious calibration map for Clopper-Pearson
def HTLB_CP_cautious_calibration(binary_sequence, subsequence_length=1000, confidence=0.99):
    lower_bounds = []

    for i in range(1, len(binary_sequence) + 1):
        if i < subsequence_length:
            lower_bounds.append(0)
        else:
            subsequence = binary_sequence[i-subsequence_length:i]
            nr_of_ones = sum(subsequence)
            nr_of_zeros = len(subsequence) - nr_of_ones
            assert len(subsequence) == subsequence_length
            lb = get_HTLB_estimate(nr_of_zeros, nr_of_ones, confidence=confidence, bound_type=bound_type)
            lower_bounds.append(lb)
    return lower_bounds