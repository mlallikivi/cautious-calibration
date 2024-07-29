
import numpy as np




# Recursive function to create a calibration map according to https://link.springer.com/chapter/10.1007/978-3-030-46147-8_7
# Essentially we choose a random index and assing to it a probability that is uniformly randomly chosen from the range between lower and upper.
# Then we recursively call the function for the left and right subarrays of the index, changing the lower and upper parts so that the final sequence is monotonic.
def generate_probabilities(n, lower=0, upper=1):
    if n == 0:
        return []
    index = np.random.randint(1, (n + 1))
    value = np.random.uniform(lower, upper)
    left = generate_probabilities(index - 1, lower, value)
    right = generate_probabilities(n - index, value, upper)
    return left + [value] + right


# Generate a monotonic true calibration map of length N with all values between lowest_prob and highest_prob.
def generate_true_calibration_map(N, seed, lowest_prob=0, highest_prob=1):
    np.random.seed(seed)
    underlying_probs = generate_probabilities(N, lower=lowest_prob, upper=highest_prob)
    return underlying_probs



# Generate a binary label sequence based on the underlying probabilities
def generate_binary_label_sequence(underlying_probs, seed):
    np.random.seed(seed)
    sequence = []
    for p in underlying_probs:
        sequence.append(np.random.binomial(1, p))
    return sequence