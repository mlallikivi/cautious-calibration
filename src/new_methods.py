from utils import get_CP_estimate
from bisect import bisect_left


### HTLB+CP ###

# method that calculates the full HTLB cautious calibration map for Clopper-Pearson
def htlb_cp(binary_sequence, subsequence_length=1000, confidence=0.99):
    lower_bounds = []

    for i in range(1, len(binary_sequence) + 1):
        if i < subsequence_length:
            lower_bounds.append(0)
        else:
            subsequence = binary_sequence[i-subsequence_length:i]
            nr_of_ones = sum(subsequence)
            nr_of_zeros = len(subsequence) - nr_of_ones
            assert len(subsequence) == subsequence_length
            lb = get_CP_estimate(nr_of_zeros, nr_of_ones, confidence=confidence)
            lower_bounds.append(lb)
    return lower_bounds


### HTLB+MAXCP ###

# calculating max statistic knowing the positions of zeros in the current subsequence
def calculate_max_statistic(positions_of_zeros, lb_map_fixed_fperc, min_size=2):
    
    max_g = -1
    nr_of_zeros = 0
    sequence_length = positions_of_zeros[-1]
    

    # we look at sequences ending before 0's, because only those are potential candidates for maximum bounds
    # we don't include the last position, because that would be an empty sequence
    for idx_of_zero in list(reversed(positions_of_zeros))[1:]:
        
        start_pos = idx_of_zero + 1 # we start immediately after the 0
        end_pos = sequence_length # we end at the final position of the sequence
        
        nr_of_ones = (end_pos - start_pos) - nr_of_zeros # we calculate how many 1's there are in this subsequence
        
        # if the sequence is empty (first sequence), we skip this iteration
        if nr_of_zeros + nr_of_ones < min_size:
            nr_of_zeros += 1
            continue
        # g is lower bound for fixed m and statistic function that is the percentage of 1's
        # we are using fixed q atm
        else:
            m = nr_of_zeros + nr_of_ones
            current_g = lb_map_fixed_fperc[m][nr_of_ones]
        
        if current_g > max_g:
            max_g = current_g
        
        nr_of_zeros += 1
        
    return max_g


# from calculated statistic to corresponding lower bound
def map_value_to_lower_bound_imprecise(value, lb_map):
    keys = list(lb_map.keys())
    ind = bisect_left(keys, value)
    corresponding_key = keys[ind - 1]
    if ind - 1 < 0:
        return 0
    return lb_map[corresponding_key]


# from subsequence -> statistic -> lower bound
def get_max_statistic_lower_bound(seq, lb_map_fixed_perc, lb_map, win_min=100):
    zero_indices = np.where(np.array(seq) == 0)[0]
    positions_of_zeros = [-1] + list(zero_indices) + [len(seq)]
    statistic = calculate_max_statistic(positions_of_zeros, lb_map_fixed_perc, min_size=win_min)
    lower_bound = map_value_to_lower_bound_imprecise(statistic, lb_map)
    return lower_bound


# from whole sequence -> lower bound map
def htlb_maxcp(binary_sequence, lb_map_fixed_perc, lb_map, window_size=2000, w_min=100):
    lower_bound_map = []

    for i in range(1, len(binary_sequence) + 1):
        if i < window_size:
            lower_bound_map.append(0)
        else:
            window = binary_sequence[i-window_size:i]
            lb = get_max_statistic_lower_bound(window, lb_map_fixed_perc, lb_map, win_min=w_min)
            lower_bound_map.append(lb)
    return lower_bound_map
