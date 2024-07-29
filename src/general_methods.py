import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration
from utils import get_isotonic_scores



### ISOCAL ###

# This method calculates the isotonic calibration map, applying platt scaling to the label sequence before.
# This ensures that the isotonic calibration is a little bit more cautious and obtains from predicting 0 and 1 probabilities.
def isocal(probs, sequence):
    nr_of_ones = np.sum(sequence)
    nr_of_zeros = len(sequence) - nr_of_ones
    platt_1 = 1 - (1 / (nr_of_ones + 2))
    platt_0 = 0 + (1 / (nr_of_zeros + 2))
    platt_sequence = [platt_1 if x == 1 else platt_0 for x in sequence]
    scores = get_isotonic_scores(probs, platt_sequence)
    return scores
    

### LOGCAL ###

# This function calculates the logistic calibration map
def logcal(probs, sequence):
    # Two important notes
    # First - I don't know if this is using platt's correction, I think that probably not since it is the general logistic regression, but I haven't checked
    # If the labels are negatively correlated with the scores then the function learned will be decreasing. So it probably won't happen, but in some weird cases can happen.
    clf = LogisticRegression(random_state=1).fit(np.array(probs).reshape(-1, 1), sequence)
    return clf.predict_proba(np.array(probs).reshape(-1, 1))[:,1]


### BETACAL ###

def betacal(probs, sequence):
    bc = BetaCalibration(parameters="abm")
    bc.fit(np.array(probs).reshape(-1, 1), sequence)
    return bc.predict(probs)


### SVA ###

# The simplified Venn-Abers method (SVA).
# To get a cautious calibration map with this method we only calculate the lower estimates of SVA (classically we can calculate both lower and upper).
# For each element in our sequence we either use the default isotonic calibration result of that element if the label was 0
# or if the label was 1 we replace it with 0 and calculate a new isotonic calibration map and take the corresponding elemnt from it.
def sva(probs, sequence):
    p0 = []
    ir = IsotonicRegression()
    ir.fit(probs, sequence)
    default_iso = ir.predict(probs)
    for i in range(len(sequence)):
        if sequence[i] == 0:
            p0.append(default_iso[i])
        else:
            sequence[i] = 0
            ir = IsotonicRegression()
            ir.fit(probs, sequence)
            p0.append(ir.predict([probs[i]])[0])
            sequence[i] = 1
    return p0
