"""Agreement measures for binary masks, and the dark threshold.

Written here from the definitions lunar-polar-traverse used, not imported from
it: Cohen's kappa, and the threshold maximising Youden's J for the rule "dark
where value < t", the candidates being the distinct values present, so the
threshold is always a value in the data, and ties going to the smallest.
"""
import numpy as np


def kappa(a, b):
    """Cohen's kappa and raw agreement for two boolean labellings."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    po = float(np.mean(a == b))
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1.0 - pa) * (1.0 - pb)
    return ((po - pe) / (1.0 - pe) if pe < 1.0 else float("nan")), po


def youden(values, positive):
    """(threshold, J, true positive rate, false positive rate) for "positive
    where value < threshold", maximising J = TPR - FPR."""
    v = np.asarray(values, float)
    positive = np.asarray(positive, bool)
    pos = np.sort(v[positive])
    neg = np.sort(v[~positive])
    cand = np.unique(v)
    tpr = np.searchsorted(pos, cand, side="left") / pos.size
    fpr = np.searchsorted(neg, cand, side="left") / neg.size
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(cand[i]), float(j[i]), float(tpr[i]), float(fpr[i])
