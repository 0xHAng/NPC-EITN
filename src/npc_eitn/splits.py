"""Recorded fold construction and weighted concordance."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from ._ridge import horizon_labels_weights
SEED=20261015
def make_splits(frame, horizon, n_splits, seed):
    labels, _ = horizon_labels_weights(frame, horizon)
    strata = np.where(labels == 1, 1, 0)
    if np.bincount(strata, minlength=2).min() < n_splits:
        raise ValueError("Too few target events for the specified cross-validation")
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(
        np.zeros(len(frame)), strata
    ))


def fast_auc(y, probability, weight):
    """Weighted concordance, including half credit for tied probabilities."""
    positive, negative = weight * (y == 1), weight * (y == 0)
    if positive.sum() <= 0 or negative.sum() <= 0:
        return np.nan
    order = np.argsort(probability, kind="stable")
    sorted_p = probability[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p) != 0) + 1]
    positives = np.add.reduceat(positive[order], starts)
    negatives = np.add.reduceat(negative[order], starts)
    concordant = np.sum(positives * (np.cumsum(negatives) - 0.5 * negatives))
    return float(concordant / (positives.sum() * negatives.sum()))

