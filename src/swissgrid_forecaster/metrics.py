"""Unweighted metrics; missing/nonfinite inputs fail explicitly. Bias = forecast - truth."""
from math import isfinite, sqrt


def _pairs(y, prediction):
    y, prediction = tuple(y), tuple(prediction)
    if not y or len(y) != len(prediction):
        raise ValueError('nonempty aligned observations required')
    if any(v is None or not isfinite(v) for v in (*y, *prediction)):
        raise ValueError('missing/nonfinite metric input')
    return tuple(zip(y, prediction))


def mae(y, prediction):
    pairs = _pairs(y, prediction)
    return sum(abs(p-t) for t, p in pairs)/len(pairs)


def rmse(y, prediction):
    pairs = _pairs(y, prediction)
    return sqrt(sum((p-t)**2 for t, p in pairs)/len(pairs))


def bias(y, prediction):
    pairs = _pairs(y, prediction)
    return sum(p-t for t, p in pairs)/len(pairs)


def pinball(y, prediction, quantile):
    if not isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError('quantile must be inside (0, 1)')
    pairs = _pairs(y, prediction)
    return sum(max(quantile*(t-p), (quantile-1)*(t-p)) for t, p in pairs)/len(pairs)


def _intervals(lower, upper):
    pairs = _pairs(lower, upper)
    if any(lo > hi for lo, hi in pairs):
        raise ValueError('crossed interval')
    return pairs


def coverage(y, lower, upper):
    intervals = _intervals(lower, upper)
    pairs = _pairs(y, [lo for lo, _ in intervals])
    return sum(lo <= t <= hi for (t, _), (lo, hi) in zip(pairs, intervals))/len(pairs)


def sharpness(lower, upper):
    pairs = _intervals(lower, upper)
    return sum(hi-lo for lo, hi in pairs)/len(pairs)


def brier(y, probability):
    pairs = _pairs(y, probability)
    if any(t not in (0, 1) or not 0 <= p <= 1 for t, p in pairs):
        raise ValueError('Brier needs binary truth and probabilities in [0, 1]')
    return sum((p-t)**2 for t, p in pairs)/len(pairs)


def crps(*args, **kwargs):
    raise NotImplementedError('CRPS is scaffolded; specify distribution representation first')


def wis(*args, **kwargs):
    raise NotImplementedError('WIS is scaffolded; specify interval levels and weights first')
