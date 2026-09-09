"""Deterministic empirical histogram payloads for the forecast frontend."""
from dataclasses import dataclass
from math import isfinite
from typing import Iterable


def empirical_quantile(values: Iterable[float], probability: float) -> float:
    """Return a linearly interpolated empirical quantile."""
    data = tuple(sorted(float(value) for value in values))
    if not data or not all(isfinite(value) for value in data):
        raise ValueError("nonempty finite samples required")
    if not 0 <= probability <= 1 or not isfinite(probability):
        raise ValueError("probability must be between zero and one")
    if len(data) == 1:
        return data[0]
    position = probability * (len(data) - 1)
    lower = int(position)
    upper = min(lower + 1, len(data) - 1)
    fraction = position - lower
    return data[lower] + fraction * (data[upper] - data[lower])


@dataclass(frozen=True, slots=True)
class HistogramPayload:
    bin_edges: tuple[float, ...]
    bin_centers: tuple[float, ...]
    probabilities: tuple[float, ...]
    total_probability: float
    median: float
    central_50_interval: tuple[float, float]
    central_90_interval: tuple[float, float]

    def __post_init__(self):
        object.__setattr__(self, "bin_edges", tuple(float(value) for value in self.bin_edges))
        object.__setattr__(self, "bin_centers", tuple(float(value) for value in self.bin_centers))
        object.__setattr__(self, "probabilities", tuple(float(value) for value in self.probabilities))
        object.__setattr__(self, "central_50_interval", tuple(float(value) for value in self.central_50_interval))
        object.__setattr__(self, "central_90_interval", tuple(float(value) for value in self.central_90_interval))
        if len(self.bin_edges) != len(self.probabilities) + 1:
            raise ValueError("histogram edges must bracket probabilities")
        if len(self.bin_centers) != len(self.probabilities) or not self.probabilities:
            raise ValueError("histogram centers must match probabilities")
        if any(not isfinite(value) for value in (*self.bin_edges, *self.bin_centers,
                                                  *self.probabilities, self.total_probability,
                                                  self.median, *self.central_50_interval,
                                                  *self.central_90_interval)):
            raise ValueError("histogram contains nonfinite values")
        if any(a >= b for a, b in zip(self.bin_edges, self.bin_edges[1:])):
            raise ValueError("histogram edges must be strictly increasing")
        if any(value < 0 for value in self.probabilities):
            raise ValueError("histogram probabilities must be nonnegative")
        if abs(sum(self.probabilities) - self.total_probability) > 1e-12:
            raise ValueError("histogram total does not match bins")
        if abs(self.total_probability - 1.0) > 1e-9:
            raise ValueError("histogram probability must sum to one")
        if not (self.central_50_interval[0] <= self.median <= self.central_50_interval[1]):
            raise ValueError("median outside central 50 interval")
        if not (self.central_90_interval[0] <= self.central_50_interval[0]
                <= self.central_50_interval[1] <= self.central_90_interval[1]):
            raise ValueError("intervals are not nested")

    @classmethod
    def from_samples(cls, samples: Iterable[float], *, bins: int = 10) -> "HistogramPayload":
        values = tuple(sorted(float(value) for value in samples))
        if not values or not all(isfinite(value) for value in values):
            raise ValueError("nonempty finite samples required")
        if type(bins) is not int or bins <= 0:
            raise ValueError("bins must be a positive integer")
        bins = min(bins, len(values))
        low, high = values[0], values[-1]
        if low == high:
            half_width = 0.5 if low == 0 else max(abs(low) * 0.01, 0.5)
            edges = (low - half_width, low + half_width)
        else:
            width = (high - low) / bins
            edges = tuple(low + index * width for index in range(bins + 1))
            edges = (*edges[:-1], high)
        counts = [0] * (len(edges) - 1)
        for value in values:
            index = len(counts) - 1 if value == edges[-1] else min(
                len(counts) - 1, int((value - edges[0]) / (edges[1] - edges[0])))
            counts[index] += 1
        probabilities = tuple(count / len(values) for count in counts)
        centers = tuple((a + b) / 2 for a, b in zip(edges, edges[1:]))
        return cls(edges, centers, probabilities, sum(probabilities),
                   empirical_quantile(values, .5),
                   (empirical_quantile(values, .25), empirical_quantile(values, .75)),
                   (empirical_quantile(values, .05), empirical_quantile(values, .95)))

    def to_dict(self) -> dict:
        return {
            "bin_edges": list(self.bin_edges),
            "bin_centers": list(self.bin_centers),
            "probabilities": list(self.probabilities),
            "total_probability": self.total_probability,
            "median": self.median,
            "central_50_interval": list(self.central_50_interval),
            "central_90_interval": list(self.central_90_interval),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "HistogramPayload":
        return cls(tuple(payload["bin_edges"]), tuple(payload["bin_centers"]),
                   tuple(payload["probabilities"]), payload["total_probability"],
                   payload["median"], tuple(payload["central_50_interval"]),
                   tuple(payload["central_90_interval"]))
