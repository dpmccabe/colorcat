"""Baselines that palette() is measured against (shared by the benchmark and quality tests)."""

from __future__ import annotations

import numpy as np

from colorcat._constraints import Constraints
from colorcat._distance import pairwise_distance
from colorcat._energy import min_distance_with_anchors
from colorcat._feasible import sample
from colorcat._format import hex_to_oklab, quantize
from colorcat._oklab import srgb_to_oklab

# Named categorical palettes (Okabe-Ito without black; Tableau 10; Matplotlib tab10 is the same as Tableau's old 10).
NAMED = {
    "okabe-ito": ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7"],
    "tableau10": ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                  "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"],
    "set1": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
             "#ffff33", "#a65628", "#f781bf", "#999999"],
}


def named_min_distance(name: str, n: int, background_lab: np.ndarray) -> float:
    """Min ΔE_OK (incl. background) of the first ``n`` colors of a named palette."""
    lab = np.array([hex_to_oklab(c) for c in NAMED[name][:n]])
    return min_distance_with_anchors(lab, background_lab[None, :])


def random_min_distance(n: int, c: Constraints, draws: int = 50, seed: int = 0) -> float:
    """Median over ``draws`` of the min distance of ``n`` random feasible 8-bit colors."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        lab = srgb_to_oklab(quantize(sample(rng, n, c), c) / 255.0)
        scores.append(min_distance_with_anchors(lab, c.background_lab[None, :]))
    return float(np.median(scores))


def farthest_point_min_distance(n: int, c: Constraints, candidates: int = 20000, seed: int = 0) -> float:
    """Greedy farthest-point sampling from the background over random feasible candidates."""
    rng = np.random.default_rng(seed)
    cand = srgb_to_oklab(quantize(sample(rng, candidates, c), c) / 255.0)
    chosen = []
    nearest = pairwise_distance(cand, c.background_lab[None, :])[:, 0]
    for _ in range(n):
        i = int(np.argmax(nearest))
        chosen.append(i)
        nearest = np.minimum(nearest, pairwise_distance(cand, cand[[i]])[:, 0])
    return min_distance_with_anchors(cand[chosen], c.background_lab[None, :])
