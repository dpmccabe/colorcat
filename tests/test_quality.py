"""palette() should beat simple baselines on the min pairwise distance (after quantization)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
from baselines import farthest_point_min_distance, named_min_distance, random_min_distance  # noqa: E402

from colorcat import palette  # noqa: E402
from colorcat._constraints import get_constraints  # noqa: E402
from colorcat._energy import min_distance_with_anchors  # noqa: E402
from colorcat._format import hex_to_oklab  # noqa: E402


def score(colors, c):
    lab = np.array([hex_to_oklab(x) for x in colors])
    return min_distance_with_anchors(lab, c.background_lab[None, :])


@pytest.mark.parametrize("background", ["white", "black"])
@pytest.mark.parametrize("n", [3, 8, 16])
def test_beats_random_and_farthest_point_baselines(background, n):
    c = get_constraints(background)
    got = score(palette(n, background=background), c)
    assert got >= random_min_distance(n, c, draws=20)
    assert got >= farthest_point_min_distance(n, c, candidates=5000)


@pytest.mark.parametrize("name, n", [("okabe-ito", 7), ("tableau10", 10)])
def test_beats_named_palettes_on_white(name, n):
    c = get_constraints("white")
    assert score(palette(n), c) >= named_min_distance(name, n, c.background_lab)


def test_distance_falls_as_n_grows_roughly_monotonically():
    c = get_constraints("white")
    scores = [score(palette(n), c) for n in (2, 6, 12, 24)]
    assert scores == sorted(scores, reverse=True)
