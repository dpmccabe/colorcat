import warnings

import numpy as np
import pytest

import colorcat
from colorcat import min_distance, palette
from colorcat._constraints import get_constraints
from colorcat._format import hex_to_oklab
from colorcat._oklab import oklab_to_oklch

HEX = r"#[0-9a-f]{6}"


def lch(colors):
    return oklab_to_oklch(np.array([hex_to_oklab(c) for c in colors]))


@pytest.mark.parametrize("n", [1, 2, 5, 12])
def test_returns_n_distinct_hex_colors(n):
    colors = palette(n, restart_count=2)
    assert len(colors) == n == len(set(colors))
    assert all(__import__("re").fullmatch(HEX, c) for c in colors)


def test_n_zero_is_empty_and_public_exports():
    assert palette(0) == []
    assert set(colorcat.__all__) == {"palette", "min_distance"}


def test_same_seed_is_reproducible_and_seed_matters():
    assert palette(6, restart_count=3, seed=1) == palette(6, restart_count=3, seed=1)
    assert palette(6, restart_count=1, seed=1) != palette(6, restart_count=1, seed=2)


def test_default_seed_is_zero():
    assert palette(4, restart_count=2) == palette(4, restart_count=2, seed=0)


def test_numpy_integers_are_accepted():
    assert palette(np.int64(3), restart_count=np.int32(1)) == palette(3, restart_count=1)


@pytest.mark.parametrize("background", ["white", "black"])
def test_colors_respect_the_built_in_lightness_range(background):
    c = get_constraints(background)
    L = lch(palette(15, background=background, restart_count=2))[:, 0]
    assert (L >= c.lightness[0]).all() and (L <= c.lightness[1]).all()


def test_background_picks_light_or_dark_band():
    on_white = lch(palette(8, background="white", restart_count=2))[:, 0]
    on_black = lch(palette(8, background="black", restart_count=2))[:, 0]
    assert on_white.max() <= 0.75 < 0.6 + 0.2 and on_black.min() >= 0.6
    assert on_white.mean() < on_black.mean()


def test_first_color_is_farthest_from_background():
    colors = palette(7, restart_count=2)
    lab = np.array([hex_to_oklab(c) for c in colors])
    d = np.linalg.norm(lab - np.array([1.0, 0.0, 0.0]), axis=1)
    assert np.argmax(d) == 0


def test_more_restarts_never_worse():
    scores = [min_distance(palette(10, restart_count=k)) for k in (1, 4, 16)]
    assert scores == sorted(scores)


def test_min_distance_helper():
    assert min_distance(["#000000", "#ffffff"]) == pytest.approx(1.0, abs=1e-6)
    assert min_distance(["#ff0000"]) == float("inf")
    assert min_distance([]) == float("inf")
    assert min_distance(["#f00", "#ff0000"]) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        min_distance(["red"])


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        (dict(n=-1), ValueError),
        (dict(n=2.5), TypeError),
        (dict(n="3"), TypeError),
        (dict(n=True), ValueError),
        (dict(n=3, restart_count=0), ValueError),
        (dict(n=3, restart_count=1.5), TypeError),
        (dict(n=3, background="red"), ValueError),
        (dict(n=3, background="#ffffff"), ValueError),
        (dict(n=3, seed=1.5), TypeError),
    ],
)
def test_validation(kwargs, exc):
    with pytest.raises(exc):
        palette(**kwargs)


def test_background_error_lists_allowed_values():
    with pytest.raises(ValueError, match="'white', 'black'"):
        palette(3, background="grey")


def test_warns_when_colors_are_hard_to_tell_apart():
    with pytest.warns(UserWarning, match="hard to tell apart"):
        palette(1000, restart_count=1)


def test_no_warning_for_normal_palettes():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        palette(8, restart_count=2)
