import warnings

import numpy as np
import pytest

import colorcat
from colorcat import min_distance, palette
from colorcat._constraints import get_constraints
from colorcat._distance import pairwise_distance
from colorcat._format import hex_to_oklab, hex_to_srgb
from colorcat._oklab import contrast_ratio, oklab_to_oklch, relative_luminance, srgb_to_linear

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
    assert palette(np.int64(3), restart_count=np.int32(1))  # pyrefly: ignore[bad-argument-type] == palette(3, restart_count=1)


@pytest.mark.parametrize("background", ["white", "black"])
@pytest.mark.parametrize("ratio", [1.0, 3.0, 4.5, 7.0])
def test_colors_meet_the_min_contrast(background, ratio):
    colors = palette(10, background=background, min_contrast=ratio, restart_count=2)
    y = relative_luminance(srgb_to_linear(np.array([hex_to_srgb(c) for c in colors])))
    yb = 1.0 if background == "white" else 0.0
    assert (contrast_ratio(y, yb) >= ratio - 1e-9).all()


def test_default_min_contrast_is_wcag_graphical_objects():
    assert palette(5, restart_count=1) == palette(5, min_contrast=3, restart_count=1)


def test_higher_contrast_costs_distance():
    low = min_distance(palette(8, min_contrast=1.5, restart_count=2))
    high = min_distance(palette(8, min_contrast=7.0, restart_count=2))
    assert high < low


@pytest.mark.parametrize("bad", [0.5, 16, float("nan")])
def test_invalid_min_contrast_raises(bad):
    with pytest.raises(ValueError, match="min_contrast"):
        palette(3, min_contrast=bad)


def test_background_picks_light_or_dark_colors():
    on_white = lch(palette(8, background="white", restart_count=2))[:, 0]
    on_black = lch(palette(8, background="black", restart_count=2))[:, 0]
    assert on_white.mean() < on_black.mean()


def test_first_color_is_farthest_from_background_and_its_opposite():
    colors = palette(7, restart_count=2)
    lab = np.array([hex_to_oklab(c) for c in colors])
    d = pairwise_distance(lab, get_constraints("white").reference_labs).min(axis=1)
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
        ({"n": -1}, ValueError),
        ({"n": 2.5}, TypeError),
        ({"n": "3"}, TypeError),
        ({"n": True}, ValueError),
        ({"n": 3, "restart_count": 0}, ValueError),
        ({"n": 3, "restart_count": 1.5}, TypeError),
        ({"n": 3, "background": "red"}, ValueError),
        ({"n": 3, "background": "#ffffff"}, ValueError),
        ({"n": 3, "seed": 1.5}, TypeError),
    ],
)
def test_validation(kwargs, exc):
    with pytest.raises(exc):
        palette(**kwargs)


def test_background_error_lists_allowed_values():
    with pytest.raises(ValueError, match="'white', 'black'"):
        palette(3, background="grey")  # pyrefly: ignore[bad-argument-type]


def test_warns_when_colors_are_hard_to_tell_apart():
    with pytest.warns(UserWarning, match="hard to tell apart"):
        palette(1000, restart_count=1)


def test_no_warning_for_normal_palettes():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        palette(8, restart_count=2)


@pytest.mark.parametrize("background", ["white", "black"])
def test_colors_stay_distinct_from_background_and_its_opposite(background):
    c = get_constraints(background)
    lab = np.array([hex_to_oklab(h) for h in palette(3, background=background, restart_count=2)])
    d = pairwise_distance(lab, c.reference_labs)
    # With 3 colors the closest pair is about 0.4 apart, so neither the background
    # nor its opposite (plot lines, text) may be much closer than that.
    assert d[:, 0].min() > 0.3
    assert d[:, 1].min() > 0.3


def test_presets_were_generated_with_the_default_settings():
    from colorcat._presets import load
    from colorcat.palette import DEFAULT_SEED

    data = load()
    assert data["seed"] == DEFAULT_SEED
    assert data["min_contrast"] == get_constraints("white").min_contrast


@pytest.mark.parametrize("background", ["white", "black"])
def test_defaults_return_the_saved_palette(background):
    from colorcat._presets import load

    for n, saved in load()["palettes"][background].items():
        assert palette(int(n), background=background) == saved
    # restart_count does not matter for a saved palette.
    assert palette(5, background=background, restart_count=1) == palette(5, background=background)


def test_saved_palette_respects_color_format():
    hexes = palette(4)
    assert palette(4, color_format="rgb") == [
        tuple(hex_to_srgb(c).__mul__(255).round().astype(int)) for c in hexes
    ]
    assert len(palette(4, color_format="oklab")[0]) == 3


def test_any_non_default_option_skips_the_saved_palette():
    saved = palette(3)
    for kwargs in ({"seed": 5}, {"seed": None}, {"min_contrast": 4.5}, {"cvd": "deutan"}):
        assert palette(3, restart_count=1, **kwargs) != saved  # pyrefly: ignore[bad-argument-type]
    assert saved[0] not in palette(3, exclude=saved[:1], restart_count=1)
    assert palette(3, fixed=saved[:1], restart_count=1) != saved


def test_sizes_without_a_saved_palette_are_optimized():
    assert len(palette(16, restart_count=1)) == 16
