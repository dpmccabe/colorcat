import dataclasses

import numpy as np
import pytest

from colorcat._constraints import FEASIBLE_INSET, Constraints, get_constraints


@pytest.mark.parametrize("name", ["white", "black"])
def test_builtin_sets_are_valid(name):
    c = get_constraints(name)
    assert c.background == name
    assert 0.0 <= c.lightness[0] < c.lightness[1] <= 1.0
    assert c.hue == (0.0, 360.0)


def test_background_colors():
    np.testing.assert_allclose(get_constraints("white").background_lab, [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(get_constraints("black").background_lab, [0.0, 0.0, 0.0], atol=1e-6)


def test_lightness_band_is_on_the_far_side_of_mid_grey():
    assert get_constraints("white").lightness[1] < 0.8
    assert get_constraints("black").lightness[0] > 0.5
    assert get_constraints("white").lightness[1] <= get_constraints("black").lightness[1]


def test_inset_shrinks_lightness():
    c = get_constraints("white")
    assert c.lightness_bounds == (c.lightness[0] + FEASIBLE_INSET, c.lightness[1] - FEASIBLE_INSET)


def test_unknown_backgrounds_raise_listing_allowed_values():
    for bad in ["red", "#ffffff", "White", "", None, 0]:
        with pytest.raises(ValueError, match="'white', 'black'"):
            get_constraints(bad)  # pyrefly: ignore[bad-argument-type]


def test_constraints_are_frozen():
    c = get_constraints("white")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.lightness = (0.0, 1.0)  # type: ignore[misc]


def test_invalid_ranges_rejected():
    base = dict(background="x", background_srgb=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="lightness"):
        Constraints(**base, lightness=(0.8, 0.4))
