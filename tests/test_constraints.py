import dataclasses

import numpy as np
import pytest

from colorcat._constraints import (
    DEFAULT_MIN_CONTRAST,
    FEASIBLE_INSET,
    MAX_MIN_CONTRAST,
    Constraints,
    get_constraints,
)


@pytest.mark.parametrize("name", ["white", "black"])
def test_builtin_sets_are_valid(name):
    c = get_constraints(name)
    assert c.background == name
    assert c.min_contrast == DEFAULT_MIN_CONTRAST == 3.0


def test_background_colors():
    np.testing.assert_allclose(get_constraints("white").background_lab, [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(get_constraints("black").background_lab, [0.0, 0.0, 0.0], atol=1e-6)


def test_opposite_and_reference_colors():
    white, black = get_constraints("white"), get_constraints("black")
    np.testing.assert_allclose(white.opposite_lab, black.background_lab, atol=1e-6)
    np.testing.assert_allclose(black.opposite_lab, white.background_lab, atol=1e-6)
    np.testing.assert_allclose(
        white.reference_labs, [white.background_lab, white.opposite_lab], atol=1e-12
    )


def test_luminance_range_follows_wcag_contrast():
    # (1 + 0.05) / (Y + 0.05) >= 3  =>  Y <= 0.3 on white; (Y + 0.05) / 0.05 >= 3  =>  Y >= 0.1 on black.
    assert get_constraints("white").luminance_range == pytest.approx((0.0, 0.3))
    assert get_constraints("black").luminance_range == pytest.approx((0.1, 1.0))
    assert get_constraints("white", 4.5).luminance_range[1] == pytest.approx(1.05 / 4.5 - 0.05)
    assert get_constraints("black", 7).luminance_range[0] == pytest.approx(0.05 * 7 - 0.05)
    assert get_constraints("white", 1).luminance_range == pytest.approx((0.0, 1.0))


def test_inset_tightens_the_bound():
    c = get_constraints("white")
    assert c.luminance_bounds == pytest.approx((0.0, c.luminance_range[1] - FEASIBLE_INSET))
    c = get_constraints("black")
    assert c.luminance_bounds == pytest.approx((c.luminance_range[0] + FEASIBLE_INSET, 1.0))


def test_higher_contrast_shrinks_the_allowed_range():
    assert (
        get_constraints("white", 7).luminance_range[1]
        < get_constraints("white", 3).luminance_range[1]
    )
    assert (
        get_constraints("black", 7).luminance_range[0]
        > get_constraints("black", 3).luminance_range[0]
    )


def test_unknown_backgrounds_raise_listing_allowed_values():
    for bad in ["red", "#ffffff", "White", "", None, 0]:
        with pytest.raises(ValueError, match="'white', 'black'"):
            get_constraints(bad)  # pyrefly: ignore[bad-argument-type]


@pytest.mark.parametrize(
    "bad", [0.5, 0.0, -3.0, MAX_MIN_CONTRAST + 0.1, 21.0, float("nan"), float("inf")]
)
def test_min_contrast_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="min_contrast"):
        get_constraints("white", bad)


@pytest.mark.parametrize("bad", ["3", None, True, [3]])
def test_min_contrast_wrong_type_raises(bad):
    with pytest.raises(TypeError, match="min_contrast"):
        get_constraints("white", bad)  # pyrefly: ignore[bad-argument-type]


def test_constraints_are_frozen():
    c = get_constraints("white")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.min_contrast = 1.0  # type: ignore[misc]


def test_invalid_contrast_rejected_directly():
    with pytest.raises(ValueError, match="min_contrast"):
        Constraints(background="x", background_srgb=(1.0, 1.0, 1.0), min_contrast=0.5)
