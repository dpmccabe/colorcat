import numpy as np
import pytest

from colorcat._constraints import get_constraints
from colorcat._feasible import _max_chroma, gamut_max_chroma, is_feasible, project, sample
from colorcat._oklab import (
    in_gamut,
    oklab_to_linear_srgb,
    oklab_to_oklch,
    oklch_to_oklab,
    relative_luminance,
    srgb_to_oklab,
)

BACKGROUNDS = ["white", "black"]
RNG = np.random.default_rng(3)


def feasible(lab, c, tol=1e-9):
    return is_feasible(lab, c, tol)


def test_gamut_max_chroma_is_the_boundary():
    L = RNG.uniform(0.05, 0.95, 500)
    h = RNG.uniform(0, 360, 500)
    cmax = gamut_max_chroma(L, h)
    inside = oklch_to_oklab(np.stack([L, cmax, h], axis=-1))
    outside = oklch_to_oklab(np.stack([L, cmax + 1e-5, h], axis=-1))
    assert in_gamut(inside).all()
    assert not in_gamut(outside).any()


def test_gamut_max_chroma_known_and_degenerate():
    # Red's hue at its own lightness: the boundary chroma is red's chroma.
    L, C, h = oklab_to_oklch(srgb_to_oklab([1.0, 0.0, 0.0]))
    assert gamut_max_chroma(L, h) == pytest.approx(C, abs=1e-6)
    assert gamut_max_chroma(0.0, 10.0) == pytest.approx(0.0, abs=5e-3)  # in_gamut tolerance
    assert gamut_max_chroma(1.0, 10.0) == pytest.approx(0.0, abs=5e-3)
    assert gamut_max_chroma(1.5, 10.0) == 0.0
    assert gamut_max_chroma(np.full((2, 3), 0.5), np.full((2, 3), 100.0)).shape == (2, 3)


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_project_returns_feasible_points(name):
    c = get_constraints(name)
    pts = RNG.uniform([-0.5, -0.6, -0.6], [1.5, 0.6, 0.6], size=(5000, 3))
    out = project(pts, c)
    assert out.shape == pts.shape
    assert feasible(out, c).all()


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_project_is_idempotent(name):
    c = get_constraints(name)
    pts = RNG.uniform([-0.5, -0.6, -0.6], [1.5, 0.6, 0.6], size=(5000, 3))
    once = project(pts, c)
    np.testing.assert_allclose(project(once, c), once, atol=1e-9)


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_project_leaves_feasible_points_alone(name):
    c = get_constraints(name)
    pts = sample(np.random.default_rng(0), 500, c)
    np.testing.assert_allclose(project(pts, c), pts, atol=1e-9)


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_project_handles_greys_and_extremes(name):
    c = get_constraints(name)
    pts = np.array([[0.5, 0, 0], [0, 0, 0], [1, 0, 0], [0.5, 0.3, 0.0], [0.7, -0.3, 0.3]])
    out = project(pts, c)
    assert feasible(out, c).all()


def test_project_works_with_any_leading_shape():
    c = get_constraints("white")
    pts = oklch_to_oklab(np.array([[[0.9, 0.3, 120.0], [0.1, 0.01, 200.0]]]))
    out = project(pts, c)
    assert out.shape == (1, 2, 3)
    assert feasible(out, c).all()


def test_project_moves_a_too_bright_color_only_as_far_as_needed():
    c = get_constraints("white")
    _, hi = c.luminance_bounds
    y = relative_luminance(oklab_to_linear_srgb(project(srgb_to_oklab([[0.9, 0.9, 0.2]]), c)))
    assert y[0] == pytest.approx(hi, abs=1e-9)
    # A color that already satisfies the contrast is not touched.
    dark = srgb_to_oklab([[0.1, 0.2, 0.5]])
    np.testing.assert_allclose(project(dark, c), dark, atol=1e-9)


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_project_is_feasible_for_dense_hues(name):
    # Dense sweep over hue and lightness, including far outside the gamut.
    c = get_constraints(name)
    L, h = np.meshgrid(np.linspace(-0.1, 1.1, 120), np.linspace(0, 360, 721))
    lch = np.stack([L, np.full_like(L, 0.25), h], axis=-1).reshape(-1, 3)
    assert feasible(project(oklch_to_oklab(lch), c), c).all()


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_sample_points_are_feasible_and_reach_every_hue(name):
    c = get_constraints(name)
    pts = sample(np.random.default_rng(1), 2000, c)
    assert pts.shape == (2000, 3)
    assert feasible(pts, c).all()
    h = oklab_to_oklch(pts)[:, 2]
    # Density follows how much feasible chroma each hue has (sparse for cyan on
    # white), but no hue sector may be empty.
    assert np.histogram(h, bins=12, range=(0, 360))[0].min() > 5


def test_sample_is_deterministic_and_handles_small_k():
    c = get_constraints("white")
    a = sample(np.random.default_rng(7), 5, c)
    b = sample(np.random.default_rng(7), 5, c)
    np.testing.assert_array_equal(a, b)
    assert sample(np.random.default_rng(7), 0, c).shape == (0, 3)
    assert not np.array_equal(a, sample(np.random.default_rng(8), 5, c))


def test_max_chroma_is_cached_and_positive():
    assert _max_chroma() > 0.1
    before = _max_chroma.cache_info().hits
    _max_chroma()
    assert _max_chroma.cache_info().hits == before + 1


@pytest.mark.parametrize("name", BACKGROUNDS)
@pytest.mark.parametrize("ratio", [1.0, 4.5, 7.0, 12.0])
def test_project_and_sample_follow_min_contrast(name, ratio):
    c = get_constraints(name, ratio)
    pts = RNG.uniform([-0.5, -0.6, -0.6], [1.5, 0.6, 0.6], size=(2000, 3))
    assert feasible(project(pts, c), c).all()
    assert feasible(sample(np.random.default_rng(2), 500, c), c).all()
