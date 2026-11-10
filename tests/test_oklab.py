import numpy as np
import pytest

from colorcat._oklab import (
    in_gamut,
    linear_to_srgb,
    oklab_to_linear_srgb,
    oklab_to_oklch,
    oklab_to_srgb,
    oklch_to_oklab,
    srgb_to_linear,
    srgb_to_oklab,
)

RNG = np.random.default_rng(0)


def test_transfer_function_round_trip():
    x = RNG.uniform(0, 1, size=(1000, 3))
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-12)


def test_transfer_function_known_values():
    assert srgb_to_linear(0.0) == 0.0
    assert srgb_to_linear(1.0) == pytest.approx(1.0)
    # sRGB 0.5 -> ~0.2140 linear
    assert srgb_to_linear(0.5) == pytest.approx(0.2140411, abs=1e-6)


def test_transfer_function_handles_out_of_range_without_nan():
    x = np.array([-0.5, -0.01, 1.2])
    assert np.all(np.isfinite(srgb_to_linear(x)))
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-12)


def test_srgb_oklab_round_trip():
    x = RNG.uniform(0, 1, size=(1000, 3))
    np.testing.assert_allclose(oklab_to_srgb(srgb_to_oklab(x)), x, atol=1e-6)


def test_srgb_oklab_round_trip_is_tight_in_float64():
    x = RNG.uniform(0, 1, size=(1000, 3))
    np.testing.assert_allclose(oklab_to_srgb(srgb_to_oklab(x)), x, atol=1e-12)


def test_oklab_round_trip_for_out_of_gamut_colors():
    lab = RNG.uniform([0.0, -0.5, -0.5], [1.0, 0.5, 0.5], size=(1000, 3))
    lin = oklab_to_linear_srgb(lab)
    # round trip through linear sRGB without clipping
    from colorcat._oklab import linear_srgb_to_oklab

    np.testing.assert_allclose(linear_srgb_to_oklab(lin), lab, atol=1e-9)


def test_leading_axes_are_preserved():
    x = RNG.uniform(0, 1, size=(4, 5, 3))
    assert srgb_to_oklab(x).shape == (4, 5, 3)
    assert in_gamut(srgb_to_oklab(x)).shape == (4, 5)
    assert srgb_to_oklab([0.1, 0.2, 0.3]).shape == (3,)


# Reference values from Ottosson's OKLab post (https://bottosson.github.io/posts/oklab/).
@pytest.mark.parametrize(
    "srgb, expected_lab",
    [
        ((1.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.627955, 0.224863, 0.125846)),
        ((0.0, 1.0, 0.0), (0.866440, -0.233888, 0.179498)),
        ((0.0, 0.0, 1.0), (0.452014, -0.032457, -0.311528)),
    ],
)
def test_known_oklab_values(srgb, expected_lab):
    np.testing.assert_allclose(srgb_to_oklab(srgb), expected_lab, atol=1e-5)


def test_pure_red_in_oklch():
    L, C, h = oklab_to_oklch(srgb_to_oklab([1.0, 0.0, 0.0]))
    assert L == pytest.approx(0.628, abs=1e-3)
    assert C == pytest.approx(0.258, abs=1e-3)
    assert h == pytest.approx(29.23, abs=0.05)


def test_oklch_round_trip():
    lab = RNG.uniform([0.0, -0.4, -0.4], [1.0, 0.4, 0.4], size=(1000, 3))
    np.testing.assert_allclose(oklch_to_oklab(oklab_to_oklch(lab)), lab, atol=1e-12)


def test_oklch_hue_is_in_range_and_wraps():
    lab = RNG.uniform([0.0, -0.4, -0.4], [1.0, 0.4, 0.4], size=(1000, 3))
    h = oklab_to_oklch(lab)[..., 2]
    assert np.all((h >= 0.0) & (h < 360.0))
    # 370 degrees is the same color as 10 degrees
    np.testing.assert_allclose(
        oklch_to_oklab([0.5, 0.1, 370.0]), oklch_to_oklab([0.5, 0.1, 10.0]), atol=1e-12
    )


def test_oklch_of_achromatic_color():
    L, C, h = oklab_to_oklch([0.5, 0.0, 0.0])
    assert (L, C, h) == (0.5, 0.0, 0.0)


def test_gamut_membership():
    # corners of the sRGB cube are in gamut
    corners = np.array([[r, g, b] for r in (0, 1) for g in (0, 1) for b in (0, 1)], dtype=float)
    assert in_gamut(srgb_to_oklab(corners)).all()
    # random sRGB colors are in gamut
    assert in_gamut(srgb_to_oklab(RNG.uniform(0, 1, size=(1000, 3)))).all()
    # clearly out of gamut: very high chroma, and lightness above white
    assert not in_gamut(oklch_to_oklab([0.5, 0.45, 150.0]))
    assert not in_gamut([1.2, 0.0, 0.0])
    assert not in_gamut([-0.1, 0.0, 0.0])


def test_gamut_tolerance():
    # nudge pure red just past the boundary in linear space
    lin = np.array([1.0 + 1e-4, 0.0, 0.0])
    from colorcat._oklab import linear_srgb_to_oklab

    lab = linear_srgb_to_oklab(lin)
    assert not in_gamut(lab)
    assert in_gamut(lab, tol=1e-3)
