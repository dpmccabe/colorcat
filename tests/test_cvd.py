import numpy as np
import pytest

from colorcat import min_distance, palette
from colorcat._cvd import CVD_TYPES, normalize_cvd, simulate
from colorcat._distance import pairwise_distance, pairwise_distance_and_gradient
from colorcat._energy import energy_and_grad
from colorcat._oklab import srgb_to_oklab

RNG = np.random.default_rng(5)


def _colors(k):
    return srgb_to_oklab(RNG.uniform(0.1, 0.9, (k, 3)))


@pytest.mark.parametrize("kind", CVD_TYPES)
def test_simulation_keeps_greys(kind):
    grey = srgb_to_oklab(np.array([[0.2, 0.2, 0.2], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]]))
    np.testing.assert_allclose(simulate(grey, kind), grey, atol=1e-4)


def test_dichromats_confuse_the_expected_colors():
    red, green, blue, yellow = srgb_to_oklab(
        np.array([[0.8, 0.4, 0.2], [0.55, 0.5, 0.1], [0.2, 0.4, 0.8], [0.8, 0.7, 0.1]])
    )
    # A red-brown and an olive green are far apart normally, close for deutans.
    normal = pairwise_distance(red[None], green[None])[0, 0]
    deutan = pairwise_distance(red[None], green[None], ("deutan",))[0, 0]
    assert deutan < 0.5 * normal
    assert pairwise_distance(blue[None], yellow[None], ("deutan",))[0, 0] > 0.15


@pytest.mark.parametrize("kind", CVD_TYPES)
def test_gradient_matches_finite_differences(kind):
    x, b = _colors(5), _colors(3)
    _, g = pairwise_distance_and_gradient(x, b, kind)
    eps = 1e-6
    for k in range(3):
        step = np.zeros(3)
        step[k] = eps
        xp, xm = x.copy(), x.copy()
        xp[2] += step
        xm[2] -= step
        dp, _ = pairwise_distance_and_gradient(xp, b, kind)
        dm, _ = pairwise_distance_and_gradient(xm, b, kind)
        np.testing.assert_allclose((dp[2] - dm[2]) / (2 * eps), g[2, :, k], atol=1e-6)


def test_energy_gradient_with_cvd_matches_finite_differences():
    x, anchors = _colors(4), _colors(2)
    cvd = ("protan", "tritan")
    _, g = energy_and_grad(x, anchors, 8.0, cvd)
    eps = 1e-6
    num = np.zeros_like(x)
    for i in range(4):
        for k in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, k] += eps
            xm[i, k] -= eps
            num[i, k] = (energy_and_grad(xp, anchors, 8.0, cvd)[0] - energy_and_grad(xm, anchors, 8.0, cvd)[0]) / (2 * eps)
    np.testing.assert_allclose(g, num, atol=1e-5)


def test_distance_is_the_minimum_over_views():
    lab = _colors(6)
    both = pairwise_distance(lab, cvd=("deutan", "tritan"))
    expected = np.minimum.reduce(
        [pairwise_distance(lab)] + [pairwise_distance(lab, cvd=(k,)) for k in ("deutan", "tritan")]
    )
    np.testing.assert_allclose(both, expected)


def test_normalize_cvd():
    assert normalize_cvd(None) == ()
    assert normalize_cvd("deutan") == ("deutan",)
    assert normalize_cvd(["deutan", "protan", "deutan"]) == ("deutan", "protan")
    with pytest.raises(ValueError, match="cvd entries"):
        normalize_cvd(["deuteranopia"])
    with pytest.raises(TypeError):
        normalize_cvd(3)


def test_palette_with_cvd_is_valid_deterministic_and_helps():
    cvd = ("deutan",)
    a = palette(6, cvd=cvd, restart_count=4)
    assert a == palette(6, cvd="deutan", restart_count=4)
    assert len(set(a)) == 6
    plain = palette(6, restart_count=4)
    assert min_distance(a, cvd=cvd) > min_distance(plain, cvd=cvd)
    # Never better than normal vision alone: more views only add constraints.
    assert min_distance(a, cvd=cvd) <= min_distance(a)


def test_palette_rejects_unknown_cvd():
    with pytest.raises(ValueError, match="cvd entries"):
        palette(3, cvd=["blue"])  # pyrefly: ignore[bad-argument-type]


def test_cli_cvd(capsys):
    from colorcat.cli import main

    assert main(["3", "--cvd", "deutan", "protan", "--restarts", "2"]) == 0
    assert len(capsys.readouterr().out.split()) == 3
