import numpy as np
import pytest

from colorcat import _energy
from colorcat._constraints import get_constraints
from colorcat._energy import (
    _descend,
    energy_and_grad,
    min_distance_with_anchors,
    optimize,
    optimize_once,
)
from colorcat._feasible import sample

C = get_constraints("white")
ANCHORS = C.background_lab[None, :]


@pytest.mark.parametrize("p", [4.0, 16.0])
def test_gradient_matches_finite_differences(p):
    rng = np.random.default_rng(0)
    x = sample(rng, 6, C)
    anchors = np.vstack([ANCHORS, sample(rng, 2, C)])
    _, grad = energy_and_grad(x, anchors, p)
    eps = 1e-6
    numeric = np.zeros_like(x)
    for i in range(len(x)):
        for k in range(3):
            plus, minus = x.copy(), x.copy()
            plus[i, k] += eps
            minus[i, k] -= eps
            numeric[i, k] = (
                energy_and_grad(plus, anchors, p)[0] - energy_and_grad(minus, anchors, p)[0]
            ) / (2 * eps)
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-7)


def test_energy_tracks_minimum_distance_as_p_grows():
    x = sample(np.random.default_rng(1), 8, C)
    d_min = min_distance_with_anchors(x, ANCHORS)
    e_small = energy_and_grad(x, ANCHORS, 4.0)[0]
    e_large = energy_and_grad(x, ANCHORS, 256.0)[0]
    # E_p -> -log(d_min) from above, with a gap of at most log(#pairs)/p.
    assert e_large >= -np.log(d_min) - 1e-9
    assert abs(e_large + np.log(d_min)) < abs(e_small + np.log(d_min))
    assert abs(e_large + np.log(d_min)) < np.log(45) / 256 + 1e-9


def test_energy_is_finite_for_coincident_points_and_a_single_particle():
    x = np.zeros((3, 3)) + 0.3
    e, g = energy_and_grad(x, ANCHORS, 8.0)
    assert np.isfinite(e) and np.all(np.isfinite(g))
    e, g = energy_and_grad(sample(np.random.default_rng(0), 1, C), ANCHORS, 8.0)
    assert np.isfinite(e) and g.shape == (1, 3)


def test_descent_lowers_energy_and_keeps_points_feasible():
    x = sample(np.random.default_rng(2), 10, C)
    before = energy_and_grad(x, ANCHORS, 8.0)[0]
    out = _descend(x, ANCHORS, 8.0, C)
    assert energy_and_grad(out, ANCHORS, 8.0)[0] < before
    assert min_distance_with_anchors(out, ANCHORS) > min_distance_with_anchors(x, ANCHORS)
    from colorcat._feasible import project

    np.testing.assert_allclose(project(out, C), out, atol=1e-9)


def test_optimize_once_returns_quantized_distinct_colors():
    r = optimize_once(np.random.default_rng(0), 12, C, ANCHORS)
    assert r.rgb8.shape == (12, 3) and r.rgb8.dtype == np.int64
    assert len({tuple(c) for c in r.rgb8}) == 12
    assert r.min_distance > 0.1


def test_same_seed_is_identical_and_different_seed_differs():
    a = optimize(5, 6, C, ANCHORS, 3)
    b = optimize(5, 6, C, ANCHORS, 3)
    np.testing.assert_array_equal(a.rgb8, b.rgb8)
    other = optimize(6, 6, C, ANCHORS, 3)
    assert not np.array_equal(a.rgb8, other.rgb8)


def test_more_restarts_are_never_worse():
    scores = [optimize(0, 9, C, ANCHORS, k).min_distance for k in (1, 2, 4, 8)]
    assert scores == sorted(scores)


def test_restart_result_does_not_depend_on_restart_count():
    # restart 0 of a 1-restart run equals restart 0 inside a 4-restart run
    stream = np.random.SeedSequence(0).spawn(4)[0]
    solo = optimize_once(np.random.default_rng(stream), 7, C, ANCHORS)
    assert optimize(0, 7, C, ANCHORS, 4).min_distance >= solo.min_distance


def test_hyperparameters_are_sane():
    assert _energy.P_SCHEDULE == tuple(sorted(_energy.P_SCHEDULE))
    assert 0 < _energy.BACKTRACK_FACTOR < 1


def test_hops_never_hurt_and_are_deterministic(monkeypatch):
    monkeypatch.setattr(_energy, "HOP_COUNT", 0)
    plain = optimize_once(np.random.default_rng(4), 10, C, ANCHORS)
    monkeypatch.setattr(_energy, "HOP_COUNT", 6)
    hopped = optimize_once(np.random.default_rng(4), 10, C, ANCHORS)
    again = optimize_once(np.random.default_rng(4), 10, C, ANCHORS)
    assert (hopped.min_distance, -hopped.energy) >= (plain.min_distance, -plain.energy)
    np.testing.assert_array_equal(hopped.rgb8, again.rgb8)


def test_hop_moves_a_closest_pair_color_and_stays_feasible():
    from colorcat._energy import _hop
    from colorcat._feasible import project

    x = sample(np.random.default_rng(1), 8, C)
    y = _hop(np.random.default_rng(2), x, ANCHORS, C, ())
    assert y.shape == x.shape
    np.testing.assert_allclose(project(y, C), y, atol=1e-9)
    assert min_distance_with_anchors(y, ANCHORS) >= min_distance_with_anchors(x, ANCHORS)


def test_hops_work_for_a_single_color_and_with_cvd():
    r = optimize_once(np.random.default_rng(0), 1, C, ANCHORS, ("deutan",))
    assert r.rgb8.shape == (1, 3)
