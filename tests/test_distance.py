import numpy as np
import pytest

from colorcat._distance import (
    min_pairwise_distance,
    pairwise_distance,
    pairwise_distance_and_gradient,
)

RNG = np.random.default_rng(1)


def test_known_distance():
    d = pairwise_distance([[0.0, 0.0, 0.0]], [[0.3, 0.4, 0.0]])
    assert d.shape == (1, 1)
    assert d[0, 0] == pytest.approx(0.5)


def test_within_set_matrix_is_symmetric_with_zero_diagonal():
    a = RNG.uniform(size=(7, 3))
    d = pairwise_distance(a)
    assert d.shape == (7, 7)
    np.testing.assert_allclose(d, d.T)
    np.testing.assert_allclose(np.diag(d), 0.0)


def test_between_sets_shape_and_values():
    a = RNG.uniform(size=(4, 3))
    b = RNG.uniform(size=(6, 3))
    d = pairwise_distance(a, b)
    assert d.shape == (4, 6)
    assert d[2, 5] == pytest.approx(np.linalg.norm(a[2] - b[5]))


def test_gradient_matches_finite_differences():
    a = RNG.uniform(size=(5, 3))
    b = RNG.uniform(size=(4, 3))
    d, g = pairwise_distance_and_gradient(a, b)
    np.testing.assert_allclose(d, pairwise_distance(a, b))
    assert g.shape == (5, 4, 3)

    eps = 1e-6
    for i in range(len(a)):
        for axis in range(3):
            plus, minus = a.copy(), a.copy()
            plus[i, axis] += eps
            minus[i, axis] -= eps
            numeric = (pairwise_distance(plus, b)[i] - pairwise_distance(minus, b)[i]) / (2 * eps)
            np.testing.assert_allclose(g[i, :, axis], numeric, atol=1e-7)


def test_gradient_is_unit_length_and_zero_at_coincident_points():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0]])
    d, g = pairwise_distance_and_gradient(a, b)
    assert d[0, 0] == 0.0
    np.testing.assert_array_equal(g[0, 0], 0.0)
    np.testing.assert_allclose(np.linalg.norm(g[1, 0]), 1.0)
    assert np.all(np.isfinite(g))


def test_min_pairwise_distance():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [5.0, 5.0, 5.0]])
    assert min_pairwise_distance(a) == pytest.approx(0.1)


def test_min_pairwise_distance_with_fewer_than_two_colors():
    assert min_pairwise_distance(np.zeros((0, 3))) == float("inf")
    assert min_pairwise_distance([[0.1, 0.2, 0.3]]) == float("inf")
