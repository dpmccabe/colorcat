"""The single place where color distance is defined.

Distance is Euclidean in OKLab (ΔE_OK). Everything that needs a distance or its
gradient (the energy objective, ``min_distance``, benchmarks) goes through this
module, so that a different notion of distance only has to be implemented here.

A *view* is how a color is perceived: ``None`` is normal vision, otherwise a
color-vision-deficiency name from :mod:`._cvd`. With ``cvd`` given, two colors are
only as far apart as in their *closest* view (normal vision plus each deficiency).

Colors are OKLab arrays of shape ``(m, 3)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._cvd import simulate
from ._oklab import Array


def views(cvd: tuple[str, ...] = ()) -> tuple[str | None, ...]:
    """Normal vision first, then each deficiency in ``cvd``."""
    return (None, *cvd)


def pairwise_distance(
    a: ArrayLike, b: ArrayLike | None = None, cvd: tuple[str, ...] = ()
) -> Array:
    """ΔE_OK between every color in ``a`` ``(m, 3)`` and every color in ``b`` ``(k, 3)``.

    Returns an ``(m, k)`` matrix. With ``b=None`` the distances are within ``a``
    (an ``(m, m)`` symmetric matrix with a zero diagonal). With ``cvd`` each
    entry is the smallest distance over normal vision and the deficiencies.
    """
    a = np.asarray(a, dtype=np.float64)
    b = a if b is None else np.asarray(b, dtype=np.float64)
    best = None
    for view in views(cvd):
        va = a if view is None else simulate(a, view)
        vb = b if view is None else simulate(b, view)
        diff = va[:, None, :] - vb[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=-1))
        best = d if best is None else np.minimum(best, d)
    return best


def pairwise_distance_and_gradient(
    a: ArrayLike, b: ArrayLike, view: str | None = None
) -> tuple[Array, Array]:
    """Distances in one ``view`` and their gradients with respect to ``a``.

    Returns ``(d, g)`` where ``d`` is ``(m, k)`` as in :func:`pairwise_distance`
    and ``g[i, j]`` is ``∂ d[i, j] / ∂ a[i]`` as a ``(m, k, 3)`` array (the unit
    vector pointing from ``b[j]`` to ``a[i]``; for a deficiency view it is mapped
    back to OKLab through the view's Jacobian). The gradient is defined as zero
    where two colors coincide (``d == 0``).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if view is None:
        va, vb, jac = a, b, None
    else:
        va, jac = simulate(a, view, jacobian=True)
        vb = simulate(b, view)
    diff = va[:, None, :] - vb[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    safe = np.where(d > 0.0, d, 1.0)
    grad = diff / safe[..., None]
    if jac is not None:
        grad = np.einsum("iab,ija->ijb", jac, grad)
    return d, grad


def min_pairwise_distance(a: ArrayLike, cvd: tuple[str, ...] = ()) -> float:
    """Smallest ΔE_OK between any two distinct rows of ``a``; ``inf`` for fewer than two colors."""
    a = np.asarray(a, dtype=np.float64)
    if len(a) < 2:
        return float("inf")
    d = pairwise_distance(a, cvd=cvd)
    return float(d[np.triu_indices(len(a), k=1)].min())
