"""The feasible color region for a set of constraints.

A color is feasible when it lies inside the sRGB gamut and its relative
luminance is inside the range that gives the required WCAG contrast with the
background. Both are convex in linear sRGB (a cube cut by a slab), which makes
projection exact and cheap there. Operations here take and return OKLab arrays
of shape ``(..., 3)``.

- :func:`gamut_max_chroma`: the largest in-gamut chroma at a lightness and hue.
- :func:`is_feasible`: membership test.
- :func:`project`: map arbitrary points onto the feasible region.
- :func:`sample`: draw feasible points for initialization.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from numpy.typing import ArrayLike

from ._constraints import Constraints
from ._oklab import (
    LUMINANCE_WEIGHTS,
    Array,
    in_gamut,
    linear_srgb_to_oklab,
    oklab_to_linear_srgb,
    oklch_to_oklab,
    relative_luminance,
)

# Upper bound for the chroma search; no sRGB color comes close to it.
_CHROMA_SEARCH_MAX = 0.4
_BISECTION_STEPS = 40

# Grid resolution for finding the largest feasible chroma.
_HUE_STEPS = 360
_L_STEPS = 257


def gamut_max_chroma(L: ArrayLike, h: ArrayLike) -> Array:
    """Largest chroma that is inside the sRGB gamut at lightness ``L``, hue ``h`` (degrees).

    Vectorized bisection; ``L`` and ``h`` broadcast. Returns 0 where no chroma is
    in gamut (e.g. ``L`` outside ``[0, 1]``).
    """
    L, h = np.broadcast_arrays(np.asarray(L, dtype=np.float64), np.asarray(h, dtype=np.float64))
    lo = np.zeros(L.shape)
    hi = np.full(L.shape, _CHROMA_SEARCH_MAX)
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        ok = in_gamut(oklch_to_oklab(np.stack([L, mid, h], axis=-1)))
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo


@cache
def _max_chroma() -> float:
    """Largest in-gamut chroma anywhere, for sampling."""
    hues = np.arange(_HUE_STEPS) * (360.0 / _HUE_STEPS)
    Ls = np.linspace(0.0, 1.0, _L_STEPS)
    return float(np.max(gamut_max_chroma(Ls[None, :], hues[:, None])))


def is_feasible(points: ArrayLike, constraints: Constraints, tol: float = 1e-9) -> np.ndarray:
    """True where an OKLab color is in gamut and has the required contrast (within ``tol``)."""
    lab = np.asarray(points, dtype=np.float64)
    y = relative_luminance(oklab_to_linear_srgb(lab))
    lo, hi = constraints.luminance_range
    return in_gamut(lab, tol) & (y >= lo - tol) & (y <= hi + tol)


# Bracket for the Lagrange multiplier in :func:`project`: wide enough that every
# channel clips to 0 (or 1) at its end.
_LAMBDA_MAX = 1000.0
_LAMBDA_STEPS = 50


def project(points: ArrayLike, constraints: Constraints) -> Array:
    """Map OKLab points ``(..., 3)`` onto the feasible region.

    This is the Euclidean projection in linear sRGB onto the cube intersected
    with the luminance slab ``lo <= Y <= hi``. The solution has the form
    ``clip(z - lam * w, 0, 1)`` for the luminance weights ``w`` and a scalar
    ``lam`` found by bisection (``lam = 0`` for points already in the slab). The
    luminance bounds include the inset margin. Idempotent.
    """
    z = oklab_to_linear_srgb(points)
    lo, hi = constraints.luminance_bounds
    w = LUMINANCE_WEIGHTS
    y0 = np.clip(z, 0.0, 1.0) @ w
    target = np.clip(y0, lo, hi)
    lam_lo = np.full(y0.shape, -_LAMBDA_MAX)
    lam_hi = np.full(y0.shape, _LAMBDA_MAX)
    for _ in range(_LAMBDA_STEPS):
        mid = 0.5 * (lam_lo + lam_hi)
        y = np.clip(z - mid[..., None] * w, 0.0, 1.0) @ w
        too_bright = y > target  # luminance falls as lam grows, so raise lam
        lam_lo = np.where(too_bright, mid, lam_lo)
        lam_hi = np.where(too_bright, lam_hi, mid)
    lam = np.where(y0 == target, 0.0, 0.5 * (lam_lo + lam_hi))
    return linear_srgb_to_oklab(np.clip(z - lam[..., None] * w, 0.0, 1.0))


def sample(rng: np.random.Generator, k: int, constraints: Constraints) -> Array:
    """Draw ``k`` feasible OKLab colors by rejection sampling in OKLCh.

    ``(L, C, h)`` is drawn uniformly from the box ``[0, 1] x [0, max chroma] x
    [0, 360)`` and kept when it is inside the gamut with the required contrast.
    Deterministic given ``rng``.
    """
    c_top = _max_chroma()
    lo, hi = constraints.luminance_bounds
    out = np.empty((0, 3))
    while len(out) < k:
        m = max(8 * (k - len(out)), 128)
        L = rng.uniform(0.0, 1.0, m)
        C = rng.uniform(0.0, c_top, m)
        h = rng.uniform(0.0, 360.0, m)
        lab = oklch_to_oklab(np.stack([L, C, h], axis=-1))
        y = relative_luminance(oklab_to_linear_srgb(lab))
        keep = (C <= gamut_max_chroma(L, h)) & (y >= lo) & (y <= hi)
        out = np.concatenate([out, lab[keep]])
    return out[:k]
