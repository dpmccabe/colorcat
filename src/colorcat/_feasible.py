"""The feasible color region for a set of constraints.

A color is feasible when its lightness is inside the band and it lies inside
the sRGB gamut, with a small inset margin so it stays feasible after 8-bit
quantization. Operations here take and
return OKLab arrays of shape ``(..., 3)``.

- :func:`gamut_max_chroma`: the largest in-gamut chroma at a lightness and hue.
- :func:`project`: map arbitrary points onto the feasible region.
- :func:`sample`: draw feasible points for initialization.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike

from ._constraints import Constraints
from ._oklab import Array, in_gamut, oklab_to_oklch, oklch_to_oklab

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


@lru_cache(maxsize=None)
def _max_chroma(constraints: Constraints) -> float:
    """Largest feasible chroma anywhere in the lightness band, for sampling."""
    L_lo, L_hi = constraints.lightness_bounds
    hues = np.arange(_HUE_STEPS) * (360.0 / _HUE_STEPS)
    Ls = np.linspace(L_lo, L_hi, _L_STEPS)
    return float(np.max(gamut_max_chroma(Ls[None, :], hues[:, None]))) - constraints.inset


def project(points: ArrayLike, constraints: Constraints) -> Array:
    """Map OKLab points ``(..., 3)`` onto the nearest-ish feasible color.

    Hue is kept. Lightness is clipped into the band, then chroma into
    ``[0, gamut maximum]`` (less the inset). Idempotent.
    """
    lch = oklab_to_oklch(points)
    L, C, h = lch[..., 0], lch[..., 1], lch[..., 2]
    L = np.clip(L, *constraints.lightness_bounds)
    C = np.clip(C, 0.0, gamut_max_chroma(L, h) - constraints.inset)
    return oklch_to_oklab(np.stack([L, C, h], axis=-1))


def sample(rng: np.random.Generator, k: int, constraints: Constraints) -> Array:
    """Draw ``k`` feasible OKLab colors by rejection sampling in OKLCh.

    ``(L, C, h)`` is drawn uniformly from the box ``[L band] x [0, max
    feasible chroma] x [0, 360)`` and kept when inside the gamut. Deterministic
    given ``rng``.
    """
    L_lo, L_hi = constraints.lightness_bounds
    c_top = _max_chroma(constraints)
    out = np.empty((0, 3))
    while len(out) < k:
        m = max(4 * (k - len(out)), 64)
        L = rng.uniform(L_lo, L_hi, m)
        C = rng.uniform(0.0, c_top, m)
        h = rng.uniform(0.0, 360.0, m)
        keep = C <= gamut_max_chroma(L, h) - constraints.inset
        out = np.concatenate([out, oklch_to_oklab(np.stack([L, C, h], axis=-1)[keep])])
    return out[:k]
