"""sRGB <-> linear sRGB <-> OKLab <-> OKLCh conversions.

OKLab is Björn Ottosson's perceptual color space (2020, matrices from the 2021
revision). All functions are vectorized over leading axes: colors are arrays of
shape ``(..., 3)``.

Conventions:

- sRGB and linear sRGB channels are floats in ``[0, 1]`` (not 0-255).
- OKLab is ``(L, a, b)`` with ``L`` in ``[0, 1]`` for in-gamut colors.
- OKLCh is ``(L, C, h)`` with ``C = hypot(a, b)`` and ``h`` in degrees, ``[0, 360)``.

Conversions are *not* clipped: out-of-gamut colors map to values outside
``[0, 1]`` and round-trip faithfully, which lets the optimizer move through
slightly out-of-gamut space. Use :func:`in_gamut` to test membership.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

# Linear sRGB -> LMS (cone-like) responses.
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)

# Cube-rooted LMS -> OKLab.
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)

# Inverses computed numerically so that forward/backward round-trips are exact
# to floating-point precision.
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)

# Default slack when testing gamut membership, to absorb floating-point noise.
GAMUT_TOL = 1e-9


def _as_float(x: ArrayLike) -> Array:
    return np.asarray(x, dtype=np.float64)


def srgb_to_linear(srgb: ArrayLike) -> Array:
    """Apply the inverse sRGB transfer function (gamma decode).

    Uses the sign-preserving extension, so out-of-range inputs stay finite.
    """
    c = _as_float(srgb)
    a = np.abs(c)
    out = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return np.copysign(out, c)


def linear_to_srgb(linear: ArrayLike) -> Array:
    """Apply the sRGB transfer function (gamma encode)."""
    c = _as_float(linear)
    a = np.abs(c)
    out = np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1.0 / 2.4) - 0.055)
    return np.copysign(out, c)


def linear_srgb_to_oklab(linear: ArrayLike) -> Array:
    """Linear sRGB ``(..., 3)`` -> OKLab ``(..., 3)``."""
    lms = _as_float(linear) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_linear_srgb(lab: ArrayLike) -> Array:
    """OKLab ``(..., 3)`` -> linear sRGB ``(..., 3)`` (unclipped)."""
    lms = (_as_float(lab) @ _M2_INV.T) ** 3
    return lms @ _M1_INV.T


def srgb_to_oklab(srgb: ArrayLike) -> Array:
    """sRGB ``(..., 3)`` in ``[0, 1]`` -> OKLab."""
    return linear_srgb_to_oklab(srgb_to_linear(srgb))


def oklab_to_srgb(lab: ArrayLike) -> Array:
    """OKLab -> sRGB in ``[0, 1]`` (unclipped; may fall outside for out-of-gamut input)."""
    return linear_to_srgb(oklab_to_linear_srgb(lab))


def oklab_to_oklch(lab: ArrayLike) -> Array:
    """OKLab ``(L, a, b)`` -> OKLCh ``(L, C, h)`` with ``h`` in degrees ``[0, 360)``.

    Hue is arbitrary (returned as 0 for exact zero chroma) for achromatic colors.
    """
    lab = _as_float(lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([L, C, h], axis=-1)


def oklch_to_oklab(lch: ArrayLike) -> Array:
    """OKLCh ``(L, C, h)`` with ``h`` in degrees -> OKLab ``(L, a, b)``."""
    lch = _as_float(lch)
    L, C, h = lch[..., 0], lch[..., 1], np.radians(lch[..., 2])
    return np.stack([L, C * np.cos(h), C * np.sin(h)], axis=-1)


def in_gamut(lab: ArrayLike, tol: float = GAMUT_TOL) -> NDArray[np.bool_]:
    """True where an OKLab color is representable in sRGB (within ``tol`` per channel).

    Returns a boolean array of shape ``lab.shape[:-1]``.
    """
    lin = oklab_to_linear_srgb(lab)
    return np.all((lin >= -tol) & (lin <= 1.0 + tol), axis=-1)
