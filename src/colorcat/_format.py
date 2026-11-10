"""Parsing, 8-bit quantization and output formatting of colors."""

from __future__ import annotations

import itertools
import re

import numpy as np
from numpy.typing import ArrayLike

from ._constraints import Constraints
from ._oklab import Array, in_gamut, oklab_to_oklch, oklab_to_srgb, srgb_to_oklab

FORMATS = ("hex", "rgb", "oklab", "oklch")

_HEX = re.compile(r"#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})")

# The 27 lattice offsets around a rounded 8-bit color.
_NEIGHBOURS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)


def hex_to_srgb(color: str) -> Array:
    """Parse ``"#rrggbb"`` / ``"#rgb"`` (``#`` optional) into sRGB floats in ``[0, 1]``."""
    m = _HEX.fullmatch(color.strip()) if isinstance(color, str) else None
    if m is None:
        raise ValueError(f"not a hex color: {color!r}")
    digits = m.group(1)
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    return np.array([int(digits[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def hex_to_oklab(color: str) -> Array:
    return srgb_to_oklab(hex_to_srgb(color))


def srgb8_to_hex(rgb8: ArrayLike) -> list[str]:
    """``(m, 3)`` integer 0-255 colors -> ``["#rrggbb", ...]``."""
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in np.asarray(rgb8, dtype=int)]


def lab_to_srgb8(lab: ArrayLike) -> np.ndarray:
    """OKLab ``(m, 3)`` -> nearest 8-bit sRGB (clipped to the gamut), ``int64``."""
    return np.clip(np.rint(oklab_to_srgb(lab) * 255.0), 0, 255).astype(np.int64)


def quantize(lab: ArrayLike, constraints: Constraints) -> np.ndarray:
    """Snap OKLab colors ``(m, 3)`` to 8-bit sRGB, returning ``(m, 3)`` ints.

    Plain rounding can step just outside the lightness band, so each color takes the nearest *feasible* lattice point among the
    27 around its rounded value (falling back to plain rounding if none is).
    """
    lab = np.asarray(lab, dtype=np.float64)
    base = lab_to_srgb8(lab)  # (m, 3)
    cand = np.clip(base[:, None, :] + _NEIGHBOURS[None, :, :], 0, 255)  # (m, 27, 3)
    cand_lab = srgb_to_oklab(cand / 255.0)
    L = oklab_to_oklch(cand_lab)[..., 0]
    lo, hi = constraints.lightness
    ok = (L >= lo) & (L <= hi) & in_gamut(cand_lab)
    dist = np.linalg.norm(cand_lab - lab[:, None, :], axis=-1)
    best = np.argmin(np.where(ok, dist, np.inf), axis=1)
    chosen = cand[np.arange(len(lab)), best]
    return np.where(ok.any(axis=1)[:, None], chosen, base)


def format_colors(rgb8: ArrayLike, fmt: str) -> list:
    """Format ``(m, 3)`` 8-bit colors: ``"#rrggbb"`` strings, or tuples of ints (``rgb``,
    0-255) or floats (``oklab`` as L, a, b and ``oklch`` as L, C, h in degrees)."""
    rgb8 = np.asarray(rgb8, dtype=np.int64)
    if fmt == "hex":
        return srgb8_to_hex(rgb8)
    if fmt == "rgb":
        return [tuple(int(v) for v in row) for row in rgb8]
    if fmt in ("oklab", "oklch"):
        lab = srgb_to_oklab(rgb8 / 255.0)
        values = lab if fmt == "oklab" else oklab_to_oklch(lab)
        return [tuple(float(v) for v in row) for row in values]
    raise ValueError(
        f"color_format must be one of {', '.join(repr(f) for f in FORMATS)}; got {fmt!r}"
    )
