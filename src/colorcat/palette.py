"""The public API: :func:`palette` and :func:`min_distance`."""

from __future__ import annotations

import operator
import warnings
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np

from ._constraints import get_constraints
from ._cvd import CvdType, normalize_cvd
from ._distance import min_pairwise_distance, pairwise_distance
from ._energy import optimize
from ._format import FORMATS, format_colors, hex_to_oklab
from ._oklab import Array

# Below this ΔE_OK the closest pair is hard to tell apart (a just-noticeable
# difference is about 0.02), so we warn.
LOW_DISTANCE_WARNING = 0.03


def _as_int(value: object, name: str, minimum: int) -> int:
    try:
        out = operator.index(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if isinstance(value, bool) or out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return out


def _parse_colors(colors: Sequence[str], name: str) -> Array:
    """Parse a sequence of hex strings into an ``(m, 3)`` OKLab array."""
    if isinstance(colors, str) or not isinstance(colors, Iterable):
        raise TypeError(f"{name} must be a sequence of hex color strings, got {colors!r}")
    try:
        return np.array([hex_to_oklab(c) for c in colors], dtype=np.float64).reshape(-1, 3)
    except ValueError as err:
        raise ValueError(f"{name}: {err}") from None


def _farthest_first(lab: Array, background: Array, cvd: tuple[str, ...]) -> np.ndarray:
    """Order colors so the first is farthest from the background, then greedily farthest apart."""
    n = len(lab)
    order = [int(np.argmax(pairwise_distance(lab, background[None, :], cvd)[:, 0]))]
    d = pairwise_distance(lab, lab[order], cvd)[:, 0]
    while len(order) < n:
        d[order] = -1.0
        nxt = int(np.argmax(d))
        order.append(nxt)
        d = np.minimum(d, pairwise_distance(lab, lab[[nxt]], cvd)[:, 0])
    return np.array(order)


def palette(
    n: int,
    *,
    background: Literal["white", "black"] = "white",
    restart_count: int = 4,
    exclude: Sequence[str] = (),
    fixed: Sequence[str] = (),
    cvd: Sequence[CvdType] | CvdType = (),
    color_format: Literal["hex", "rgb", "oklab", "oklch"] = "hex",
    seed: int | None = 0,
) -> list:
    """Return ``n`` maximally distinct colors, by default as ``"#rrggbb"`` strings.

    The colors are chosen to maximize the smallest perceptual (OKLab) distance
    between any two of them and between each of them and the background.

    Args:
        n: Number of colors (``>= 0``).
        background: ``"white"`` or ``"black"``, the color the palette will be
            drawn on. It selects a suitable lightness range.
        restart_count: Number of independent optimization restarts (``>= 1``).
            Each restart also refines its result with a number of basin-hopping
            moves. More restarts never give a worse palette, but take longer.
        exclude: Hex colors the new colors must stay distinct from (for example
            colors already used elsewhere in a figure). They are not returned.
        fixed: Hex colors already in the palette. The ``n`` new colors are chosen
            to extend them, staying as distinct from them as from each other.
            The fixed colors are never moved, are not validated against the
            built-in ranges, and are not included in the result.
        cvd: One or more color-vision deficiencies to design for: ``"protan"``,
            ``"deutan"`` and/or ``"tritan"`` (a name or a sequence of names;
            full dichromacy, simulated after Machado et al. 2009). Every pair of
            colors must then be distinct both to normal vision and as seen with
            each deficiency, so the distance of the closest pair drops as views
            are added, more so for larger ``n``. Default: normal vision only.
        color_format: ``"hex"`` (strings), ``"rgb"`` (0-255 integer tuples),
            ``"oklab"`` (L, a, b tuples) or ``"oklch"`` (L, C, h tuples, hue in degrees).
        seed: Seed for the random starting points. The same arguments and seed
            always return the same palette. ``None`` uses fresh entropy.

    The colors are ordered so that the first is the most distant from the
    background and each next is the farthest from those before it. Each ``n``
    is optimized separately: ``palette(4)`` is not a prefix of ``palette(5)``.
    """
    n = _as_int(n, "n", 0)
    restart_count = _as_int(restart_count, "restart_count", 1)
    if seed is not None:
        seed = _as_int(seed, "seed", 0)
    constraints = get_constraints(background)
    cvd_views = normalize_cvd(cvd)
    if color_format not in FORMATS:
        raise ValueError(
            f"color_format must be one of {', '.join(repr(f) for f in FORMATS)}; got {color_format!r}"
        )
    extra = np.concatenate([_parse_colors(exclude, "exclude"), _parse_colors(fixed, "fixed")])
    if n == 0:
        return []

    anchors = np.concatenate([constraints.background_lab[None, :], extra])
    result = optimize(seed, n, constraints, anchors, restart_count, cvd_views)
    if result.min_distance < LOW_DISTANCE_WARNING:
        warnings.warn(
            f"the closest colors in this {n}-color palette differ by only "
            f"{result.min_distance:.3f} (ΔE_OK{' in the worst view' if cvd_views else ''}); "
            "they may be hard to tell apart",
            stacklevel=2,
        )
    order = _farthest_first(result.lab, constraints.background_lab, cvd_views)
    return format_colors(result.rgb8[order], color_format)


def min_distance(colors: Sequence[str], *, cvd: Sequence[CvdType] | CvdType = ()) -> float:
    """Perceptual distance (ΔE_OK) between the two closest of ``colors`` (hex strings).

    With ``cvd`` (as in :func:`palette`), the closest pair over normal vision and
    each listed deficiency. Returns ``inf`` for fewer than two colors.
    """
    lab = np.array([hex_to_oklab(c) for c in colors]).reshape(-1, 3)
    return min_pairwise_distance(lab, normalize_cvd(cvd))
