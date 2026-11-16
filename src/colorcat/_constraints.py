"""Built-in color constraints for each supported background.

v1 supports exactly two backgrounds, ``"white"`` and ``"black"``. A color is
allowed when it has at least ``min_contrast`` WCAG contrast against the
background (``(Y_light + 0.05) / (Y_dark + 0.05)`` on relative luminance ``Y``),
which is a one-sided bound on ``Y``: colors must be darker than the background
or lighter than it. Chroma and hue have no bounds: the sRGB gamut is the only
limit. The opposite of the background (where lines and text are drawn) is
handled by the optimizer as a repelling anchor, not as a constraint.

The constraint set is a plain data object so that v2 can build one from an
arbitrary background without changing the consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._oklab import Array, relative_luminance, srgb_to_linear, srgb_to_oklab

Background = Literal["white", "black"]

# WCAG 2.x requires 3:1 for graphical objects such as chart marks (SC 1.4.11) and
# 4.5:1 for text (SC 1.4.3, AA) or 7:1 (AAA).
DEFAULT_MIN_CONTRAST = 3.0

# The contrast ratio above which the feasible set is too small to optimize over.
# (The maximum possible, 21:1, allows only pure black on white.)
MAX_MIN_CONTRAST = 15.0

# Margin kept inside the luminance bound, so colors still satisfy the contrast
# requirement after 8-bit quantization.
FEASIBLE_INSET = 0.002


@dataclass(frozen=True, slots=True)
class Constraints:
    """The feasible color range for one background.

    ``min_contrast`` is the WCAG contrast ratio every color must have against the
    background. Chroma is limited only by the sRGB gamut, applied elsewhere.
    """

    background: str
    background_srgb: tuple[float, float, float]
    min_contrast: float = DEFAULT_MIN_CONTRAST
    inset: float = FEASIBLE_INSET

    def __post_init__(self) -> None:
        if not 1.0 <= self.min_contrast <= MAX_MIN_CONTRAST:
            raise ValueError(
                f"min_contrast must be between 1 and {MAX_MIN_CONTRAST:g}, got {self.min_contrast!r}"
            )

    @property
    def background_lab(self) -> Array:
        """The background color in OKLab (a fixed anchor in the objective)."""
        return srgb_to_oklab(np.asarray(self.background_srgb))

    @property
    def opposite_lab(self) -> Array:
        """The opposite of the background (black on white, white on black) in OKLab.

        Marks such as plot lines, axes and text are drawn in it, so colors must stay
        distinct from it just as from the background.
        """
        return srgb_to_oklab(1.0 - np.asarray(self.background_srgb))

    @property
    def reference_labs(self) -> Array:
        """``(2, 3)`` OKLab array: the background and its opposite, the fixed anchors."""
        return np.stack([self.background_lab, self.opposite_lab])

    @property
    def background_luminance(self) -> float:
        return float(relative_luminance(srgb_to_linear(np.asarray(self.background_srgb))))

    @property
    def luminance_range(self) -> tuple[float, float]:
        """Exact ``(low, high)`` bounds on relative luminance ``Y`` for the contrast ratio.

        Colors on a light background must be darker (``Y <= (Yb + 0.05) / r - 0.05``);
        colors on a dark one lighter (``Y >= r * (Yb + 0.05) - 0.05``).
        """
        yb, r = self.background_luminance, self.min_contrast
        if yb > 0.179:  # the luminance where white and black text contrast equally
            return 0.0, (yb + 0.05) / r - 0.05
        return r * (yb + 0.05) - 0.05, 1.0

    @property
    def luminance_bounds(self) -> tuple[float, float]:
        """:attr:`luminance_range` tightened by the inset margin."""
        lo, hi = self.luminance_range
        if lo > 0.0:
            lo += self.inset
        if hi < 1.0:
            hi -= self.inset
        return max(lo, 0.0), min(hi, 1.0)


_WHITE = Constraints(background="white", background_srgb=(1.0, 1.0, 1.0))
_BLACK = Constraints(background="black", background_srgb=(0.0, 0.0, 0.0))

_BUILTIN: dict[str, Constraints] = {"white": _WHITE, "black": _BLACK}


def get_constraints(background: str, min_contrast: float = DEFAULT_MIN_CONTRAST) -> Constraints:
    """Return the constraints for ``background`` (``"white"`` or ``"black"``)."""
    try:
        base = _BUILTIN[background]
    except (KeyError, TypeError):
        allowed = ", ".join(repr(k) for k in _BUILTIN)
        raise ValueError(f"background must be one of {allowed}; got {background!r}") from None
    if isinstance(min_contrast, bool) or not isinstance(min_contrast, (int, float)):
        raise TypeError(f"min_contrast must be a number, got {min_contrast!r}")
    if min_contrast == base.min_contrast:
        return base
    return Constraints(
        background=base.background,
        background_srgb=base.background_srgb,
        min_contrast=float(min_contrast),
    )
