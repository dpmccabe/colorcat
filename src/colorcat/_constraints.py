"""Built-in color constraints for each supported background.

v1 supports exactly two backgrounds, ``"white"`` and ``"black"``. The lightness
band is chosen by the background alone: far enough from it for clear contrast,
and away from pure white/black. Chroma has no bounds of its own: the sRGB gamut
is its only limit. The numbers live here, in one place, so they are easy to tune. They are starting values.

The constraint set is a plain data object so that v2 can build one from
user-supplied overrides or from an arbitrary background without changing
the consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._oklab import Array, srgb_to_oklab

Background = Literal["white", "black"]

# Margin kept inside the L and C bounds, so colors still satisfy the
# constraints after 8-bit quantization.
FEASIBLE_INSET = 0.002


@dataclass(frozen=True, slots=True)
class Constraints:
    """The feasible color range for one background.

    Lightness is OKLab L and hue is in degrees (``hue`` is the full circle).
    Chroma is limited only by the sRGB gamut, applied elsewhere.
    """

    background: str
    background_srgb: tuple[float, float, float]
    lightness: tuple[float, float]
    hue: tuple[float, float] = (0.0, 360.0)
    inset: float = FEASIBLE_INSET

    def __post_init__(self) -> None:
        for name in ("lightness", "hue"):
            lo, hi = getattr(self, name)
            if not lo < hi:
                raise ValueError(f"{name} range must have lo < hi, got {(lo, hi)}")

    @property
    def background_lab(self) -> Array:
        """The background color in OKLab (a fixed anchor in the objective)."""
        return srgb_to_oklab(np.asarray(self.background_srgb))

    @property
    def lightness_bounds(self) -> tuple[float, float]:
        """Lightness range shrunk by the inset margin."""
        return self.lightness[0] + self.inset, self.lightness[1] - self.inset


_WHITE = Constraints(
    background="white",
    background_srgb=(1.0, 1.0, 1.0),
    lightness=(0.40, 0.75),
)

# Mirrored: the band sits above mid-grey.
_BLACK = Constraints(
    background="black",
    background_srgb=(0.0, 0.0, 0.0),
    lightness=(0.60, 0.90),
)

_BUILTIN: dict[str, Constraints] = {"white": _WHITE, "black": _BLACK}


def get_constraints(background: str) -> Constraints:
    """Return the built-in constraints for ``background`` (``"white"`` or ``"black"``)."""
    try:
        return _BUILTIN[background]
    except (KeyError, TypeError):
        allowed = ", ".join(repr(k) for k in _BUILTIN)
        raise ValueError(f"background must be one of {allowed}; got {background!r}") from None
