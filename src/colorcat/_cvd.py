"""Color-vision-deficiency (CVD) simulation.

Each deficiency is simulated by a 3x3 matrix applied in linear sRGB (Machado,
Oliveira & Fernandes 2009, severity 1.0, i.e. full dichromacy). The simulated
color is then compared in OKLab like any other, so a CVD "view" of a palette is
just a different map from OKLab to OKLab. Where a view's Jacobian is needed (for
the optimizer's gradients) it is computed analytically.

- ``protan``: missing L (red) cones.
- ``deutan``: missing M (green) cones.
- ``tritan``: missing S (blue) cones.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._oklab import _M1, _M1_INV, _M2, _M2_INV, Array

CvdType = Literal["protan", "deutan", "tritan"]

_MATRICES: dict[str, Array] = {
    "protan": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    "deutan": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
    "tritan": np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ]
    ),
}

CVD_TYPES: tuple[str, ...] = tuple(_MATRICES)

# The simulation as a linear map on LMS, so only cube roots are nonlinear.
_LMS_MAPS = {k: _M1 @ m @ _M1_INV for k, m in _MATRICES.items()}

_MIN_CBRT_SQ = 1e-8  # keeps the cube-root derivative finite near black


def normalize_cvd(cvd: str | None | ArrayLike) -> tuple[str, ...]:
    """Validate ``cvd`` (a name or a sequence of names) into a tuple without repeats."""
    if cvd is None:
        return ()
    if isinstance(cvd, str):
        cvd = (cvd,)
    try:
        names = tuple(cvd)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"cvd must be a sequence of names, got {cvd!r}") from None
    for name in names:
        if name not in CVD_TYPES:
            allowed = ", ".join(repr(k) for k in CVD_TYPES)
            raise ValueError(f"cvd entries must be one of {allowed}; got {name!r}")
    return tuple(dict.fromkeys(names))


def simulate(lab: ArrayLike, kind: str, jacobian: bool = False) -> Array | tuple[Array, Array]:
    """The OKLab colors ``lab`` ``(..., 3)`` as seen with deficiency ``kind``.

    With ``jacobian=True`` also returns ``J`` ``(..., 3, 3)`` with
    ``J[..., k, l] = d simulated_k / d lab_l``.
    """
    lab = np.asarray(lab, dtype=np.float64)
    a_map = _LMS_MAPS[kind]
    root = lab @ _M2_INV.T  # cube-rooted LMS
    seen = (root**3) @ a_map.T
    seen_root = np.cbrt(seen)
    out = seen_root @ _M2.T
    if not jacobian:
        return out
    inner = 3.0 * root**2
    outer = 1.0 / (3.0 * np.maximum(seen_root**2, _MIN_CBRT_SQ))
    # J = M2 . diag(outer) . A . diag(inner) . M2^-1
    jac = (
        _M2
        @ (outer[..., :, None] * (a_map @ (inner[..., :, None] * _M2_INV)))
    )
    return out, jac
