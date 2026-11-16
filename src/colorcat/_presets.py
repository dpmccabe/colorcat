"""Precomputed palettes for the standard usage pattern.

``presets.json`` is written by ``benchmarks/presets.py``: palettes for every
``n`` up to 15 on each background, optimized with many restarts
using the default ``min_contrast``, ``seed`` and no ``exclude``, ``fixed`` or ``cvd``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

_PATH = Path(__file__).with_name("presets.json")


@cache
def load() -> dict:
    return json.loads(_PATH.read_text())


def preset_palette(n: int, background: str) -> list[str] | None:
    """The saved hex palette of ``n`` colors for ``background``, or ``None`` if there is none."""
    colors = load()["palettes"].get(background, {}).get(str(n))
    return list(colors) if colors is not None else None
