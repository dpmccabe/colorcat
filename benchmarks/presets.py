"""Precompute palettes for the standard usage patterns.

Runs :func:`colorcat.palette` with ``restart_count=20`` for ``n = 1..15`` on both
backgrounds, using the default ``min_contrast`` and seed and no other options, and
writes the results to ``src/colorcat/presets.json`` so the package can return them
without optimizing. Rerun this whenever the optimizer or its defaults change:

    uv run python benchmarks/presets.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from colorcat import min_distance, palette
from colorcat._constraints import DEFAULT_MIN_CONTRAST

NS = range(1, 16)
BACKGROUNDS = ("white", "black")
RESTART_COUNT = 20
SEED = 0
OUT = Path(__file__).resolve().parent.parent / "src" / "colorcat" / "presets.json"


def main() -> None:
    presets: dict[str, dict[str, list[str]]] = {}
    for background in BACKGROUNDS:
        presets[background] = {}
        for n in NS:
            start = time.perf_counter()
            colors = palette(
                n,
                background=background,
                min_contrast=DEFAULT_MIN_CONTRAST,
                restart_count=RESTART_COUNT,
                seed=SEED,
            )
            presets[background][str(n)] = colors
            print(
                f"{background:5} n={n:2}  min ΔE {min_distance(colors):.4f}  "
                f"{time.perf_counter() - start:5.1f}s  {' '.join(colors)}"
            )
    data = {
        "min_contrast": DEFAULT_MIN_CONTRAST,
        "restart_count": RESTART_COUNT,
        "seed": SEED,
        "palettes": presets,
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
