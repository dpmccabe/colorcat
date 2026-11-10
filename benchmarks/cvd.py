"""Show what every combination of ``cvd`` options does to a palette.

    python benchmarks/cvd.py [--n 4] [--background white] [--seeds 3] [--restarts 32]
                             [--out benchmarks/cvd.md]

For each subset of {protan, deutan, tritan} (including none) this builds
``palette(n, cvd=subset)`` and reports, in the same table, the smallest pairwise
ΔE_OK (including the background) seen with normal vision and with each
deficiency. The "design" column is the worst of the views the subset asked for,
which is what the optimizer maximizes. Hex codes and the swatch image
(``cvd.svg``, next to the markdown) are for seed 0; the distances are also
averaged over ``--seeds`` seeds.
"""

from __future__ import annotations

import argparse
import itertools
import platform
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np

from colorcat import palette
from colorcat._constraints import get_constraints
from colorcat._cvd import CVD_TYPES, simulate
from colorcat._energy import min_distance_with_anchors
from colorcat._oklab import oklab_to_srgb
from colorcat._format import hex_to_oklab

VIEWS = ("normal", *CVD_TYPES)


def subsets():
    for k in range(len(CVD_TYPES) + 1):
        yield from itertools.combinations(CVD_TYPES, k)


def view_distances(colors: list[str], background: str) -> dict[str, float]:
    """Smallest pairwise ΔE_OK (with the background) in each single view."""
    lab = np.array([hex_to_oklab(c) for c in colors])
    bg = get_constraints(background).background_lab[None, :]
    return {
        v: min_distance_with_anchors(lab, bg, () if v == "normal" else (v,))
        for v in VIEWS
    }


def label(combo: tuple[str, ...]) -> str:
    return ", ".join(combo) if combo else "(none)"


def swatch_svg(rows, background: str) -> str:
    """One row per subset; columns: as designed (normal) and as seen with each deficiency."""
    cell, gap, pad, label_w, row_gap = 34, 3, 12, 170, 14
    n = len(rows[0][1])
    group_w = n * cell + (n - 1) * gap
    col_gap = 22
    width = pad * 2 + label_w + len(VIEWS) * group_w + (len(VIEWS) - 1) * col_gap
    head = 26
    height = pad * 2 + head + len(rows) * (cell + row_gap)
    bg = "#ffffff" if background == "white" else "#000000"
    fg = "#222222" if background == "white" else "#dddddd"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
    ]
    for j, v in enumerate(VIEWS):
        x = pad + label_w + j * (group_w + col_gap)
        title = "normal vision" if v == "normal" else f"as seen with {v}"
        out.append(
            f'<text x="{x}" y="{pad + 12}" fill="{fg}" font-weight="bold">{title}</text>'
        )
    for i, (combo, colors) in enumerate(rows):
        y = pad + head + i * (cell + row_gap)
        out.append(
            f'<text x="{pad}" y="{y + cell / 2 + 4}" fill="{fg}">cvd = {label(combo)}</text>'
        )
        lab = np.array([hex_to_oklab(c) for c in colors])
        for j, v in enumerate(VIEWS):
            seen = lab if v == "normal" else simulate(lab, v)
            rgb = np.clip(np.round(oklab_to_srgb(seen) * 255), 0, 255).astype(int)
            for k, (r, g, b) in enumerate(rgb):
                x = pad + label_w + j * (group_w + col_gap) + k * (cell + gap)
                out.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3" '
                    f'fill="#{r:02x}{g:02x}{b:02x}"/>'
                )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--background", choices=["white", "black"], default="white")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "cvd.md")
    args = parser.parse_args()

    rows, stats = [], []
    start = time.perf_counter()
    for combo in subsets():
        per_seed = []
        for seed in range(args.seeds):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                colors = palette(
                    args.n,
                    background=args.background,
                    restart_count=args.restarts,
                    cvd=combo,
                    seed=seed,
                )
            if seed == 0:
                rows.append((combo, colors))
            per_seed.append(view_distances(colors, args.background))
        mean = {v: float(np.mean([d[v] for d in per_seed])) for v in VIEWS}
        asked = ["normal", *combo]
        design = float(np.mean([min(d[v] for v in asked) for d in per_seed]))
        stats.append(
            (combo, mean, design, view_distances(rows[-1][1], args.background))
        )
        print(f"{label(combo):28s} design={design:.3f}", flush=True)
    elapsed = time.perf_counter() - start

    base = stats[0][2]
    svg_path = args.out.with_suffix(".svg")
    svg_path.write_text(swatch_svg(rows, args.background))

    lines = [
        f"# CVD options: n={args.n}, {args.background} background",
        "",
        f"Generated {date.today()} with `benchmarks/cvd.py` "
        f"(restart_count={args.restarts}, {args.seeds} seeds, Python {platform.python_version()}, "
        f"{elapsed:.0f} s).",
        "",
        "Smallest pairwise ΔE_OK between any two colors or a color and the background, "
        "mean over seeds. Bold = a view the optimizer was asked to protect. "
        "*design* is the worst of the requested views (normal vision always counts), "
        "and *vs. none* compares it with the unconstrained palette.",
        "",
        "| `cvd` | design | vs. none | normal | protan | deutan | tritan | colors (seed 0) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (combo, mean, design, _), (_, colors) in zip(stats, rows):
        cells = []
        for v in VIEWS:
            text = f"{mean[v]:.3f}"
            cells.append(f"**{text}**" if v == "normal" or v in combo else text)
        lines.append(
            f"| {label(combo)} | {design:.3f} | {design / base:.0%} | "
            + " | ".join(cells)
            + " | "
            + " ".join(f"`{c}`" for c in colors)
            + " |"
        )
    lines += [
        "",
        f"![Palettes as seen with each deficiency]({svg_path.name})",
        "",
        "Each row is one `cvd` setting; the columns are the same palette as seen with "
        "normal vision and simulated for each deficiency (Machado et al. 2009, full "
        "dichromacy).",
        "",
    ]
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out} and {svg_path}")


if __name__ == "__main__":
    main()
