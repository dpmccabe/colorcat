"""Sweep the optimizer's hard-coded hyperparameters and write a markdown report.

    python benchmarks/tune.py [--quick] [--seeds 3] [--ns 3 5 8 12 20 30] [--jobs 4]
                              [--groups P_SCHEDULE PATIENCE ...] [--out benchmarks/tune.md]

Method: one-at-a-time sweeps around the current defaults. Every candidate is
run on the same (background, n, seed) cases as the current defaults, and its
min ΔE_OK (including the background, after 8-bit quantization) is divided by the
default's score on that case. The report gives, per candidate, the mean of that
ratio (>100% = better than current), its standard error, wins/ties/losses, and
the mean runtime relative to the current defaults.

The built-in L bands are NOT swept: changing them changes the
feasible set, so min-distance scores would not be comparable.

Candidates are applied by overriding the constants in ``colorcat._energy`` inside
worker processes (the library itself is never modified). Runtimes are measured
with several workers running at once, so read them as relative, not absolute.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

from colorcat import _energy, palette
from colorcat._constraints import get_constraints
from colorcat._energy import min_distance_with_anchors
from colorcat._format import hex_to_oklab

# Group name -> candidate values. A value equal to the current default is
# reported using the shared baseline run rather than being run twice.
# "restart_count" is a palette() argument; everything else is a _energy constant.
SWEEPS: dict[str, list] = {
    "P_SCHEDULE": [
        (16.0,),
        (8.0, 32.0),
        (8.0, 16.0, 32.0),
        (4.0, 16.0, 64.0),
        (4.0, 8.0, 16.0, 32.0),
        (2.0, 4.0, 8.0, 16.0, 32.0),
        (4.0, 8.0, 16.0, 32.0, 64.0),
        (4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
    ],
    "MAX_ITERS_PER_STAGE": [50, 100, 150, 300, 600],
    "STOP_TOL": [1e-5, 1e-4, 1e-3],
    "PATIENCE": [5, 10, 20, 40],
    "INITIAL_STEP": [0.005, 0.01, 0.02, 0.05, 0.1],
    "BACKTRACK_FACTOR": [0.3, 0.5, 0.7],
    "HOP_COUNT": [0, 4, 8, 16, 32],
    "HOP_P_SCHEDULE": [(32.0,), (16.0, 32.0), (8.0, 16.0, 32.0), (4.0, 8.0, 16.0, 32.0)],
    "HOP_CLOSE": [1.0, 1.05, 1.3, 2.0],
    "restart_count": [1, 2, 4, 8, 16, 32],
}

Case = tuple[str, int, int]  # (background, n, seed)


def current_value(name: str):
    if name == "restart_count":
        return inspect.signature(palette).parameters["restart_count"].default
    return getattr(_energy, name)


def run_case(overrides: dict, case: Case) -> tuple[float, float]:
    """Score one palette under ``overrides``; returns (min ΔE_OK, seconds)."""
    background, n, seed = case
    saved = {k: getattr(_energy, k) for k in overrides if k != "restart_count"}
    restarts = overrides.get("restart_count", current_value("restart_count"))
    try:
        for k, v in overrides.items():
            if k != "restart_count":
                setattr(_energy, k, v)
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        colors = palette(n, background=background, restart_count=restarts, seed=seed)
        sec = time.perf_counter() - t
    finally:
        for k, v in saved.items():
            setattr(_energy, k, v)
    lab = np.array([hex_to_oklab(c) for c in colors])
    bg = get_constraints(background).background_lab[None, :]
    return min_distance_with_anchors(lab, bg), sec


def fmt_value(v) -> str:
    if isinstance(v, tuple):
        return "(" + ", ".join(f"{x:g}" for x in v) + ")"
    return f"{v:g}" if isinstance(v, float) else str(v)


def summarize(scores: dict[Case, float], secs: dict[Case, float], base: dict[Case, float], base_secs: dict[Case, float]):
    cases = list(base)
    ratios = np.array([scores[c] / base[c] for c in cases])
    n = len(ratios)
    se = ratios.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    wins = int((ratios > 1 + 1e-9).sum())
    losses = int((ratios < 1 - 1e-9).sum())
    return {
        "ratio": ratios.mean(),
        "se": se,
        "wins": wins,
        "ties": n - wins - losses,
        "losses": losses,
        "mean_score": float(np.mean([scores[c] for c in cases])),
        "time_ratio": float(np.mean([secs[c] for c in cases]) / np.mean([base_secs[c] for c in cases])),
        "mean_sec": float(np.mean([secs[c] for c in cases])),
        "by_n": {
            n_: float(np.mean([scores[c] / base[c] for c in cases if c[1] == n_]))
            for n_ in sorted({c[1] for c in cases})
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer n values and seeds (smoke test)")
    ap.add_argument("--seeds", type=int, default=3, help="seeds per (background, n) case")
    ap.add_argument("--ns", type=int, nargs="+", default=[3, 5, 8, 12, 20, 30])
    ap.add_argument("--backgrounds", nargs="+", default=["white", "black"], choices=["white", "black"])
    ap.add_argument("--groups", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "tune.md")
    args = ap.parse_args()
    if args.quick:
        args.ns, args.seeds = [3, 8, 20], 2

    cases: list[Case] = [(b, n, s) for b in args.backgrounds for n in args.ns for s in range(args.seeds)]

    # Unique configs: the baseline plus every non-default candidate.
    configs: dict[tuple, dict] = {(): {}}
    for group in args.groups:
        for v in SWEEPS[group]:
            if v != current_value(group):
                configs[((group, v),)] = {group: v}

    tasks = [(key, c) for key in configs for c in cases]
    print(f"{len(configs)} configs x {len(cases)} cases = {len(tasks)} palettes, {args.jobs} workers", flush=True)

    results: dict[tuple, dict[Case, tuple[float, float]]] = {key: {} for key in configs}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [(key, c, pool.submit(run_case, configs[key], c)) for key, c in tasks]
        for i, (key, c, fut) in enumerate(futures, 1):
            results[key][c] = fut.result()
            if i % 25 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)}  {time.perf_counter() - start:.0f}s", flush=True)

    def split(key):
        return {c: r[0] for c, r in results[key].items()}, {c: r[1] for c, r in results[key].items()}

    base_scores, base_secs = split(())
    summaries = {key: summarize(*split(key), base_scores, base_secs) for key in configs}
    base_summary = summaries[()]

    lines: list[str] = []
    w = lines.append
    w("# colorcat hyperparameter tuning")
    w("")
    w(f"Generated {date.today()} by `benchmarks/tune.py` on {platform.platform()}, "
      f"Python {platform.python_version()}, NumPy {np.__version__}.")
    w("")
    w("## Setup")
    w("")
    w(f"- Backgrounds: {', '.join(args.backgrounds)}; n: {', '.join(map(str, args.ns))}; "
      f"seeds: 0-{args.seeds - 1} ({len(cases)} cases per candidate).")
    w("- Score: min ΔE_OK between any two colors or a color and the background, after 8-bit "
      "quantization (higher is better).")
    w("- **vs current** = mean over cases of `candidate score / current-default score`; "
      "above 100% is better than the current defaults. **±** is the standard error of that mean, "
      "so differences smaller than about 2× it are likely noise.")
    w("- **W/T/L** counts cases where the candidate scored higher / equal / lower than the current defaults.")
    w("- **time** is mean runtime relative to the current defaults (measured with parallel workers: "
      "relative values are meaningful, absolute seconds are not).")
    w("- Sweeps change one setting at a time with everything else at its current default. "
      "Interactions are not explored.")
    w("")
    w("Current defaults:")
    w("")
    for name in SWEEPS:
        w(f"- `{name}` = {fmt_value(current_value(name))}")
    w("")
    w(f"Current-default mean score over these cases: **{base_summary['mean_score']:.4f}**, "
      f"mean time {base_summary['mean_sec']:.2f} s per palette.")
    w("")

    ns = sorted({c[1] for c in cases})
    for group in args.groups:
        w(f"## `{group}`")
        w("")
        w("| value | vs current | ± | W/T/L | time | " + " | ".join(f"n={n}" for n in ns) + " |")
        w("|---|---|---|---|---|" + "---|" * len(ns))
        for v in SWEEPS[group]:
            is_default = v == current_value(group)
            s = base_summary if is_default else summaries[((group, v),)]
            label = f"{fmt_value(v)} **(current)**" if is_default else fmt_value(v)
            se = "" if is_default else f"{100 * s['se']:.2f}"
            wtl = "" if is_default else f"{s['wins']}/{s['ties']}/{s['losses']}"
            by_n = " | ".join(f"{100 * s['by_n'][n]:.1f}%" for n in ns)
            w(f"| {label} | {100 * s['ratio']:.2f}% | {se} | {wtl} | {s['time_ratio']:.2f}x | {by_n} |")
        w("")

    # Ranked shortlist: best quality per candidate, flagged by cost.
    w("## All candidates ranked by quality")
    w("")
    w("| rank | setting | vs current | ± | time |")
    w("|---|---|---|---|---|")
    ranked = sorted(
        (k for k in configs if k), key=lambda k: summaries[k]["ratio"], reverse=True
    )
    for i, key in enumerate(ranked, 1):
        (name, v), = key
        s = summaries[key]
        w(f"| {i} | `{name}` = {fmt_value(v)} | {100 * s['ratio']:.2f}% | {100 * s['se']:.2f} | {s['time_ratio']:.2f}x |")
    w("")
    w("## How to choose")
    w("")
    w("- Prefer a change only if its gain is clearly larger than **±** and it wins more cases than it loses.")
    w("- Gains that cost much more **time** are a trade-off; `restart_count` is the user-facing knob for that, "
      "so prefer fixing quality per restart first.")
    w("- Check the per-`n` columns: a setting that helps large `n` but hurts small `n` (or vice versa) "
      "shows up there, not in the overall mean.")
    w("- After changing defaults, re-run this script (candidates are always compared with whatever the "
      "defaults are at run time) and `uv run pytest`; the reproducibility tests do not pin exact colors, "
      "but the README example and gallery would change.")
    w("")

    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
