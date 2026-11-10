"""Compare palette() with baselines: python benchmarks/benchmark.py [--restarts 4] [--max-n 30]

Prints, for each background and n, the min ΔE_OK (including the background) of
palette(), the spread across seeds, runtime, and the baselines.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from baselines import NAMED, farthest_point_min_distance, named_min_distance, random_min_distance

from colorcat import palette
from colorcat._constraints import get_constraints
from colorcat._energy import min_distance_with_anchors
from colorcat._format import hex_to_oklab


def score(colors: list[str], bg: np.ndarray) -> float:
    return min_distance_with_anchors(np.array([hex_to_oklab(c) for c in colors]), bg[None, :])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--max-n", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    for background in ("white", "black"):
        c = get_constraints(background)
        print(f"\n=== background: {background}  (restart_count={args.restarts}) ===")
        print(
            f"{'n':>3} {'palette':>8} {'seeds min..max':>15} {'sec':>6} {'random':>8} {'farthest':>9}  named"
        )
        for n in list(range(2, 13)) + list(range(14, args.max_n + 1, 2)):
            t = time.perf_counter()
            main_score = score(
                palette(n, background=background, restart_count=args.restarts), c.background_lab
            )
            sec = time.perf_counter() - t
            seeds = [
                score(
                    palette(n, background=background, restart_count=args.restarts, seed=s),
                    c.background_lab,
                )
                for s in range(1, args.seeds + 1)
            ]
            named = " ".join(
                f"{k}={named_min_distance(k, n, c.background_lab):.3f}"
                for k in NAMED
                if n <= len(NAMED[k])
            )
            print(
                f"{n:>3} {main_score:>8.4f} {min(seeds):>7.4f}..{max(seeds):.4f} {sec:>6.2f} "
                f"{random_min_distance(n, c):>8.4f} {farthest_point_min_distance(n, c):>9.4f}  {named}",
                flush=True,
            )


if __name__ == "__main__":
    main()
