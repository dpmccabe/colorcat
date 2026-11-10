# colorcat hyperparameter tuning

Generated 2026-11-10 by `benchmarks/tune.py` on macOS-26.7.1-arm64-arm-64bit, Python 3.12.12, NumPy 2.5.4.

## Setup

- Backgrounds: white, black; n: 3, 5, 8, 12, 20, 30; seeds: 0-2 (36 cases per candidate).
- Score: min ΔE_OK between any two colors or a color and the background, after 8-bit quantization (higher is better).
- **vs current** = mean over cases of `candidate score / current-default score`; above 100% is better than the current defaults. **±** is the standard error of that mean, so differences smaller than about 2× it are likely noise.
- **W/T/L** counts cases where the candidate scored higher / equal / lower than the current defaults.
- **time** is mean runtime relative to the current defaults (measured with parallel workers: relative values are meaningful, absolute seconds are not).
- Sweeps change one setting at a time with everything else at its current default. Interactions are not explored.

Current defaults:

- `P_SCHEDULE` = (4, 8, 16, 32)
- `MAX_ITERS_PER_STAGE` = 150
- `STOP_TOL` = 0.0001
- `PATIENCE` = 10
- `INITIAL_STEP` = 0.02
- `BACKTRACK_FACTOR` = 0.5
- `restart_count` = 16

Current-default mean score over these cases: **0.2364**, mean time 1.47 s per palette.

## `P_SCHEDULE`

| value | vs current | ± | W/T/L | time | n=3 | n=5 | n=8 | n=12 | n=20 | n=30 |
|---|---|---|---|---|---|---|---|---|---|---|
| (16) | 98.14% | 0.60 | 10/0/26 | 0.48x | 98.5% | 100.0% | 100.5% | 97.2% | 96.8% | 95.8% |
| (8, 32) | 99.70% | 0.43 | 18/0/18 | 0.72x | 99.2% | 100.5% | 100.2% | 99.0% | 99.8% | 99.6% |
| (8, 16, 32) | 99.83% | 0.48 | 21/0/15 | 0.77x | 100.3% | 100.6% | 100.9% | 99.0% | 99.5% | 98.7% |
| (4, 16, 64) | 100.59% | 0.23 | 25/2/9 | 0.86x | 100.4% | 100.5% | 102.1% | 100.1% | 99.9% | 100.6% |
| (4, 8, 16, 32) **(current)** | 100.00% |  |  | 1.00x | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| (2, 4, 8, 16, 32) | 98.85% | 0.63 | 16/0/20 | 1.17x | 96.6% | 99.1% | 101.2% | 98.8% | 100.7% | 96.8% |
| (4, 8, 16, 32, 64) | 100.71% | 0.14 | 23/11/2 | 1.10x | 100.3% | 101.0% | 101.1% | 100.6% | 100.8% | 100.5% |
| (4, 8, 16, 32, 64, 128) | 100.96% | 0.16 | 29/5/2 | 1.25x | 100.4% | 101.5% | 101.4% | 100.8% | 101.0% | 100.7% |

## `MAX_ITERS_PER_STAGE`

| value | vs current | ± | W/T/L | time | n=3 | n=5 | n=8 | n=12 | n=20 | n=30 |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 99.98% | 0.09 | 4/27/5 | 0.89x | 100.3% | 100.1% | 100.0% | 100.0% | 99.7% | 99.8% |
| 100 | 100.01% | 0.01 | 2/34/0 | 0.91x | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 150 **(current)** | 100.00% |  |  | 1.00x | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 300 | 100.00% | 0.00 | 0/36/0 | 0.92x | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 600 | 100.00% | 0.00 | 0/36/0 | 0.92x | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## `STOP_TOL`

| value | vs current | ± | W/T/L | time | n=3 | n=5 | n=8 | n=12 | n=20 | n=30 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-05 | 100.00% | 0.00 | 0/35/1 | 0.91x | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.0001 **(current)** | 100.00% |  |  | 1.00x | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.001 | 99.92% | 0.08 | 3/27/6 | 0.92x | 100.0% | 100.0% | 100.1% | 100.0% | 100.0% | 99.4% |

## `PATIENCE`

| value | vs current | ± | W/T/L | time | n=3 | n=5 | n=8 | n=12 | n=20 | n=30 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 99.76% | 0.28 | 15/12/9 | 0.82x | 100.5% | 100.1% | 99.9% | 99.0% | 99.6% | 99.5% |
| 10 **(current)** | 100.00% |  |  | 1.00x | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 20 | 99.83% | 0.15 | 5/25/6 | 0.97x | 100.1% | 100.0% | 99.9% | 100.0% | 100.0% | 99.0% |
| 40 | 99.82% | 0.15 | 5/25/6 | 0.98x | 100.1% | 99.9% | 99.9% | 100.0% | 100.0% | 99.0% |

## `INITIAL_STEP`

| value | vs current | ± | W/T/L | time | n=3 | n=5 | n=8 | n=12 | n=20 | n=30 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 99.98% | 0.42 | 18/0/18 | 1.24x | 100.3% | 99.9% | 101.6% | 99.4% | 99.4% | 99.3% |
| 0.01 | 99.83% | 0.28 | 17/0/19 | 0.99x | 100.5% | 99.9% | 101.1% | 98.5% | 100.2% | 98.8% |
| 0.02 **(current)** | 100.00% |  |  | 1.00x | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.05 | 99.68% | 0.32 | 15/0/21 | 0.86x | 99.8% | 100.8% | 99.7% | 98.6% | 100.0% | 99.2% |
| 0.1 | 100.12% | 0.47 | 16/0/20 | 0.92x | 99.7% | 101.3% | 100.7% | 98.6% | 100.6% | 99.8% |

## `BACKTRACK_FACTOR`

| value | vs current | ± | W/T/L | time | n=3 | n=5 | n=8 | n=12 | n=20 | n=30 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 99.56% | 0.18 | 8/14/14 | 0.69x | 100.0% | 100.0% | 100.0% | 100.1% | 98.5% | 98.7% |
| 0.5 **(current)** | 100.00% |  |  | 1.00x | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.7 | 99.95% | 0.18 | 10/16/10 | 1.30x | 100.0% | 100.0% | 99.8% | 100.0% | 99.6% | 100.4% |

## `restart_count`

| value | vs current | ± | W/T/L | time | n=3 | n=5 | n=8 | n=12 | n=20 | n=30 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 96.07% | 0.97 | 0/14/22 | 0.23x | 89.3% | 95.4% | 98.8% | 96.4% | 98.4% | 98.1% |
| 8 | 98.63% | 0.55 | 0/23/13 | 0.45x | 97.0% | 99.2% | 99.5% | 99.1% | 98.7% | 98.3% |
| 16 **(current)** | 100.00% |  |  | 1.00x | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 32 | 101.76% | 0.55 | 14/22/0 | 1.80x | 102.9% | 104.0% | 100.8% | 100.0% | 101.6% | 101.2% |
| 64 | 102.82% | 0.67 | 23/13/0 | 3.84x | 105.3% | 104.5% | 102.1% | 100.3% | 103.0% | 101.6% |

## All candidates ranked by quality

| rank | setting | vs current | ± | time |
|---|---|---|---|---|
| 1 | `restart_count` = 64 | 102.82% | 0.67 | 3.84x |
| 2 | `restart_count` = 32 | 101.76% | 0.55 | 1.80x |
| 3 | `P_SCHEDULE` = (4, 8, 16, 32, 64, 128) | 100.96% | 0.16 | 1.25x |
| 4 | `P_SCHEDULE` = (4, 8, 16, 32, 64) | 100.71% | 0.14 | 1.10x |
| 5 | `P_SCHEDULE` = (4, 16, 64) | 100.59% | 0.23 | 0.86x |
| 6 | `INITIAL_STEP` = 0.1 | 100.12% | 0.47 | 0.92x |
| 7 | `MAX_ITERS_PER_STAGE` = 100 | 100.01% | 0.01 | 0.91x |
| 8 | `MAX_ITERS_PER_STAGE` = 300 | 100.00% | 0.00 | 0.92x |
| 9 | `MAX_ITERS_PER_STAGE` = 600 | 100.00% | 0.00 | 0.92x |
| 10 | `STOP_TOL` = 1e-05 | 100.00% | 0.00 | 0.91x |
| 11 | `INITIAL_STEP` = 0.005 | 99.98% | 0.42 | 1.24x |
| 12 | `MAX_ITERS_PER_STAGE` = 50 | 99.98% | 0.09 | 0.89x |
| 13 | `BACKTRACK_FACTOR` = 0.7 | 99.95% | 0.18 | 1.30x |
| 14 | `STOP_TOL` = 0.001 | 99.92% | 0.08 | 0.92x |
| 15 | `INITIAL_STEP` = 0.01 | 99.83% | 0.28 | 0.99x |
| 16 | `PATIENCE` = 20 | 99.83% | 0.15 | 0.97x |
| 17 | `P_SCHEDULE` = (8, 16, 32) | 99.83% | 0.48 | 0.77x |
| 18 | `PATIENCE` = 40 | 99.82% | 0.15 | 0.98x |
| 19 | `PATIENCE` = 5 | 99.76% | 0.28 | 0.82x |
| 20 | `P_SCHEDULE` = (8, 32) | 99.70% | 0.43 | 0.72x |
| 21 | `INITIAL_STEP` = 0.05 | 99.68% | 0.32 | 0.86x |
| 22 | `BACKTRACK_FACTOR` = 0.3 | 99.56% | 0.18 | 0.69x |
| 23 | `P_SCHEDULE` = (2, 4, 8, 16, 32) | 98.85% | 0.63 | 1.17x |
| 24 | `restart_count` = 8 | 98.63% | 0.55 | 0.45x |
| 25 | `P_SCHEDULE` = (16) | 98.14% | 0.60 | 0.48x |
| 26 | `restart_count` = 4 | 96.07% | 0.97 | 0.23x |

## How to choose

- Prefer a change only if its gain is clearly larger than **±** and it wins more cases than it loses.
- Gains that cost much more **time** are a trade-off; `restart_count` is the user-facing knob for that, so prefer fixing quality per restart first.
- Check the per-`n` columns: a setting that helps large `n` but hurts small `n` (or vice versa) shows up there, not in the overall mean.
- After changing defaults, re-run this script (candidates are always compared with whatever the defaults are at run time) and `uv run pytest`; the reproducibility tests do not pin exact colors, but the README example and gallery would change.
