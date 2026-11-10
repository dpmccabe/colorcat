# colorcat — Plan

A Python package exposing `palette(n, ...)`, which returns `n` maximally distinct colors for categorical data (e.g. chart series, class labels).

## Goal

Choose `n` colors in a perceptually uniform space so that the *collective* set is as mutually distinct as possible. Each `n` gets its own optimization: quality of the palette *for that n* matters more than consistency between palettes of different sizes.

v1 is deliberately small: the user picks a `white` or `black` background, and the package picks good lightness/chroma/hue ranges and optimizer settings itself. Everything tunable is hard-coded until there is evidence for exposing it (see v2).

## Scope by version

| | v1 | v2 |
|---|---|---|
| Energy minimization / repulsion (multi-restart, NumPy only) | ✅ | |
| `background="white"` or `"black"`, selecting built-in LCH ranges | ✅ | |
| `restart_count` argument | ✅ | |
| `fixed`, `exclude`, output formats, CLI | ✅ | |
| User-set `lightness` / `chroma` / `hue` overrides | | ✅ |
| Arbitrary `background` colors (with derived ranges) | | ✅ |
| Other optimizer hyperparameters exposed | | maybe |
| Farthest-point sampling (`method="farthest"`) | | ✅ |
| Maximin refinement (discrete local search) | | ✅ |
| `method=` argument | | ✅ (no choice to make in v1) |
| `min_background_contrast` (WCAG contrast ratio) | | ✅ |
| `cvd=` (color-vision-deficiency awareness) | ✅ (added after v1) | ✅ |

## Public API (v1)

```python
def palette(
    n: int,
    *,
    background: Literal["white", "black"] = "white",  # the main knob; selects built-in LCH ranges
    restart_count: int = 16,                          # independent optimizer restarts; more = better, slower
    exclude: Sequence[str] = (),                      # hex colors that must be kept distinct from
    fixed: Sequence[str] = (),                        # hex colors already in use; new colors extend them
    color_format: Literal["hex", "rgb", "oklab", "oklch"] = "hex",
    seed: int | None = 0,                             # same args + same seed -> same palette
) -> list[str] | list[tuple]
```

Design notes:
- **Typical use is `palette(n)` or `palette(n, background="black")` and nothing else.** There are no lightness/chroma/hue arguments in v1; the ranges are chosen by the package (see "Built-in color ranges").
- **`restart_count`** (default **4**; each restart is followed by `HOP_COUNT=16` basin-hopping moves, see Algorithm. At roughly equal runtime, 4 restarts x 16 hops beat 32 plain restarts by about 1-3% min distance. It was 32 before hopping was added.) is the number of independent optimization runs; the best result is kept. Rationale for 32 (raised from an initial 16 after `benchmarks/tune.py`, which showed about +2.5% min distance for ~1.6x the time): the energy landscape has many local optima and restarts are the main quality lever, restarts are cheap for typical `n` (and vectorizable as a batch dimension), and the quality gain flattens out past roughly this point. The default is confirmed or adjusted by the benchmark (milestone 4) so that typical `n` stays under about a second. Validated as an integer ≥ 1.
- **Each `n` is optimized independently.** The best 4-color palette can look quite different from the best 14-color one. There is no prefix stability: `palette(4)` is not `palette(14)[:4]`, and no first color is pinned.
- **The seed controls all randomness and nothing else.** The only random choices are the starting positions of the restarts. Restart `i` draws its start from its own child stream, `SeedSequence(seed).spawn(restart_count)[i]`, so a restart's result does not depend on `restart_count`, and **raising `restart_count` can never make a palette worse** (the extra restarts only add candidates). Same arguments + same seed always returns the identical palette (reproducible charts); a different seed may converge to a different, similarly good palette. `seed=None` draws fresh entropy. The default is `0`.
- **Output order:** the optimizer returns an unordered set; it is then ordered farthest-first, starting with the color most distant from the background, so adjacent legend entries are not near-duplicates and the first series gets the strongest contrast. Order is not a stability guarantee across `n`.
- **Reserved for v2:** `lightness`, `chroma`, `hue`, other background colors, `method=`, `min_background_contrast`, `cvd`. v1 does not accept them, but the internals are built so they slot in without an API break (the constraint set is already a data object, see `_constraints.py`).
- Validate: `n >= 0` (integer), `background` is exactly `"white"` or `"black"` (anything else raises `ValueError` listing the allowed values), `restart_count >= 1`, `fixed`/`exclude` parse as hex colors. Warn (don't fail) if the achieved min distance is very low (e.g. very large `n`).
- `n=0` returns `[]`. `n=1` works naturally (the one color is repelled only by the anchors).
- Companion helper: `min_distance(colors) -> float` (ΔE_OK of the closest pair) to quantify quality.

## Color space

**Recommendation: OKLab / OKLCh** (Ottosson, 2020).

| Space | Pros | Cons |
|---|---|---|
| **OKLab / OKLCh** | Perceptually uniform, cheap closed-form conversion from sRGB, good hue linearity, L/C/h are natural knobs for the constraints | Euclidean distance is only approximately ΔE; less rigorous than CAM16-UCS at large distances |
| CIELAB (ΔE76/94) | Ubiquitous | Hue non-linearity (blue→purple), uneven uniformity |
| CIEDE2000 | Best small-difference metric | Expensive, non-Euclidean (not a true metric), awkward in optimizers; designed for small differences, not categorical-scale ones |
| CAM16-UCS | Very good uniformity, models viewing conditions | Heavier, more parameters, minimal gain here |
| HSL/HSV | Easy | Not perceptual; do not use |

Distance metric: Euclidean in OKLab (ΔE_OK). It is also what makes the energy method's gradients simple and analytic. Optionally offer `metric="ciede2000"` later (v2) for post-hoc evaluation only.

## Built-in color ranges

`background` selects one of two hard-coded constraint sets (`_constraints.py`), each a frozen `Constraints` dataclass. The background color itself is exactly `#ffffff` or `#000000`.

| | `"white"` | `"black"` |
|---|---|---|
| Lightness (OKLab L) | 0.40 – 0.75 | 0.60 – 0.90 |
| Chroma (OKLCh C) | no bounds; limited only by the sRGB gamut | same |
| Hue | 0° – 360° (full circle) | same |

- **Lightness:** the band sits on the opposite side of mid-grey from the background so marks always read clearly. On white, the upper bound keeps colors clearly darker than the background and the lower bound avoids near-black marks that read as text or axes. On black it is mirrored.
- **Chroma, no bounds:** there is no chroma floor or ceiling; the sRGB gamut at each (L, h) is the only limit. (An earlier 0.09 floor was removed: maximizing the minimum distance already pushes colors toward the gamut surface, so greys are rare except at large `n`, and the floor rarely bound.)
- **Full hue circle:** nothing is excluded by default.
- **These numbers are starting values**, to be tuned visually (swatch strips on white and black) during milestone 2. Because users cannot override them in v1, getting them right matters; they live in one place so they are easy to change.

**Background in the objective.** The background color is a fixed (non-movable) repeller alongside `fixed` and `exclude` colors, so every chosen color is pushed away from the background as strongly as it is from the other colors. Together with the lightness band, this keeps marks visible without a separate contrast rule. (A WCAG contrast-ratio option is deferred to v2; see below.)

If `fixed` colors lie outside the built-in ranges, that is allowed: they are anchors and are never moved or validated against the ranges.

## Feasible region

The feasible set is the intersection of {L range} × × {hue range} × {sRGB gamut} for the selected background, with a small inset margin (0.002 on each L bound and on the gamut ceiling) so colors still satisfy the constraints after 8-bit quantization.

`_feasible.py` provides three operations on it, all vectorized NumPy:
- **`gamut_max_chroma(L, h)`:** the largest in-gamut chroma at a given lightness and hue, found by vectorized bisection on chroma (convert to linear sRGB, test 0–1 bounds). Can be tabulated on a coarse (L, h) grid and interpolated if profiling says so.
- **`project(points)`:** map arbitrary OKLab points onto the feasible region. Clip L to the band; convert to (C, h); clamp C into `[0, gamut_max_chroma(L, h)]`. Must be idempotent (tested). (Hue-window clamping with wraparound is added in v2 along with the `hue` override.)
- **`sample(rng, k)`:** draw `k` uniformly distributed feasible points (rejection sampling in OKLCh).

## Algorithm (v1): energy minimization with restarts

Colors are `n` free particles in OKLab (3 coordinates each), repelled from one another and from fixed anchors (`background`, `exclude`, `fixed`), confined to the feasible region.

**Objective.** A smooth surrogate for maximin (maximize the minimum pairwise distance). With `d_ij` the ΔE_OK between particles (and anchors), minimize the log-domain Riesz energy

`E_p = (1/p) · log Σ_{i<j} d_ij^(-p)`

computed with `logsumexp` for numerical stability. As `p → ∞`, minimizing `E_p` maximizes the minimum distance. Gradients are analytic. Anchor–particle pairs are included; anchor–anchor pairs are constant and skipped.

**Procedure (per restart).**
1. **Init:** `n` points from `sample(rng_i, n)`, where `rng_i` is the restart's own child stream.
2. **Descend with a `p` schedule:** projected gradient descent in plain NumPy. Each step: gradient step, then `project`. Backtracking line search rejects steps that increase `E_p`. Early stop per stage when the improvement in minimum distance is below the tolerance, or at the iteration cap.
3. **Quantize:** snap to the 8-bit sRGB lattice, re-check feasibility, and recompute the true objective: the minimum pairwise distance including anchors (what users see).

**Basin hopping (added after v1).** After a restart's first descent, `HOP_COUNT` (16) moves follow: pick at random one of the colors in the (nearly) closest pairs, re-seed it at the best of 256 random feasible points (the farthest from everything else), re-descend with `p` = 16, 32, and keep the result only if the quantized min distance (then energy) improved. Hops use the restart's own random stream, so determinism and "more restarts is never worse" still hold. Measured gain over plain restarts at equal runtime: about +1-3% (n=5 to 30).

**Restarts.** Run `restart_count` independent restarts and keep the one with the largest quantized minimum distance (ties: lower final `E_p`, then lower restart index, so the choice is deterministic).

**Hard-coded hyperparameters (not exposed in v1).** Module-level constants in `_energy.py`, chosen as sensible starting values and then tuned by the benchmark (n = 2…30, white and black backgrounds), never by users:

| Constant | Starting value | Meaning |
|---|---|---|
| `P_SCHEDULE` | `(4, 8, 16, 32)` | Riesz exponent per stage: low `p` spreads colors globally, high `p` focuses on the closest pair |
| `MAX_ITERS_PER_STAGE` | 150 | iteration cap per `p` stage |
| `STOP_TOL` | 1e-4 | stop a stage when min distance improves less than this (relative) over 10 iterations |
| `INITIAL_STEP` | 0.02 | first gradient step length, in OKLab units |
| `BACKTRACK_FACTOR` | 0.5 | step shrink factor when a step increases `E_p` |
| `FEASIBLE_INSET` | 0.002 | safety margin from range edges for 8-bit rounding |

**Why this fits the goals:**
- Each `n` is optimized as its own problem, in continuous space with no grid discretization error.
- `seed` has a real, natural role: it picks the restart starts, so different seeds give different, comparable palettes.
- `fixed`, `exclude` and `background` are handled uniformly as non-moving anchors.
- Needs only NumPy (no SciPy), and the distance goes through a single function (CVD-ready).

**Risks and mitigations:**
- *Gamut-boundary projection is fiddly* (the optimum often sits on the boundary): property-test `project` for idempotence and feasibility; test near the edges of the L/C bands.
- *Sensitive to the `p` schedule:* tune on the benchmark; constants stay internal so they can change without an API break.
- *Local optima / run-to-run variance:* restarts, plus a test that different seeds land within a small tolerance of each other's minimum distance.
- *Duplicates after quantization:* check after snapping; if two colors collide, nudge one to its best nearby lattice point.

## Dependencies and caching

- **NumPy is the only runtime dependency** (project currently has `dependencies = []`; add `numpy`). No SciPy, no scikit-learn, no color libraries: OKLab conversions (~30 lines) and the optimizer are hand-written on NumPy. Dev-only dependencies (pytest, ruff, matplotlib for the gallery) stay out of runtime deps.
- With only two constraint sets in v1, cache expensive feasible-region artifacts (the feasible pool for `sample`, any gamut table) once per background with a small bounded `functools.lru_cache`. Cached data must be **seed-independent**; all seed-dependent randomness happens after the cache lookup. `fixed`/`exclude` anchors are applied after the lookup, not baked into the cache. Keep a `cvd` slot in the cache key from the start (always `None` in v1) so v2 does not change the cache shape.

## v2 features

### User-tunable constraints and arbitrary backgrounds
- Add `lightness`, `chroma` and `hue` overrides (each `None` → built-in default), replacing only their own value. Hue-window clamping with wraparound joins `project`.
- Allow arbitrary `background` colors (hex or sRGB tuple). Ranges are then *derived* from the background's OKLab lightness by the same logic as the built-in table: light backgrounds → darker band, dark backgrounds → lighter band, and a mid-tone background (roughly 0.45 < L_bg < 0.65, placeholder cutoffs) **raises `ValueError`** telling the user to pass `lightness` explicitly. `"white"` / `"black"` remain valid shorthand.
- Infeasible or too-tight combinations raise a `ValueError` naming the constraints involved; merely cramped ones warn.
- Expose a `resolve_constraints(background, ...)` helper that shows the resolved values.

### Other methods (`method=`)
Exposed via a new `method: Literal["energy", "farthest", "maximin"]` argument, with `"energy"` remaining the default unless benchmarks say otherwise.

**Farthest-point sampling over a discretized gamut.** Build a dense deterministic candidate grid in the feasible region; start from the anchors; repeatedly pick the candidate maximizing its minimum distance to everything chosen so far (O(N·n) with NumPy).
- Pros: very fast, simple, deterministic, prefix-stable (`palette(k)` is a prefix of `palette(n)`), which some users want for consistent color assignment as categories are added.
- Cons: greedy, so not globally optimal (factor-2 max-min-dispersion approximation); early picks bias later ones. It is also effectively Glasbey et al. (2007) in a better color space. Also useful as a candidate initialization or benchmark baseline for energy. A seed-meaningful variant would break near-ties randomly.

**Maximin refinement (discrete local search).** Start from an energy or farthest-point result; remove each color and re-insert it at the grid candidate maximizing the minimum distance to the others, until no improvement. Polishes the last bit of worst-pair distance on the 8-bit lattice.

### CVD awareness
**Implemented** (`_cvd.py`, full dichromacy only; no `cvd_severity` yet). Each view adds its own Riesz term per pair (a soft-min over views), with analytic Jacobians. Exposed as `cvd: Sequence[Literal["protan", "deutan", "tritan"]] | None = None` (maybe a `cvd_severity` float 0–1).

- Simulate each deficiency by a linear transform in linear-RGB (Machado, Oliveira & Fernandes 2009 matrices, severity-interpolated). A few 3×3 matrices, so it fits the NumPy-only constraint.
- The objective's distance becomes `d(a, b) = min over views v ∈ {normal} ∪ cvd of ΔE_OK(sim_v(a), sim_v(b))`, i.e. a pair only counts as distinct if it is distinct under every requested view. The energy objective stays smooth enough (min over a few views, or a soft-min), but the analytic gradients must account for which view is active.
- Expect a tradeoff: larger `n` cannot be fully CVD-safe, so the achieved `min_distance` will drop as views are added; `min_distance` should accept the same `cvd` argument so users can see the cost.
- **v1 preparations so this is cheap to add:** (1) put distance behind a single internal function `_pairwise_distance(a, b)` used by the objective and the helpers (no inlined `np.linalg.norm`), with a matching gradient function next to it; (2) keep particles and anchors in OKLab arrays and convert only at the edges; (3) keep `cvd` in the cache key (always `None` in v1).

### `min_background_contrast` (WCAG)
Deferred. A possible v2 argument: a WCAG 2.x contrast-ratio threshold (e.g. 3:1, the WCAG 1.4.11 bar for non-text marks) applied on exact relative luminance, as an additional feasibility constraint in `_feasible.py` (in `project`, it would clip or move points along L). The concern is how it interacts with `lightness` and `chroma`: both would constrain the same region, so an explicit lightness could make the contrast threshold unreachable (or silently redundant), and the derived lightness band would need to be derived jointly with it. Before adding it, decide the precedence rules (e.g. explicit values intersect and an empty result raises a clear error; `0` disables the filter) and whether the derived L band should be computed from the contrast target. Sensibly designed together with the "user-tunable constraints" item above.

### Other options considered (not planned)
- **K-means / Lloyd (CVT) on the gamut:** produces *evenly covering* colors rather than maximally separated ones; centroids sit in the interior and have lower chroma. Useful as a contrast, not the goal.
- **Simulated annealing / genetic search:** flexible objective (e.g. include CVD terms) but slower and non-deterministic without care.
- **Fixed curated palettes (Glasbey, Brewer, Tableau, Okabe-Ito):** good for small n but not parametric; used as benchmarks.

## Package layout (v1)

```
src/colorcat/
  __init__.py          # exports palette, min_distance
  palette.py           # public palette() + argument validation + ordering
  _constraints.py      # Constraints dataclass + the two built-in sets (white / black)
  _oklab.py            # sRGB <-> linear <-> OKLab <-> OKLCh, gamut test
  _distance.py         # single pairwise-distance entry point + gradient (ΔE_OK now; CVD views in v2)
  _feasible.py         # feasible region: gamut_max_chroma, project, sample (+ lru_cache)
  _energy.py           # log-domain Riesz energy, projected descent, hyperparameter constants, restarts
  _format.py           # output formatting, 8-bit quantization (hex/rgb/...)
  py.typed
tests/
  test_oklab.py        # round-trip, known reference values
  test_constraints.py  # built-in ranges, background validation
  test_feasible.py     # projection idempotence/feasibility, gamut boundary, sampling
  test_energy.py       # finite-difference gradient check, energy decreases, restarts
  test_palette.py      # shape, seed reproducibility, constraints respected, restart_count
  test_quality.py      # min distance beats random / farthest-point baseline / known palettes
README.md              # usage, rationale, gallery image
```

v2 adds `_farthest.py` and `_maximin.py`.

Housekeeping: fix `pyproject.toml` description; remove the placeholder `colorcat = "colorcat:main"` script (or implement a small CLI: `colorcat 8 --background black`); add `numpy`, and `pytest` (+ optionally `ruff`, `mypy`) as dev dependencies. Set `requires-python = ">=3.12"` (the template currently says `>=3.14`) and change `.python-version` to `3.12` to match.

## Testing & validation

- **Correctness:** OKLab round-trips to within 1e-6; known values (white → L=1, black → L=0, pure red ≈ L 0.628 / C 0.258 / h 29.2°).
- **Objective:** analytic gradient matches finite differences; energy is non-increasing across accepted steps; as `p` grows, `E_p` tracks the true minimum distance.
- **Feasible region:** `project` is idempotent and always returns feasible points, including near the band edges; `sample` points are all feasible.
- **Argument validation:** `background` other than `"white"`/`"black"` raises; `restart_count < 1` or non-integer raises; bad hex in `fixed`/`exclude` raises; `n=0` → `[]`.
- **Built-in ranges:** white → outputs in the lower, darker L band; black → upper, lighter band; every output color lies inside the documented L/C ranges (within a quantization tolerance) and inside sRGB. Visual review: render swatch strips on white and black while tuning the ranges.
- **Anchors:** `fixed` colors are never re-selected or moved; chosen colors keep distance from `exclude` colors and the background at least comparable to their distance from each other.
- **Properties:** same args + same seed → identical output (across repeated calls and fresh processes); `restart_count=k+m` is never worse than `restart_count=k` for the same seed (nested child streams); different seeds can differ but stay within a small tolerance of each other's min distance; `len == n`; no duplicates after quantization; n=0/1/2 edge cases. Explicitly *not* tested: any relationship between `palette(k)` and `palette(n)`.
- **Quality:** for n in {2…30}, on white and black backgrounds, min pairwise ΔE_OK (after quantization) of `palette` ≥ random sampling from the same feasible region (median of many draws), ≥ a greedy farthest-point baseline implemented in the benchmark script, and comparable to or better than named palettes at the same n. Track results, runtime, and variance across seeds in a benchmark script; use it to confirm the `restart_count=32` default and the hyperparameter constants.
- **Quantization:** colors are measured *after* rounding to 8-bit sRGB, since that is what users see.

## Milestones

1. **Foundations:** OKLab/OKLCh conversions, gamut test, `_distance.py`, tests.
2. **Constraints:** `_constraints.py` with the two built-in sets and background validation; tune the numbers visually.
3. **Feasible region:** `gamut_max_chroma`, `project`, `sample`, with property tests.
4. **Energy optimizer:** objective + gradient, projected descent, hyperparameter constants, restarts, seeding; `palette()` end to end with hex output; `min_distance`; benchmark vs random, farthest-point baseline and named palettes; confirm `restart_count` default and tune constants.
5. **Extras:** `fixed`, `exclude`, output formats, ordering, CLI.
6. **Docs & release (v1):** README with swatch gallery, API docs, versioning.
7. **v2:** user-tunable `lightness`/`chroma`/`hue` and arbitrary backgrounds (mid-tone error); `method=` with farthest-point sampling and maximin refinement; `cvd=`; WCAG `min_background_contrast`; optional CIEDE2000/CAM16 evaluation metric; optional output ordering modes (e.g. sort by hue).

## Open questions

- Exact built-in L/C bands per background (starting values above; tune visually on white and black).
- Whether the benchmark supports keeping `restart_count=16` as the default, or a different value fits typical `n` and the runtime budget better.

## Decisions (resolved)

- `seed` defaults to `0`; it chooses restart starts.
- No chroma floor (an earlier 0.09 floor was removed); greys are not excluded, though the optimizer rarely picks them.
- Minimum Python version is 3.12.
- **`background` is `"white"` or `"black"` only in v1** and selects hard-coded lightness/chroma/hue ranges; no range overrides in v1.
- **`restart_count` is the one exposed tuning argument** (default 4). All other optimizer hyperparameters are hard-coded constants.
- No pinned first color; the color farthest from the background only determines output ordering.
- Mid-tone backgrounds (v2, with arbitrary backgrounds) raise an error unless `lightness` is given explicitly.
- WCAG contrast (`min_background_contrast`) is deferred to v2.
- **v1 algorithm is energy minimization** (NumPy-only projected gradient descent with restarts), chosen because per-`n` optimality matters more than speed or prefix stability. Farthest-point sampling and maximin refinement are kept as v2 options.
- Each `n` is optimized independently; no prefix stability.
- NumPy is the only runtime dependency; CVD awareness is a v2 option.
