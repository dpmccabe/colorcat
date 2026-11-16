# colorcat

Generate **maximally distinct colors** for categorical data palettes (chart series,
legend entries, cluster labels).

```python
>>> from colorcat import palette
>>> palette(5)
['#0000d4', '#ff4b00', '#01a600', '#6a0000', '#d500ff']
>>> palette(8, background="black")
```

![Palettes on white and black backgrounds](https://raw.githubusercontent.com/dpmccabe/colorcat/main/docs/gallery.svg)

- Colors are chosen to maximize the smallest perceptual distance between any two of
  them **and** between each of them and the background **and its opposite** (black on white,
  white on black), so nothing blends into plot lines or text.
- Deterministic: the same arguments and `seed` always give the same palette.
- NumPy is the only dependency. Python 3.12+.

## Install

```sh
uv add colorcat        # once published; from a checkout: uv sync
```

## Usage

```python
from colorcat import palette, min_distance

palette(6)  # on a white background (the default)
palette(6, background="black")  # lighter colors for dark backgrounds
palette(6, min_contrast=4.5)  # WCAG AA contrast with the background
palette(4, fixed=["#e41a1c", "#377eb8"])  # 4 NEW colors that extend two you already use
palette(4, exclude=["#00aa00"])  # keep clear of a color used elsewhere
palette(6, cvd=["deutan", "protan"])  # also distinct for red-green color blindness
palette(6, color_format="rgb")  # (r, g, b) tuples; also "oklab" and "oklch"
palette(6, seed=3)  # a different, similarly good palette

min_distance(palette(6))  # distance between the closest pair
```

### `palette(n, *, background, min_contrast, restart_count, exclude, fixed, cvd, color_format, seed)`

| Argument | Default | Meaning |
|---|---|---|
| `n` | | Number of colors, `>= 0`. |
| `background` | `"white"` | `"white"` or `"black"`, the color the palette is drawn on. Other values raise `ValueError`. |
| `min_contrast` | `3` | Minimum [WCAG contrast ratio](https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio) (1 to 15) between every color and the background. 3 is the WCAG minimum for graphical objects such as chart marks; use 4.5 for text. Higher values leave fewer colors to choose from, so the closest-pair distance drops. |
| `restart_count` | `4` | Independent optimizer runs (each followed by 16 basin-hopping moves); the best is kept. More is never worse, just slower (roughly 1-3.5 s for 4 at `n` up to 30). Ignored when a saved palette is returned (see below). |
| `exclude` | `()` | Hex colors the new colors must stay distinct from. Not returned. |
| `fixed` | `()` | Hex colors already in use; the `n` new colors extend them. Never moved or returned, and exempt from the built-in ranges. |
| `cvd` | `()` | One or more of `"protan"`, `"deutan"`, `"tritan"` (or a single name). Every pair must then be distinct to normal vision *and* as seen with each deficiency (full dichromacy, Machado et al. 2009 simulation). The closest-pair distance drops as views are added, more so for larger `n`. |
| `color_format` | `"hex"` | `"hex"` (`"#rrggbb"`), `"rgb"` (0-255 ints), `"oklab"` (L, a, b) or `"oklch"` (L, C, h in degrees). |
| `seed` | `0` | Seed for the random starting points; `None` uses fresh entropy. |

**Instant results for common cases.** With the default `min_contrast` and `seed` and no `exclude`,
`fixed` or `cvd`, `palette(n)` for `n` up to 15 returns a saved palette (optimized with 20
restarts) instead of running the optimizer, so it is instant and `restart_count` has no effect.
Any other argument, or a larger `n`, runs the optimizer. `benchmarks/presets.py` regenerates the
saved palettes.

Colors come back ordered so that the first is the most distant from the background and its
opposite, and each next one is the farthest from those before it. Each `n` is optimized separately, so
`palette(4)` is **not** the first four colors of `palette(5)`.

`palette` warns (it does not fail) if the closest pair differs by less than ΔE_OK 0.03,
which happens only for very large `n`.

### `min_distance(colors, *, cvd=())`

ΔE_OK between the two closest colors of a list of hex strings (`inf` for fewer than two).
With `cvd`, the closest pair over normal vision and each listed deficiency.

### Command line

```sh
colorcat 8 --background black
colorcat 4 --fixed '#e41a1c' '#377eb8' --format rgb
python -m colorcat 6 --seed 3
```

## How it works

- **Color space:** [OKLab](https://bottosson.github.io/posts/oklab/), where Euclidean
  distance (ΔE_OK) approximates perceived difference. The conversions are implemented
  directly in NumPy.
- **Constraints:** every color must have at least `min_contrast` WCAG contrast with the
  background, `(Y_light + 0.05) / (Y_dark + 0.05)` on relative luminance `Y`. This is a
  one-sided bound on `Y` (on white, `Y <= 0.30` at the default 3:1). Chroma and hue have
  no bounds of their own; the sRGB gamut is the only other limit. The opposite of the
  background (black on white, white on black: where lines and text are drawn) is not a
  constraint but a fixed point the colors are repelled from, like the background itself.
- **Optimization:** colors are particles repelled from each other and from fixed anchors
  (the background, its opposite, `fixed` and `exclude`) by the Riesz energy
  `E_p = (1/p) log Σ d_ij^-p`, which tends to the max-min distance as `p` grows.
  Projected gradient descent with a `p` schedule of 4, 8, 16, 32 minimizes it,
  restricted to the feasible region. The landscape has local optima, so it is restarted
  from `restart_count` random starts (each with its own child of `SeedSequence(seed)`).
  After its first descent, each restart does 16 basin-hopping moves: re-seed one of the
  colors in the closest pair at the roomiest spot, re-descend (`p` = 16, 32), and keep the
  result only if the closest-pair distance improved.
- **Quantization:** the best restart is chosen after snapping to 8-bit sRGB, so the
  reported distances are those of the colors you actually get.

### Quality

`benchmarks/benchmark.py` compares `palette` with random feasible colors, greedy
farthest-point sampling and two named palettes at the `n` that matches their size: Okabe-Ito at `n = 7` (its 8
colors minus black) and seaborn's tab10 at `n = 9` (minus its grey `#7f7f7f`).
On white, the smallest pairwise ΔE_OK (including the background and its opposite) is:

| n | colorcat | farthest-point | random | Okabe-Ito | tab10 |
|---|---|---|---|---|---|
| 5 | 0.34 | 0.30 | 0.08 | | |
| 7 | 0.31 | 0.25 | 0.06 | 0.16 | |
| 9 | 0.26 | 0.20 | 0.05 | | 0.14 |
| 20 | 0.18 | 0.14 | 0.03 | | |
| 30 | 0.15 | 0.12 | 0.02 | | |

Results vary by about 5% across seeds at small `n`.

## Not in v1

Planned for later: user-set lightness, chroma and hue ranges; arbitrary backgrounds
(the "opposite" color repelled alongside the background would need defining);
partial-severity CVD (`cvd=` simulates full dichromacy); and alternative methods (farthest-point,
which gives prefix-stable palettes).

## Development

```sh
uv sync
uv run pytest                              # ~30 s
uv run python benchmarks/benchmark.py      # quality vs. baselines
uv run python benchmarks/make_gallery.py   # regenerate docs/gallery.svg
```

Layout: `_oklab.py` (conversions), `_distance.py` (the single distance definition),
`_constraints.py` (built-in ranges), `_feasible.py` (gamut and projection),
`_energy.py` (optimizer), `_format.py` (parsing, quantization, formats),
`palette.py` (public API), `cli.py`. See [`plan.md`](https://github.com/dpmccabe/colorcat/blob/main/plan.md) for the design.
