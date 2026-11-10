"""Energy-minimization palette optimizer.

Colors are free particles in OKLab, repelled from each other and from fixed
anchors (the background, later ``fixed``/``exclude`` colors) and confined to the
feasible region. The objective is the log-domain Riesz energy

    E_p = (1/p) * log( sum over pairs of d_ij ** -p )

which tends to ``-log(min d_ij)`` as ``p`` grows, so minimizing it spreads the
colors by maximizing the smallest pairwise distance. Anchor-anchor pairs are
constant and skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._constraints import Constraints
from ._distance import pairwise_distance, pairwise_distance_and_gradient, views
from ._feasible import project, sample
from ._format import quantize
from ._oklab import Array, srgb_to_oklab

# Hard-coded hyperparameters (not user-facing; tuned by the benchmark).
P_SCHEDULE = (4.0, 8.0, 16.0, 32.0)
MAX_ITERS_PER_STAGE = 150
STOP_TOL = 1e-4  # minimum-distance improvement that counts as progress
PATIENCE = 10  # iterations without progress before a stage stops
INITIAL_STEP = 0.02  # largest per-iteration move of any color (OKLab units)
BACKTRACK_FACTOR = 0.5
MIN_STEP = 1e-6
HOP_COUNT = 16  # basin-hopping moves after each restart's first descent
HOP_P_SCHEDULE = (16.0, 32.0)  # stages re-run after a hop
HOP_CANDIDATES = 256  # feasible points tried as the new home of a re-seeded color
HOP_CLOSE = 1.05  # pairs within this factor of the closest pair are all "closest"
_MIN_DISTANCE = 1e-12  # floor that keeps log(d) finite for coincident colors


def energy_and_grad(
    x: ArrayLike, anchors: ArrayLike, p: float, cvd: tuple[str, ...] = ()
) -> tuple[float, Array]:
    """Riesz energy ``E_p`` of particles ``x`` (n, 3) with fixed ``anchors`` (m, 3).

    With ``cvd``, every pair contributes one term per view (normal vision and each
    deficiency), so a pair is repelled hardest in the view where it is closest.
    Returns ``(E, dE/dx)`` with the gradient of shape ``(n, 3)``.
    """
    x = np.asarray(x, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 3)
    n = len(x)
    off_diag = ~np.eye(n, dtype=bool)

    # Log of each term d**-p. Each particle pair appears twice in the full
    # matrix, so those terms carry a factor 1/2.
    terms = []
    for view in views(cvd):
        d_pp, g_pp = pairwise_distance_and_gradient(x, x, view)
        d_pa, g_pa = pairwise_distance_and_gradient(x, anchors, view)
        log_pp = np.where(
            off_diag, np.log(0.5) - p * np.log(np.maximum(d_pp, _MIN_DISTANCE)), -np.inf
        )
        log_pa = -p * np.log(np.maximum(d_pa, _MIN_DISTANCE))
        terms.append((d_pp, g_pp, log_pp, d_pa, g_pa, log_pa))

    all_terms = np.concatenate([np.concatenate([t[2].ravel(), t[5].ravel()]) for t in terms])
    top = all_terms.max()
    if not np.isfinite(top):  # nothing repels anything
        return 0.0, np.zeros_like(x)
    log_z = top + np.log(np.sum(np.exp(all_terms - top)))

    grad = np.zeros_like(x)
    for d_pp, g_pp, log_pp, d_pa, g_pa, log_pa in terms:
        w_pp = np.exp(log_pp - log_z)
        w_pa = np.exp(log_pa - log_z)
        inv_pp = 1.0 / np.maximum(d_pp, _MIN_DISTANCE)
        inv_pa = 1.0 / np.maximum(d_pa, _MIN_DISTANCE)
        # dE/dx_i = -sum_j 2 w_ij / d_ij * u_ij  -  sum_a w_ia / d_ia * u_ia
        grad -= 2.0 * np.einsum("ij,ijk->ik", w_pp * inv_pp, g_pp)
        grad -= np.einsum("ij,ijk->ik", w_pa * inv_pa, g_pa)
    return log_z / p, grad


def min_distance_with_anchors(
    x: ArrayLike, anchors: ArrayLike, cvd: tuple[str, ...] = ()
) -> float:
    """Smallest distance between any two particles or between a particle and an anchor."""
    x = np.asarray(x, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 3)
    best = float(pairwise_distance(x, anchors, cvd).min()) if len(anchors) else np.inf
    if len(x) > 1:
        d = pairwise_distance(x, cvd=cvd)
        best = min(best, float(d[np.triu_indices(len(x), k=1)].min()))
    return best


def _descend(
    x: Array,
    anchors: Array,
    p: float,
    constraints: Constraints,
    cvd: tuple[str, ...] = (),
) -> Array:
    """One stage of projected gradient descent with backtracking at a fixed ``p``."""
    energy, grad = energy_and_grad(x, anchors, p, cvd)
    step = INITIAL_STEP
    best_min = min_distance_with_anchors(x, anchors, cvd)
    stale = 0
    for _ in range(MAX_ITERS_PER_STAGE):
        scale = np.linalg.norm(grad, axis=1).max()
        if scale < 1e-12:
            break
        direction = -grad / scale
        while True:
            candidate = project(x + step * direction, constraints)
            new_energy, new_grad = energy_and_grad(candidate, anchors, p, cvd)
            if new_energy < energy:
                break
            step *= BACKTRACK_FACTOR
            if step < MIN_STEP:
                return x
        x, energy, grad = candidate, new_energy, new_grad
        step = min(2.0 * step, INITIAL_STEP)
        current_min = min_distance_with_anchors(x, anchors, cvd)
        if current_min - best_min > STOP_TOL:
            best_min, stale = current_min, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return x


@dataclass(frozen=True)
class Result:
    rgb8: np.ndarray  # (n, 3) quantized colors
    lab: Array  # (n, 3) their OKLab values
    min_distance: float  # after quantization, including anchors
    energy: float  # final-stage energy of the quantized colors


def _evaluate(
    x: Array, anchors: Array, constraints: Constraints, cvd: tuple[str, ...]
) -> Result:
    """Quantize ``x`` and score it."""
    rgb8 = quantize(x, constraints)
    lab = srgb_to_oklab(rgb8 / 255.0)
    return Result(
        rgb8=rgb8,
        lab=lab,
        min_distance=min_distance_with_anchors(lab, anchors, cvd),
        energy=energy_and_grad(lab, anchors, P_SCHEDULE[-1], cvd)[0],
    )


def _better(a: Result, b: Result) -> bool:
    return (a.min_distance, -a.energy) > (b.min_distance, -b.energy)


def _hop(
    rng: np.random.Generator,
    x: Array,
    anchors: Array,
    constraints: Constraints,
    cvd: tuple[str, ...],
) -> Array:
    """Re-seed one of the colors in a closest pair at the roomiest spot, then re-descend.

    The color to move is drawn at random among those in (nearly) the closest pairs.
    Its new position is the best of ``HOP_CANDIDATES`` random feasible points, i.e.
    the one farthest from everything else.
    """
    n = len(x)
    d_pp = pairwise_distance(x, cvd=cvd)
    d_pp[np.arange(n), np.arange(n)] = np.inf
    d_pa = pairwise_distance(x, anchors, cvd)
    closest = min(d_pp.min(), d_pa.min())
    limit = closest * HOP_CLOSE
    movers = np.flatnonzero((d_pp.min(axis=1) <= limit) | (d_pa.min(axis=1) <= limit))
    mover = int(rng.choice(movers))
    others = np.concatenate([np.delete(x, mover, axis=0), anchors])
    candidates = sample(rng, HOP_CANDIDATES, constraints)
    roomiest = candidates[np.argmax(pairwise_distance(candidates, others, cvd).min(axis=1))]
    y = x.copy()
    y[mover] = roomiest
    for p in HOP_P_SCHEDULE:
        y = _descend(y, anchors, p, constraints, cvd)
    return y


def optimize_once(
    rng: np.random.Generator,
    n: int,
    constraints: Constraints,
    anchors: ArrayLike,
    cvd: tuple[str, ...] = (),
) -> Result:
    """One restart: random feasible start, descent through the ``p`` schedule, then
    ``HOP_COUNT`` basin-hopping moves (each kept only if it improves), quantized."""
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 3)
    x = sample(rng, n, constraints)
    for p in P_SCHEDULE:
        x = _descend(x, anchors, p, constraints, cvd)
    best = _evaluate(x, anchors, constraints, cvd)
    if n > 1 or len(anchors) > 0:
        for _ in range(HOP_COUNT):
            y = _hop(rng, x, anchors, constraints, cvd)
            r = _evaluate(y, anchors, constraints, cvd)
            if _better(r, best):
                x, best = y, r
    return best


def optimize(
    seed: int | None,
    n: int,
    constraints: Constraints,
    anchors: ArrayLike,
    restart_count: int,
    cvd: tuple[str, ...] = (),
) -> Result:
    """Best of ``restart_count`` restarts, each with its own child random stream.

    Restart ``i`` uses ``SeedSequence(seed).spawn(restart_count)[i]``, so its result
    does not depend on ``restart_count`` and more restarts can never be worse.
    Ties go to lower energy, then to the lower restart index.
    """
    streams = np.random.SeedSequence(seed).spawn(restart_count)
    best: Result | None = None
    for stream in streams:
        r = optimize_once(np.random.default_rng(stream), n, constraints, anchors, cvd)
        if best is None or _better(r, best):
            best = r
    assert best is not None
    return best
