"""Command-line interface: ``colorcat 8 --background black``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ._cvd import CVD_TYPES
from ._format import FORMATS
from .palette import palette


def _line(color: object) -> str:
    if isinstance(color, str):
        return color
    return " ".join(str(v) if isinstance(v, int) else f"{v:.4f}" for v in color)  # type: ignore[attr-defined]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="colorcat", description="Print N maximally distinct colors, one per line."
    )
    parser.add_argument("n", type=int, help="number of colors")
    parser.add_argument("--background", choices=["white", "black"], default="white")
    parser.add_argument("--restarts", type=int, default=4, metavar="N", help="restart_count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude", nargs="+", default=[], metavar="HEX", help="colors to avoid")
    parser.add_argument("--fixed", nargs="+", default=[], metavar="HEX", help="colors to extend")
    parser.add_argument(
        "--cvd", nargs="+", default=[], choices=CVD_TYPES, help="deficiencies to design for"
    )
    parser.add_argument("--format", choices=FORMATS, default="hex")
    args = parser.parse_args(argv)
    try:
        colors = palette(
            args.n,
            background=args.background,
            restart_count=args.restarts,
            seed=args.seed,
            exclude=args.exclude,
            fixed=args.fixed,
            cvd=args.cvd,
            color_format=args.format,
        )
    except (ValueError, TypeError) as err:
        parser.error(str(err))
    for color in colors:
        print(_line(color))
    return 0
