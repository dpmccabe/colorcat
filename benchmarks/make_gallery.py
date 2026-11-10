"""Render docs/gallery.svg: palettes for several n on white and black backgrounds.

Usage: python benchmarks/make_gallery.py   (plain SVG, no plotting dependency)
"""

from pathlib import Path

from colorcat import min_distance, palette
from colorcat._constraints import Background

SIZES = (2, 3, 5, 8, 13)
SW, GAP, PAD, ROW, LABEL = 28, 4, 16, 44, 92


def panel(background: Background, x0: int, width: int, fg: str, bg: str) -> str:
    out = [
        f'<rect x="{x0}" y="0" width="{width}" height="{PAD * 2 + ROW * len(SIZES)}" fill="{bg}"/>'
    ]
    for row, n in enumerate(SIZES):
        colors = palette(n, background=background)
        y = PAD + row * ROW
        out.append(
            f'<text x="{x0 + PAD}" y="{y + 20}" fill="{fg}" font-size="13" '
            f'font-family="system-ui, sans-serif">n = {n}</text>'
            f'<text x="{x0 + PAD}" y="{y + 36}" fill="{fg}" opacity="0.6" font-size="10" '
            f'font-family="system-ui, sans-serif">min ΔE {min_distance(colors):.2f}</text>'
        )
        for i, c in enumerate(colors):
            out.append(
                f'<rect x="{x0 + LABEL + i * (SW + GAP)}" y="{y + 4}" width="{SW}" height="{SW + 8}" rx="4" fill="{c}"/>'
            )
    return "\n".join(out)


def main() -> None:
    width = LABEL + max(SIZES) * (SW + GAP) + PAD
    height = PAD * 2 + ROW * len(SIZES)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width * 2} {height}" '
        f'width="{width * 2}" height="{height}">\n'
        + panel("white", 0, width, "#222", "#ffffff")
        + "\n"
        + panel("black", width, width, "#eee", "#000000")
        + "\n</svg>\n"
    )
    out = Path(__file__).parent.parent / "docs" / "gallery.svg"
    out.write_text(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
