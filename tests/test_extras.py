import re

import numpy as np
import pytest

from colorcat import min_distance, palette
from colorcat._format import hex_to_oklab, hex_to_srgb
from colorcat._oklab import oklab_to_oklch
from colorcat.cli import main

FAST = dict(restart_count=2)


def lab(colors):
    return np.array([hex_to_oklab(c) for c in colors])


def test_formats():
    hexes = palette(4, **FAST)
    rgb = palette(4, color_format="rgb", **FAST)
    assert rgb == [tuple(int(round(v * 255)) for v in hex_to_srgb(h)) for h in hexes]
    assert all(isinstance(v, int) for c in rgb for v in c)
    np.testing.assert_allclose(palette(4, color_format="oklab", **FAST), lab(hexes), atol=1e-12)
    np.testing.assert_allclose(
        palette(4, color_format="oklch", **FAST), oklab_to_oklch(lab(hexes)), atol=1e-9
    )
    assert palette(0, color_format="rgb") == []


def test_bad_format():
    with pytest.raises(ValueError, match="'hex', 'rgb', 'oklab', 'oklch'"):
        palette(3, color_format="cmyk")  # pyrefly: ignore[bad-argument-type]


def test_fixed_colors_are_kept_apart_and_not_returned():
    fixed = ["#e41a1c", "#377eb8"]
    new = palette(5, fixed=fixed, **FAST)
    assert len(new) == 5 and not set(new) & set(fixed)
    # new colors stay well apart from the fixed ones too
    assert min_distance(fixed + new) > 0.1


def test_exclude_pushes_colors_away():
    plain = palette(1, **FAST)
    ex = palette(1, exclude=plain, **FAST)
    assert ex != plain
    d_to = np.linalg.norm(lab(ex) - lab(plain))
    assert d_to > 0.15


def test_fixed_and_exclude_are_anchors_only():
    # A fixed color that violates the built-in ranges (grey) is accepted.
    out = palette(3, fixed=["#808080"], **FAST)
    assert len(out) == 3
    assert min_distance(["#808080"] + out) > 0.1


def test_extras_are_deterministic():
    kw = dict(exclude=["#00ff00"], fixed=["#0000ff"], **FAST)
    assert palette(4, **kw) == palette(4, **kw)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        (dict(fixed=["nothex"]), ValueError),
        (dict(exclude=["#12"]), ValueError),
        (dict(fixed="#ff0000"), TypeError),
        (dict(exclude=5), TypeError),
        (dict(fixed=[1]), ValueError),
    ],
)
def test_color_argument_validation(kwargs, exc):
    with pytest.raises(exc):
        palette(3, **kwargs)


def test_short_hex_and_bare_hex_accepted():
    assert len(palette(2, fixed=["#f00", "00ff00"], **FAST)) == 2


def test_cli_prints_one_color_per_line(capsys):
    assert main(["5", "--restarts", "2"]) == 0
    lines = capsys.readouterr().out.split()
    assert len(lines) == 5 and all(re.fullmatch(r"#[0-9a-f]{6}", x) for x in lines)
    assert lines == palette(5, restart_count=2)


def test_cli_options(capsys):
    main(["3", "--background", "black", "--restarts", "2", "--format", "rgb", "--fixed", "#ff0000"])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 3 and all(len(line.split()) == 3 for line in out)


def test_cli_errors_exit_nonzero(capsys):
    with pytest.raises(SystemExit) as e:
        main(["3", "--fixed", "nothex"])
    assert e.value.code == 2
    assert "not a hex color" in capsys.readouterr().err
