"""Linear-axis GPU mapping precision contracts (§4/§16)."""

from __future__ import annotations

import struct
from math import isclose, isfinite
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _linear_map(meta: dict[str, Any] | None, lo: float, hi: float) -> tuple[float, float]:
    span = hi - lo
    if not isfinite(span) or span == 0:
        return (0, -2)
    scale = float(meta["scale"]) if meta else 1
    if not isfinite(scale) or scale <= 0:
        return (0, -2)
    offset = float(meta["offset"]) if meta and isfinite(meta["offset"]) else 0
    mul = 2 / (span * scale)
    center = ((lo - offset) + span * 0.5) * scale
    return (mul, center) if isfinite(mul) and isfinite(center) else (0, -2)


def test_shader_centers_before_multiplying() -> None:
    shader = (ROOT / "js/src/40_gl.ts").read_text(encoding="utf-8")
    chartview = (ROOT / "js/src/50_chartview.ts").read_text(encoding="utf-8")

    assert "return (encoded - map.y) * map.x;" in shader
    assert "if (!Number.isFinite(scale) || scale <= 0) return [0, -2];" in chartview
    assert "const center = ((lo - offset) + span * 0.5) * scale;" in chartview
    assert "return [mul, center];" in chartview


def test_centered_affine_avoids_large_term_cancellation() -> None:
    encoded_center = 2_000_000.0
    span = 7.0
    encoded = _f32(encoded_center - span / 4)
    mul = _f32(2 / span)

    large_term_result = _f32(_f32(encoded * mul) + _f32(-encoded_center * 2 / span))
    centered_result = _f32(_f32(encoded - _f32(encoded_center)) * mul)

    assert large_term_result == -0.4375
    assert centered_result == -0.5


def test_linear_map_keeps_valid_extreme_scales_and_culls_malformed_ones() -> None:
    mul, center = _linear_map({"offset": 0, "scale": 1e-263}, -1e300, 1e300)

    assert isclose(mul, 1e-37)
    assert center == 0
    assert _linear_map({"offset": 0, "scale": 0}, -1, 1) == (0, -2)
    assert _linear_map({"offset": 0, "scale": float("nan")}, -1, 1) == (0, -2)
