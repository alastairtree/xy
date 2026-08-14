"""Linear-axis GPU mapping precision contracts (§4/§16)."""

from __future__ import annotations

import struct
from math import isclose
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


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


def test_extreme_domain_scale_remains_finite_without_clamping() -> None:
    span = 2e300
    scale = 1e-263

    assert isclose(2 / (span * scale), 1e-37)
