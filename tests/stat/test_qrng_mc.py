"""QRNG z pamięcią detektorów: obciążenie na tle wzoru dokładnego i wyjście ekstraktora (F7)."""

import math

import numpy as np
import pytest

from conftest import SEED
from qkdsim.defenses.randomness_tests import TESTS
from qkdsim.qrng import BeamSplitterQrng, QrngSource, min_entropy


@pytest.mark.parametrize(
    ("efficiency", "dead", "afterpulse"),
    [((0.1, 0.08), 0, 0.0), ((0.1, 0.08), 20, 0.0), ((0.1, 0.06), 5, 0.1)],
)
def test_bias_and_rate_match_exact_formula(
    efficiency: tuple[float, float], dead: int, afterpulse: float
) -> None:
    device = BeamSplitterQrng(
        1.0, efficiency=efficiency, dead_time_gates=dead, afterpulse=afterpulse
    )
    slots = 20_000_000
    bits = device.raw_bits(slots, np.random.default_rng(SEED))
    means = bits[: len(bits) // 50 * 50].reshape(50, -1).mean(axis=1)
    sigma = float(np.std(means, ddof=1)) / math.sqrt(50)
    assert abs(bits.mean() - device.bias()) < 4 * sigma
    assert len(bits) / slots == pytest.approx(device.bit_rate(), rel=3e-3)


def test_extracted_output_of_correlated_qrng_passes_tests() -> None:
    device = BeamSplitterQrng(1.0, efficiency=(0.1, 0.09), dead_time_gates=10, afterpulse=0.03)
    calibration = device.raw_bits(10_000_000, np.random.default_rng(SEED))
    entropy = min_entropy(calibration, context=4)
    assert 0.4 < entropy < 0.6  # czas martwy: bit zależy od poprzedniego
    source = QrngSource(
        device, np.random.default_rng(SEED + 1), np.random.default_rng(SEED + 2), entropy
    )
    bits = source.bits(500_000)
    assert all(test(bits) > 1e-3 for test in TESTS.values())
    assert min_entropy(bits, context=4) > 0.95
