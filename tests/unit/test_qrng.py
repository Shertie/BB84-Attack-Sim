"""Symulator QRNG i testy zdrowia NIST SP 800-90B (F7)."""

import math

import numpy as np
import pytest

from conftest import SEED
from qkdsim.defenses.health_tests import (
    adaptive_alarms,
    adaptive_cutoff,
    adaptive_detection_probability,
    first_repetition_alarm,
    repetition_cutoff,
)
from qkdsim.qrng import BeamSplitterQrng, QrngSource, extract, min_entropy, output_length


@pytest.mark.parametrize(
    "kwargs",
    [{"mean_photons": 0.0}, {"mean_photons": 1.0, "transmission": 1.1},
     {"mean_photons": 1.0, "efficiency": (0.1, 1.5)}],
)  # fmt: skip
def test_qrng_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BeamSplitterQrng(**kwargs)  # type: ignore[arg-type]


def test_symmetric_qrng_is_unbiased_and_rate_is_exact() -> None:
    device = BeamSplitterQrng(1.0, dark_count=0.0)
    f = 1.0 - math.exp(-0.05)
    assert device.bias() == pytest.approx(0.5, abs=1e-15)
    assert device.bit_rate() == pytest.approx(2 * f * (1 - f), rel=1e-12)


def test_blinded_detector_gives_constant_bits() -> None:
    device = BeamSplitterQrng(1.0, efficiency=(0.1, 0.0), dark_count=0.0)
    bits = device.raw_bits(100_000, np.random.default_rng(SEED))
    assert len(bits) > 0 and not bits.any()
    assert device.bias() == 0.0


def test_min_entropy_extremes() -> None:
    assert min_entropy(np.zeros(1000, dtype=np.uint8)) == 0.0
    alternating = np.tile(np.array([0, 1], dtype=np.uint8), 5000)
    assert min_entropy(alternating, context=0) > 0.95
    assert min_entropy(alternating, context=1) == 0.0


def test_min_entropy_follows_sp800_90b_example_bound() -> None:
    """p̂ = 0,4 z L = 20 daje p_u = 0,4 + 2,576·√(0,24/19) (NIST SP 800-90B, 6.3.1)."""
    bits = np.array([1] * 12 + [0] * 8, dtype=np.uint8)  # p̂ = 0,6 dla wartości 1
    p_u = 0.6 + 2.576 * math.sqrt(0.6 * 0.4 / 19)
    assert min_entropy(bits, confidence=0.995) == pytest.approx(-math.log2(p_u), rel=1e-3)


def test_output_length_follows_leftover_hash_lemma() -> None:
    assert output_length(65536, 0.5, 1e-10) == math.floor(32768 - 2 * math.log2(1e10))
    assert output_length(100, 0.1, 1e-10) == 0


def test_extract_uses_whole_blocks() -> None:
    rng = np.random.default_rng(SEED)
    raw = rng.integers(0, 2, 1000, dtype=np.uint8)
    seed = rng.integers(0, 2, 256 + 100 - 1, dtype=np.uint8)
    assert len(extract(raw, seed, 256, 100)) == 3 * 100


def test_qrng_source_is_deterministic_and_balanced() -> None:
    device = BeamSplitterQrng(1.0, efficiency=(0.1, 0.07), dead_time_gates=5, afterpulse=0.02)

    def make() -> QrngSource:
        return QrngSource(device, np.random.default_rng(SEED), np.random.default_rng(SEED + 1), 0.4)

    a, b = make(), make()
    bits = a.bits(200_000)
    assert np.array_equal(bits, b.bits(200_000))
    assert abs(bits.mean() - 0.5) < 4 * 0.5 / math.sqrt(len(bits))
    assert a.raw_used % 65536 == 0 and a.raw_used >= len(bits) / (a.out_per_block / 65536)


def test_qrng_source_choice_has_requested_distribution() -> None:
    device = BeamSplitterQrng(1.0)
    source = QrngSource(device, np.random.default_rng(SEED), np.random.default_rng(SEED + 1), 0.9)
    picks = source.choice(100_000, (0.7, 0.2, 0.1))
    for value, p in enumerate((0.7, 0.2, 0.1)):
        assert abs(np.mean(picks == value) - p) < 4 * math.sqrt(p * (1 - p) / 100_000)


def test_raw_qrng_source_passes_raw_bits_through() -> None:
    device = BeamSplitterQrng(1.0, efficiency=(0.1, 0.05))
    source = QrngSource(device, np.random.default_rng(SEED), np.random.default_rng(SEED + 1), 1.0)
    bits = source.bits(500_000)
    assert abs(bits.mean() - device.bias()) < 4 * math.sqrt(0.25 / len(bits))


# --- testy zdrowia ----------------------------------------------------------------------


def test_cutoffs_match_sp800_90b_examples() -> None:
    """Tabela 2 (APT, α = 2⁻²⁰) i przykład RCT (H = 2 → C = 11) z NIST SP 800-90B."""
    assert [adaptive_cutoff(h) for h in (0.2, 0.4, 0.6, 0.8, 1.0)] == [941, 840, 748, 664, 589]
    assert [adaptive_cutoff(h, 512) for h in (0.5, 1, 2, 4, 8)] == [410, 311, 177, 62, 13]
    assert repetition_cutoff(2.0) == 11


def test_repetition_alarm_position() -> None:
    bits = np.array([0, 1, 1, 0, 0, 0, 0, 1], dtype=np.uint8)
    assert first_repetition_alarm(bits, 4) == 6
    assert first_repetition_alarm(bits, 5) is None
    assert first_repetition_alarm(np.zeros(0, dtype=np.uint8), 3) is None


def test_adaptive_alarm_counts_first_value_of_window() -> None:
    first = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)  # pierwsza wartość 1: 3 razy
    second = np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)  # pierwsza wartość 0: raz
    bits = np.concatenate([first, second, first[:3]])  # niepełne okno odpada
    assert adaptive_alarms(bits, 3, window=8).tolist() == [True, False]
    assert adaptive_alarms(bits, 4, window=8).tolist() == [False, False]


def test_adaptive_detection_probability_matches_simulation() -> None:
    rng = np.random.default_rng(SEED)
    p, cutoff, windows = 0.4, 600, 20_000
    alarms = adaptive_alarms((rng.random(windows * 1024) < p).astype(np.uint8), cutoff)
    model = adaptive_detection_probability(p, cutoff)
    assert abs(alarms.mean() - model) < 4 * math.sqrt(model * (1 - model) / windows)
