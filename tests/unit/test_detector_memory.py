"""Pamięć detektora: czas martwy i afterpulsy."""

import math

import numpy as np
import pytest

from conftest import SEED
from qkdsim.detector import DetectorPair, stationary_click_rate, with_memory
from qkdsim.pulses import Pulses

N = 2_000_000


@pytest.mark.parametrize(("dead", "afterpulse"), [(-1, 0.0), (0, -0.1), (0, 1.0)])
def test_detector_rejects_invalid_memory(dead: int, afterpulse: float) -> None:
    with pytest.raises(ValueError):
        DetectorPair(0.5, 0.0, dead_time_gates=dead, afterpulse=afterpulse)


def test_dead_time_drops_clicks_inside_the_window() -> None:
    fresh = np.zeros(20, dtype=np.bool_)
    fresh[[0, 2, 3, 4, 10, 13, 14]] = True
    out = with_memory(fresh, 3, 0.0, np.random.default_rng(SEED))
    assert np.flatnonzero(out).tolist() == [0, 4, 10, 14]


def test_certain_afterpulse_fires_in_the_first_live_gate() -> None:
    fresh = np.zeros(12, dtype=np.bool_)
    fresh[0] = True
    out = with_memory(fresh, 2, 0.999999999, np.random.default_rng(SEED))
    assert np.flatnonzero(out).tolist() == [0, 3, 6, 9]


def test_memoryless_detector_keeps_fresh_clicks() -> None:
    fresh = np.random.default_rng(SEED).random(10_000) < 0.3
    assert np.array_equal(with_memory(fresh, 0, 0.0, np.random.default_rng(SEED)), fresh)


def test_stationary_rate_limits() -> None:
    assert stationary_click_rate(0.2, 0, 0.0) == pytest.approx(0.2)
    assert stationary_click_rate(0.2, 9, 0.0) == pytest.approx(0.2 / (1 + 9 * 0.2))
    assert stationary_click_rate(0.2, 0, 0.1) == pytest.approx(0.2 / (1 - 0.1 * 0.8))
    assert stationary_click_rate(1.0, 4, 0.5) == pytest.approx(1 / 5)


@pytest.mark.parametrize(
    ("fresh_rate", "dead", "afterpulse"),
    [(1e-2, 0, 0.3), (1e-2, 50, 0.0), (3e-2, 20, 0.2), (0.3, 3, 0.1)],
)
def test_click_rate_matches_renewal_formula(
    fresh_rate: float, dead: int, afterpulse: float
) -> None:
    rng = np.random.default_rng(SEED)
    out = with_memory(rng.random(N) < fresh_rate, dead, afterpulse, rng)
    expected = stationary_click_rate(fresh_rate, dead, afterpulse)
    assert abs(out.mean() - expected) < 4.0 * math.sqrt(expected / N)


def test_memory_detector_reproduces_memoryless_seeds_when_disabled() -> None:
    pulses = Pulses.build(
        n=np.ones(1000), bit=np.zeros(1000), basis=np.zeros(1000), theta=np.zeros(1000)
    )
    bases = np.zeros(1000, dtype=np.uint8)
    plain = DetectorPair(0.3, 0.01).clicks(pulses, bases, np.random.default_rng(SEED))
    explicit = DetectorPair(0.3, 0.01, dead_time_gates=0, afterpulse=0.0).clicks(
        pulses, bases, np.random.default_rng(SEED)
    )
    assert all(np.array_equal(a, b) for a, b in zip(plain, explicit, strict=True))
