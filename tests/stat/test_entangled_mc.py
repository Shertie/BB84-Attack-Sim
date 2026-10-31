"""Splątane pary: Monte Carlo kontra model (BBM92, CHSH) i fałszywe naruszenie CHSH (F7)."""

import math

import pytest

from conftest import SEED
from qkdsim.analysis.entangled import chsh_value, key_setting_qber, pair_statistics
from qkdsim.entangled import DetectorControl, simulate_chsh, simulate_pairs
from qkdsim.rng import RandomStreams

Z_MAX = 4.0


@pytest.mark.parametrize(
    ("lam", "eta_a", "eta_b", "dark", "misalignment"),
    [(0.05, 0.1, 0.1, 1e-5, 0.03), (0.3, 0.2, 0.05, 1e-4, 0.0)],
)
def test_pair_source_matches_model(
    lam: float, eta_a: float, eta_b: float, dark: float, misalignment: float
) -> None:
    counts = simulate_pairs(
        lam, eta_a, eta_b, dark, misalignment, 10_000_000, RandomStreams.from_seed(SEED)
    )
    model = pair_statistics(lam, eta_a, eta_b, dark, misalignment)
    expected = model.gain * counts.pulses
    assert abs(counts.coincidences - expected) < Z_MAX * math.sqrt(expected)
    wrong = model.qber * counts.coincidences
    assert abs(counts.errors - wrong) < Z_MAX * math.sqrt(wrong * (1 - model.qber))


def test_chsh_counts_match_model() -> None:
    counts = simulate_chsh(0.9, 0.98, 3_000_000, RandomStreams.from_seed(SEED))
    tolerance = Z_MAX * 4 * math.sqrt(1 / 500_000)  # cztery korelacje z ~500 tys. rund
    assert counts.chsh() == pytest.approx(chsh_value(0.9, 0.98), abs=tolerance)
    assert counts.chsh(postselected=True) == pytest.approx(chsh_value(1.0, 0.98), abs=tolerance)
    assert counts.key_qber() == pytest.approx(key_setting_qber(0.9, 0.98), abs=0.005)


def test_detector_control_fakes_violation_only_in_coincidences() -> None:
    """Gerhardt i in. 2011: na koincydencjach S = 4p, na wszystkich rundach S = p + ½ < 2."""
    fidelity = 1 / math.sqrt(2)
    counts = simulate_chsh(
        0.1, 1.0, 3_000_000, RandomStreams.from_seed(SEED), DetectorControl(fidelity)
    )
    assert counts.chsh(postselected=True) == pytest.approx(4 * fidelity, abs=0.03)
    assert counts.chsh() == pytest.approx(fidelity + 0.5, abs=0.02)
    assert counts.chsh() < 2.0
