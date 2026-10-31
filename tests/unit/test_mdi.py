"""MDI-QKD: pomiar Bella w przekaźniku, Tabela 1, idealny przekaźnik i Ewa jako przekaźnik."""

import math

import numpy as np
import pytest

from golden import LCQ12_MIN_RANGE_KM
from qkdsim.analysis.mdi import coherent_gains, max_distance, single_photon_pair
from qkdsim.mdi import (
    FAIL,
    PSI_MINUS,
    PSI_PLUS,
    MeasureAndAnnounce,
    Relay,
    bell_outcome,
    bob_flips,
)
from qkdsim.optics import polarization_angle

LCQ12 = Relay.with_misalignment(0.145, 3.01e-6, 0.015)


def test_bell_outcome_patterns() -> None:
    patterns = np.array(
        [
            [1, 0, 0, 1],  # 1H, 2V → ψ⁻
            [0, 1, 1, 0],  # 1V, 2H → ψ⁻
            [1, 1, 0, 0],  # 1H, 1V → ψ⁺
            [0, 0, 1, 1],  # 2H, 2V → ψ⁺
            [1, 0, 1, 0],  # ta sama polaryzacja: nie
            [1, 1, 1, 0],  # trzy kliknięcia: nie
            [0, 0, 0, 1],
        ],
        dtype=bool,
    ).T
    expected = [PSI_MINUS, PSI_MINUS, PSI_PLUS, PSI_PLUS, FAIL, FAIL, FAIL]
    assert list(bell_outcome(patterns)) == expected


def test_table_1_bit_flips() -> None:
    bases = np.array([0, 0, 1, 1], dtype=np.uint8)
    outcome = np.array([PSI_MINUS, PSI_PLUS, PSI_MINUS, PSI_PLUS], dtype=np.int8)
    assert list(bob_flips(bases, outcome)) == [True, True, True, False]


@pytest.mark.parametrize("basis", [0, 1])
def test_ideal_relay_has_no_single_photon_errors(basis: int) -> None:
    """Różne bity: fotony rozróżnialne, pomiar zawsze się udaje; równe: HOM łączy je w jednym."""
    yield_11, error_11 = single_photon_pair(1.0, 1.0, Relay(1.0, 0.0), basis)
    assert yield_11 == pytest.approx(0.5)
    assert error_11 == pytest.approx(0.0, abs=1e-15)


def test_misalignment_split_into_two_rotations() -> None:
    assert math.sin(LCQ12.rotation_in) ** 2 == pytest.approx(0.0075)
    assert math.sin(LCQ12.rotation_out) ** 2 == pytest.approx(0.0075)
    _, error_z = coherent_gains(0.3, 0.3, LCQ12, 0)
    _, error_z_ideal = coherent_gains(0.3, 0.3, Relay(0.145, 3.01e-6), 0)
    assert error_z > error_z_ideal


def test_reproduces_lcq12_range() -> None:
    """Przekaźnik pośrodku toleruje ponad 40 dB, czyli ponad 200 km światłowodu."""
    assert max_distance(LCQ12, 1.16) > LCQ12_MIN_RANGE_KM


def test_relay_attack_announces_only_anticorrelated_key_bits() -> None:
    rng = np.random.default_rng(1)
    size = 100_000
    bits_a, bits_b = (
        rng.integers(0, 2, size, dtype=np.uint8),
        rng.integers(0, 2, size, dtype=np.uint8),
    )
    zeros = np.zeros(size, dtype=np.uint8)
    mu = np.full(size, 5.0)
    outcome, eve = MeasureAndAnnounce().announce(
        mu, mu, polarization_angle(bits_a, zeros), polarization_angle(bits_b, zeros), rng
    )
    announced = outcome == PSI_MINUS
    assert (bits_a[announced] != bits_b[announced]).all()
    assert np.array_equal(eve[announced], bits_a[announced])
