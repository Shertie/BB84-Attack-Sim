"""Skończony klucz z decoy (Lim et al. 2014): wzory (1)–(5) i modele kanału (F7)."""

import math

import pytest

from qkdsim.analysis.finite_key_decoy import (
    LimChannel,
    ProtocolChoice,
    SimulatorChannel,
    expected_counts,
    key_rate,
)
from qkdsim.postprocessing.finite_key_decoy import (
    DecoySettings,
    gamma,
    key_length_decoy,
    phase_error_upper,
)

SETTINGS = DecoySettings((0.5, 0.1, 2e-4), (0.6, 0.3, 0.1))
CHOICE = ProtocolChoice(0.8, SETTINGS)


def test_settings_validation_and_tau() -> None:
    with pytest.raises(ValueError):
        DecoySettings((0.3, 0.2, 0.15), (0.6, 0.3, 0.1))  # μ₁ ≤ μ₂ + μ₃
    with pytest.raises(ValueError):
        DecoySettings((0.5, 0.1, 0.0), (0.6, 0.3, 0.2))
    assert sum(SETTINGS.tau(n) for n in range(30)) == pytest.approx(1.0)
    assert SETTINGS.tau(0) == pytest.approx(
        0.6 * math.exp(-0.5) + 0.3 * math.exp(-0.1) + 0.1 * math.exp(-2e-4)
    )


def test_gamma_and_phase_error() -> None:
    a, b, c, d = 1e-10, 0.03, 1e5, 2e5
    expected = math.sqrt(
        (c + d)
        * (1 - b)
        * b
        / (c * d * math.log(2))
        * math.log2((c + d) / (c * d * (1 - b) * b) * 441 / a**2)
    )
    assert gamma(a, b, c, d) == pytest.approx(expected)
    assert phase_error_upper(3000.0, 1e5, 2e5, 1e-10) == pytest.approx(
        0.03 + gamma(1e-10, 0.03, 1e5, 2e5)
    )
    assert phase_error_upper(3000.0, 1e5, 2e5, 1e-10, finite=False) == pytest.approx(0.03)
    assert phase_error_upper(1.0, 0.0, 1e5, 1e-10) == 0.5
    assert phase_error_upper(6e4, 1e5, 2e5, 1e-10) == 0.5


def test_expected_counts_fill_the_block() -> None:
    key, test, pulses = expected_counts(CHOICE, LimChannel(50.0), 1e6)
    assert key.total == pytest.approx(1e6)
    assert test.total == pytest.approx(1e6 * (0.2 / 0.8) ** 2)
    assert pulses > 1e6


def test_finite_length_grows_towards_asymptotic() -> None:
    channel = SimulatorChannel(
        eta=0.045 * 10 ** (-0.21 * 25 / 10), dark_count=8.5e-7, misalignment=0.033
    )
    asymptotic, _ = key_rate(CHOICE, channel, 1e6, f_ec=1.22, finite=False)
    rates = [key_rate(CHOICE, channel, n, f_ec=1.22)[0] for n in (1e5, 1e6, 1e8, 1e10)]
    assert rates == sorted(rates)
    assert rates[-1] < asymptotic
    assert rates[-1] == pytest.approx(asymptotic, rel=0.05)


def test_length_terms() -> None:
    key, test, _ = expected_counts(CHOICE, LimChannel(50.0), 1e7)  # 10⁶: za mało testu, φ = ½
    result = key_length_decoy(key, test, SETTINGS, 1.16, 1e-10, 1e-15)
    assert 0.0 <= result.vacuum < result.singles < key.total  # s₀ często 0: δ ≫ n_{μ₃}
    assert result.length == math.floor(result.value)
    loose = key_length_decoy(key, test, SETTINGS, 1.16, 1e-20, 1e-15)
    assert loose.value < result.value  # mniejsze ε_sec, dłuższy margines, krótszy klucz
