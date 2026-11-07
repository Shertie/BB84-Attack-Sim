"""Scenariusze interfejsu interaktywnego: Monte Carlo zgodne z modelem każdego ataku."""

import math

import pytest

from qkdsim.scenarios import (
    ATTACKS,
    BLINDING,
    INTERCEPT_RESEND,
    NONE,
    TIME_SHIFT,
    TROJAN,
    WEAK_RNG,
    Attack,
    Settings,
    final_key,
    key_rates,
    simulate,
    time_shift_min_length,
)

Z_MAX = 4.0


@pytest.mark.parametrize("name", [a for a in ATTACKS if a != TIME_SHIFT])
def test_monte_carlo_matches_model(name: str) -> None:
    outcome = simulate(Settings(pulses=2_000_000), Attack(name))
    assert outcome.model_gain is not None
    assert abs(outcome.gain - outcome.model_gain) < Z_MAX * outcome.gain_sigma
    assert abs(outcome.qber - outcome.model_qber) < Z_MAX * outcome.qber_sigma
    assert abs(outcome.eve - outcome.model_eve) < Z_MAX * max(outcome.eve_sigma, 1e-3)


def test_time_shift_needs_lossy_link_and_defense_removes_information() -> None:
    minimum = time_shift_min_length(0.48)
    assert 60.0 < minimum < 70.0
    with pytest.raises(ValueError, match="km"):
        simulate(Settings(length_km=minimum - 5.0), Attack(TIME_SHIFT))
    attacked = simulate(Settings(length_km=75.0, pulses=3_000_000), Attack(TIME_SHIFT))
    defended = simulate(
        Settings(length_km=75.0, pulses=3_000_000), Attack(TIME_SHIFT, random_assignment=True)
    )
    assert math.isnan(attacked.eve_sigma)
    assert attacked.eve > 0.2 and attacked.model_eve > 0.2
    assert defended.eve < 0.05 and defended.model_eve == pytest.approx(0.0, abs=1e-12)


def test_blinding_defense_raises_alarm() -> None:
    outcome = simulate(Settings(), Attack(BLINDING, attenuator=True))
    assert outcome.defense is not None and outcome.defense_alarm
    assert outcome.model_gain is None  # model zysku nie obejmuje tłumika Boba


def test_weak_rng_lowers_qber_of_full_intercept_resend() -> None:
    fair = simulate(Settings(), Attack(INTERCEPT_RESEND))
    biased = simulate(Settings(), Attack(WEAK_RNG, basis_bias=0.9))
    assert biased.qber < fair.qber - 0.1
    assert biased.model_eve == pytest.approx(0.9, abs=0.01)


def test_final_key_vanishes_only_when_qber_reveals_the_attack() -> None:
    settings = Settings(pulses=3_000_000)
    assert final_key(settings, Attack(NONE)).final_length > 0
    assert final_key(settings, Attack(INTERCEPT_RESEND)).final_length == 0
    assert final_key(settings, Attack(TROJAN)).final_length > 0  # QBER nie zdradza ataku
    assert final_key(settings, Attack(NONE), "ldpc").verified


def test_key_rates_order_and_validation() -> None:
    rates = key_rates(0.48, 0.05, [25.0, 100.0])
    gllp, weak, ideal = rates.values()
    assert gllp[0] > 0.0 >= gllp[1]  # GLLP bez decoy kończy się ok. 40 km (e06)
    assert 0.0 < weak[1] <= ideal[1]
    with pytest.raises(ValueError):
        Attack("nieznany")
    with pytest.raises(ValueError):
        Settings(pulses=0)
