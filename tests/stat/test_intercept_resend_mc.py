"""Atak 1: Monte Carlo vs model analityczny (F3.4). Szybkie testy; pełna siatka w test_gate_g3."""

import numpy as np
import pytest

from conftest import SEED
from qkdsim.analysis.intercept_resend import intercept_resend_model
from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.attacks.intercept_resend import InterceptResend
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.metrics import eve_guess_accuracy, eve_known_fraction, gain, matched_qber
from qkdsim.pipeline import Link, run_bits, run_counts
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

Z_MAX = 4.0
MU = 0.48
P_D = dark_count_per_detector(GYS.background)


def attacked_link(length_km: float, fraction: float, position: float) -> Link:
    return Link(
        WCPSource((MU,)),
        FiberChannel(length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
        eve=InterceptResend(fraction, position),
    )


@pytest.mark.parametrize("position", [0.0, 1.0])
def test_full_attack_matches_model(position: float) -> None:
    counts = run_counts(
        attacked_link(25.0, 1.0, position), 4_000_000, RandomStreams.from_seed(SEED)
    )
    model = intercept_resend_model(GYS, 25.0, MU, 1.0, position, P_D)
    assert abs(gain(counts).z_score(model.clicks.detected)) < Z_MAX
    assert abs(matched_qber(counts).z_score(model.clicks.qber)) < Z_MAX
    assert abs(eve_known_fraction(counts).z_score(model.known_fraction)) < Z_MAX
    # Wśród impulsów, które Ewa zmierzyła, trafia w 3/4 przypadków (plan: 75%).
    assert abs(eve_guess_accuracy(counts).z_score(0.75)) < Z_MAX


def test_qber_rises_to_about_a_quarter() -> None:
    counts = run_counts(attacked_link(25.0, 1.0, 1.0), 4_000_000, RandomStreams.from_seed(SEED))
    assert 0.25 < matched_qber(counts).value < 0.28


def test_eve_next_to_alice_changes_gain() -> None:
    """Drugi detektor ataku: Ewa tuż za Alice wysyła 1 foton zamiast średnio μ."""
    counts = run_counts(attacked_link(25.0, 1.0, 0.0), 4_000_000, RandomStreams.from_seed(SEED))
    honest = matched_basis_clicks(MU, GYS.eta(25.0), P_D, GYS.misalignment).detected
    assert gain(counts).z_score(honest) < -10.0


def test_bits_layer_eve_guesses_three_quarters_of_sifted_key() -> None:
    run = run_bits(attacked_link(0.0, 1.0, 1.0), 1_000_000, RandomStreams.from_seed(SEED))
    measured = run.eve.measured[run.key.index]
    guesses = run.eve.bit[run.key.index][measured]
    truth = run.key.alice[measured]
    accuracy = float(np.mean(guesses == truth))
    assert abs(accuracy - 0.75) < Z_MAX * np.sqrt(0.75 * 0.25 / len(truth))
