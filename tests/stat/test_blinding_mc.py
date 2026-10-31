"""Atak 2: Monte Carlo vs model analityczny i wykrywanie przez obrony."""

import math

import pytest

from conftest import SEED
from qkdsim.analysis.blinding import blinding_model
from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.wcp import gain as gain_model
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.attacks.blinding import DetectorBlinding, matching_send_fraction
from qkdsim.channel import FiberChannel
from qkdsim.defenses.random_attenuation import RandomAttenuator, attenuation_z
from qkdsim.detector import CLAVIS2, DetectorPair, dark_count_per_detector
from qkdsim.metrics import attenuation_tally, eve_known_fraction, gain, matched_qber
from qkdsim.pipeline import Link, run_counts
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

Z_MAX = 4.0
MU, LENGTH = 0.48, 25.0
P_D = dark_count_per_detector(GYS.background)
DETECTORS = DetectorPair(GYS.eta_bob, P_D, CLAVIS2)
HONEST = matched_basis_clicks(MU, GYS.eta(LENGTH), P_D, GYS.misalignment)
THETA_E = math.asin(math.sqrt(GYS.misalignment))


def attack(power: float, calibrated: bool = True) -> DetectorBlinding:
    return DetectorBlinding(
        power,
        send_fraction=matching_send_fraction(HONEST.detected, MU),
        mimic_qber=HONEST.qber,
        compensation_rad=THETA_E if calibrated else 0.0,
    )


def link(eve: DetectorBlinding | None, attenuator: RandomAttenuator | None = None) -> Link:
    return Link(
        WCPSource((MU,)),
        FiberChannel(LENGTH, GYS.alpha_db_per_km, GYS.misalignment),
        DETECTORS,
        eve=eve,
        attenuator=attenuator,
    )


def test_calibrated_attack_in_window_is_invisible_and_eve_knows_everything() -> None:
    eve = attack(1.1e-3)
    counts = run_counts(link(eve), 4_000_000, RandomStreams.from_seed(SEED))
    model = blinding_model(DETECTORS, eve, MU, GYS.misalignment)
    assert abs(gain(counts).z_score(model.detected)) < Z_MAX
    assert abs(gain(counts).z_score(HONEST.detected)) < Z_MAX  # zysk jak bez ataku
    assert abs(matched_qber(counts).z_score(HONEST.qber)) < Z_MAX  # QBER jak bez ataku
    assert eve_known_fraction(counts).value == 1.0  # Ewa zna każdy bit po siftingu


@pytest.mark.parametrize(
    ("power", "calibrated"), [(1.1e-3, False), (1.5e-3, True), (0.85e-3, True)]
)
def test_attack_outside_perfect_conditions_matches_model(power: float, calibrated: bool) -> None:
    eve = attack(power, calibrated)
    counts = run_counts(link(eve), 4_000_000, RandomStreams.from_seed(SEED))
    model = blinding_model(DETECTORS, eve, MU, GYS.misalignment)
    assert abs(gain(counts).z_score(model.detected)) < Z_MAX
    assert abs(matched_qber(counts).z_score(model.qber)) < Z_MAX
    known = eve_known_fraction(counts)
    if model.known_fraction == 1.0:
        assert known.value == 1.0  # oślepione detektory nie mają ciemnych zliczeń
    else:
        assert abs(known.z_score(model.known_fraction)) < Z_MAX


def test_naive_eve_is_revealed_by_qber() -> None:
    """Bez kompensacji obrotu podział mocy w bazie sprzężonej to 32/68, nie 50/50."""
    counts = run_counts(
        link(attack(1.1e-3, calibrated=False)), 4_000_000, RandomStreams.from_seed(SEED)
    )
    assert matched_qber(counts).value > 0.10


def test_random_attenuation_detects_calibrated_attack() -> None:
    attenuator = RandomAttenuator(fraction=0.1, factor=0.5)
    eta = GYS.eta(LENGTH)
    ratio = gain_model(MU, 0.5 * eta, GYS.background) / gain_model(MU, eta, GYS.background)
    honest = run_counts(link(None, attenuator), 4_000_000, RandomStreams.from_seed(SEED))
    attacked = run_counts(
        link(attack(1.1e-3), attenuator), 4_000_000, RandomStreams.from_seed(SEED)
    )
    assert abs(attenuation_z(attenuation_tally(honest), ratio)) < Z_MAX
    assert attenuation_z(attenuation_tally(attacked), ratio) < -10.0
