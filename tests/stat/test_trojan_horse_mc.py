"""Atak 5 w torze impulsów: QBER i zysk bez zmian, wiedza Ewy zgodna z modelem."""

import math

import pytest

from conftest import SEED
from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.trojan_horse import bit_guess_probability
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.attacks.trojan_horse import TrojanHorse
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.metrics import eve_guess_accuracy, eve_known_fraction, gain, matched_qber
from qkdsim.pipeline import Counts, Link, run_counts_parallel
from qkdsim.source import WCPSource

Z_MAX = 4.0
MU, LENGTH = 0.48, 25.0
P_D = dark_count_per_detector(GYS.background)
ATTACK = TrojanHorse(1e9, 100.0)  # wraca 0,1 fotonu, jak w przykładzie Gisina et al.


def run(random_phase: bool) -> Counts:
    link = Link(
        WCPSource((MU,)),
        FiberChannel(LENGTH, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
        eve=TrojanHorse(ATTACK.probe_photons, ATTACK.attenuation_db, random_phase),
    )
    return run_counts_parallel(link, 20_000_000, SEED, chunk_size=5_000_000)


@pytest.mark.parametrize("random_phase", [False, True])
def test_signal_untouched_and_eve_matches_model(random_phase: bool) -> None:
    counts = run(random_phase)
    honest = matched_basis_clicks(MU, GYS.eta(LENGTH), P_D, GYS.misalignment)
    mu = ATTACK.returned_photons
    assert abs(matched_qber(counts).z_score(honest.qber)) < Z_MAX
    assert abs(gain(counts).z_score(honest.detected)) < Z_MAX  # zysk nie zależy od baz
    if random_phase:
        assert abs(eve_known_fraction(counts).z_score(-math.expm1(-mu))) < Z_MAX
    else:
        assert abs(eve_guess_accuracy(counts).z_score(bit_guess_probability(mu))) < Z_MAX
