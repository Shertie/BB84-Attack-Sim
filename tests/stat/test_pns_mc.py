"""Atak 3: Monte Carlo vs model PNS, estymacja decoy z danych i wykrywanie."""

import pytest

from conftest import SEED
from qkdsim.analysis.key_rate import GYS, infinite_decoy_singles, rate_vacuum_weak
from qkdsim.analysis.pns import pns_model, pns_strategy
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.attacks.pns import PhotonNumberSplitting
from qkdsim.channel import FiberChannel
from qkdsim.defenses.decoy_check import decoy_consistency_z
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.metrics import eve_known_fraction, gain, matched_qber
from qkdsim.pipeline import Counts, Link, run_counts_parallel
from qkdsim.postprocessing.decoy import estimate_decoy
from qkdsim.source import WCPSource

Z_MAX = 4.0
MU, NU, LENGTH = 0.48, 0.05, 50.0
P_D = dark_count_per_detector(GYS.background)
SOURCE = WCPSource((MU, NU, 0.0), (0.7, 0.2, 0.1))


def run(eve: PhotonNumberSplitting | None, pulses: int = 60_000_000) -> Counts:
    link = Link(
        SOURCE,
        FiberChannel(LENGTH, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
        eve=eve,
    )
    return run_counts_parallel(link, pulses, SEED, chunk_size=10_000_000)


@pytest.fixture(scope="module")
def strategy() -> tuple[float, float]:
    honest = matched_basis_clicks(MU, GYS.eta(LENGTH), P_D, GYS.misalignment)
    return pns_strategy(honest.detected, MU, GYS.eta_bob, P_D)


def test_honest_decoy_estimate_matches_analytic(strategy: tuple[float, float]) -> None:
    counts = run(None)
    estimate = estimate_decoy(counts, MU, NU, GYS.f_ec)
    y1_true, _ = infinite_decoy_singles(GYS.eta(LENGTH), GYS.background, GYS.misalignment)
    assert estimate.rate == pytest.approx(rate_vacuum_weak(GYS, LENGTH, MU, NU), rel=0.1)
    assert estimate.y1_lower < y1_true * 1.03  # granica dolna (z marginesem na szum)
    assert abs(decoy_consistency_z(counts, MU, NU)) < Z_MAX


def test_pns_matches_model_and_is_caught_by_decoy(strategy: tuple[float, float]) -> None:
    block, forward = strategy
    counts = run(PhotonNumberSplitting(block, forward))
    for i, mu_i in enumerate((MU, NU)):
        model = pns_model(mu_i, block, forward, GYS.eta_bob, P_D, GYS.misalignment)
        assert abs(gain(counts, i).z_score(model.clicks.detected)) < Z_MAX
        assert abs(matched_qber(counts, i).z_score(model.clicks.qber)) < Z_MAX
    signal = pns_model(MU, block, forward, GYS.eta_bob, P_D, GYS.misalignment)
    assert abs(eve_known_fraction(counts, 0).z_score(signal.known_fraction)) < Z_MAX
    estimate = estimate_decoy(counts, MU, NU, GYS.f_ec)
    assert estimate.aborted  # decoy: Y₁^L załamuje się, protokół przerywa
    assert estimate.naive_rate > 0.0  # naiwny rachunek dałby klucz, który zna Ewa
    assert decoy_consistency_z(counts, MU, NU) < -10.0
