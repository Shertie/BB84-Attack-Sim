"""Monte Carlo vs dokładny model detektorów (porównanie przez z-score).

Szybkie testy na krótkich dystansach; pełna walidacja na 0–150 km jest w test_validation_link.py.
"""

import pytest

from conftest import SEED
from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.wcp import click_probabilities, matched_basis_clicks, mismatched_basis_clicks
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.metrics import double_click_rate, gain, matched_qber
from qkdsim.pipeline import Link, run_counts
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

Z_MAX = 4.0
P_D = dark_count_per_detector(GYS.background)


def gys_link(length_km: float, mu: float = 0.48) -> Link:
    return Link(
        WCPSource((mu,)),
        FiberChannel(length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
    )


@pytest.mark.parametrize("length_km", [0.0, 25.0, 50.0])
def test_gain_and_qber_at_short_distance(length_km: float) -> None:
    counts = run_counts(gys_link(length_km), 3_000_000, RandomStreams.from_seed(SEED))
    model = matched_basis_clicks(0.48, GYS.eta(length_km), P_D, GYS.misalignment)
    assert abs(gain(counts).z_score(model.detected)) < Z_MAX
    assert abs(matched_qber(counts).z_score(model.qber)) < Z_MAX


def test_qber_is_the_same_in_both_bases() -> None:
    counts = run_counts(gys_link(0.0), 3_000_000, RandomStreams.from_seed(SEED))
    z_basis, x_basis = matched_qber(counts, basis=0), matched_qber(counts, basis=1)
    diff = z_basis.value - x_basis.value
    sigma = (z_basis.sigma**2 + x_basis.sigma**2) ** 0.5
    assert abs(diff) < Z_MAX * sigma


def test_click_statistics_in_strong_multiphoton_regime() -> None:
    """Duże μ i e_d: dużo podwójnych kliknięć, więc test sprawdza niezależność detektorów
    (rozkład koherentny dzieli się na niezależne rozkłady Poissona, docs/model.md)."""
    mu, e_d, eff, p_d = 2.0, 0.2, 0.6, 0.01
    link = Link(WCPSource((mu,)), FiberChannel(0.0, 0.21, e_d), DetectorPair(eff, p_d))
    counts = run_counts(link, 1_000_000, RandomStreams.from_seed(SEED))
    matched = matched_basis_clicks(mu, eff, p_d, e_d)
    mismatched = mismatched_basis_clicks(mu, eff, p_d, e_d)
    assert abs(gain(counts).z_score(matched.detected)) < Z_MAX
    assert abs(matched_qber(counts).z_score(matched.qber)) < Z_MAX
    assert abs(double_click_rate(counts, matched=True).z_score(matched.double)) < Z_MAX
    assert abs(double_click_rate(counts, matched=False).z_score(mismatched.double)) < Z_MAX


def test_vacuum_decoy_gives_background_only() -> None:
    link = Link(
        WCPSource((0.48, 0.0), (0.5, 0.5)),
        FiberChannel(0.0, 0.21, GYS.misalignment),
        DetectorPair(GYS.eta_bob, 1e-3),
    )
    counts = run_counts(link, 2_000_000, RandomStreams.from_seed(SEED))
    vacuum = click_probabilities(0.0, 0.0, 1e-3, 0.5)
    assert abs(gain(counts, intensity=1).z_score(vacuum.detected)) < Z_MAX
    assert abs(matched_qber(counts, intensity=1).z_score(0.5)) < Z_MAX
