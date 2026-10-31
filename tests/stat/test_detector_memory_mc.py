"""Tor z afterpulsami: Monte Carlo a dokładny model (afterpuls jako ciemne zliczenie), F7."""

import pytest

from conftest import SEED
from qkdsim.analysis.finite_key_decoy import LimChannel
from qkdsim.analysis.wcp import afterpulse_dark_count, matched_basis_clicks
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair
from qkdsim.metrics import gain, matched_qber
from qkdsim.pipeline import Link, run_counts
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

Z_MAX = 4.0
LIM = LimChannel(0.0)


@pytest.mark.parametrize(("length_km", "mu"), [(0.0, 0.5), (0.0, 3.0), (20.0, 0.5)])
def test_afterpulse_link_matches_exact_model(length_km: float, mu: float) -> None:
    link = Link(
        WCPSource((mu,)),
        FiberChannel(length_km, LIM.alpha_db_per_km, LIM.misalignment),
        DetectorPair(LIM.eta_bob, LIM.dark_count, afterpulse=0.1),
    )
    counts = run_counts(link, 3_000_000, RandomStreams.from_seed(SEED))
    eta = 10.0 ** (-LIM.alpha_db_per_km * length_km / 10.0) * LIM.eta_bob
    dark = afterpulse_dark_count(mu, eta, LIM.dark_count, LIM.misalignment, 0.1)
    model = matched_basis_clicks(mu, eta, dark, LIM.misalignment)
    assert abs(gain(counts).z_score(model.detected)) < Z_MAX
    assert abs(matched_qber(counts).z_score(model.qber)) < Z_MAX


def test_lim_formula_is_first_order_of_exact_model() -> None:
    """Przy rzadkich kliknięciach wzory Lima et al. z p_ap = p/(1 − p) zgadzają się z dokładnym."""
    p = 0.04
    eta = 10.0 ** (-LIM.alpha_db_per_km * 50.0 / 10.0) * LIM.eta_bob
    dark = afterpulse_dark_count(0.5, eta, LIM.dark_count, LIM.misalignment, p)
    exact = matched_basis_clicks(0.5, eta, dark, LIM.misalignment)
    detection, error = LimChannel(50.0, afterpulse=p / (1 - p))(0.5)
    assert exact.detected == pytest.approx(detection, rel=2e-3)
    assert exact.qber == pytest.approx(error / detection, rel=2e-3)
