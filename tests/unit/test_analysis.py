"""Moduł analityczny: entropia, modele zysku i QBER, wzory stopy klucza (F1.9)."""

import math

import pytest

from qkdsim.analysis.entropy import h2
from qkdsim.analysis.key_rate import (
    GYS,
    gllp_rate,
    infinite_decoy_singles,
    shor_preskill_rate,
    vacuum_weak_bounds,
)
from qkdsim.analysis.wcp import (
    click_probabilities,
    gain,
    gain_eq10,
    matched_basis_clicks,
    mismatched_basis_clicks,
    qber,
    qber_squashed,
)
from qkdsim.detector import dark_count_per_detector


def test_binary_entropy_edges_and_symmetry() -> None:
    assert h2(0.0) == 0.0
    assert h2(1.0) == 0.0
    assert h2(0.5) == pytest.approx(1.0)
    assert h2(0.11) == pytest.approx(h2(0.89))
    with pytest.raises(ValueError):
        h2(1.5)


def test_gain_eq10_differs_from_exact_sum_by_overlap_term() -> None:
    mu, eta, y0 = 0.48, 0.045, 1.7e-6
    overlap = y0 * (1 - math.exp(-eta * mu))
    assert gain_eq10(mu, eta, y0) - gain(mu, eta, y0) == pytest.approx(overlap, rel=1e-6)


def test_qber_without_signal_is_background_error() -> None:
    assert qber(0.48, 0.0, 1e-6, 0.033) == pytest.approx(0.5)


@pytest.mark.parametrize("fraction", [0.0, 0.3, 0.5, 0.967, 1.0])
def test_click_probabilities_sum_to_one(fraction: float) -> None:
    c = click_probabilities(2.0, 0.5, 0.01, fraction)
    assert c.right_only + c.wrong_only + c.double + c.none == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("length_km", [0.0, 50.0, 100.0, 150.0, 200.0])
def test_simulator_model_matches_paper_model(length_km: float) -> None:
    """Dokładny model detektorów daje ten sam zysk co Ma et al. i QBER różny o < 3·10⁻⁵."""
    eta, p_d = GYS.eta(length_km), dark_count_per_detector(GYS.background)
    clicks = matched_basis_clicks(0.48, eta, p_d, GYS.misalignment)
    assert clicks.detected == pytest.approx(gain(0.48, eta, GYS.background), rel=1e-12)
    paper = qber(0.48, eta, GYS.background, GYS.misalignment)
    assert qber_squashed(0.48, eta, p_d, GYS.misalignment) == pytest.approx(paper, rel=3e-5)


def test_gain_does_not_depend_on_basis() -> None:
    args = (0.48, 0.045, 1e-3, 0.033)
    assert matched_basis_clicks(*args).detected == pytest.approx(
        mismatched_basis_clicks(*args).detected, rel=1e-14
    )


def test_shor_preskill_rate_vanishes_at_threshold() -> None:
    assert shor_preskill_rate(0.0) == 1.0
    assert shor_preskill_rate(0.110028) == pytest.approx(0.0, abs=1e-5)


def test_gllp_rate_formula_and_clipping() -> None:
    expected = 0.5 * (-0.01 * 1.22 * h2(0.03) + 0.006 * (1 - h2(0.04)))
    assert gllp_rate(0.01, 0.03, 0.006, 0.04, 1.22) == pytest.approx(expected)
    # e₁ > 1/2 nie może zwiększać stopy.
    assert gllp_rate(0.01, 0.03, 0.006, 0.7, 1.22) == gllp_rate(0.01, 0.03, 0.006, 0.5, 1.22)


def test_infinite_decoy_without_background() -> None:
    y1, e1 = infinite_decoy_singles(0.3, 0.0, 0.033)
    assert y1 == pytest.approx(0.3)
    assert e1 == pytest.approx(0.033)


def test_vacuum_weak_bounds_bracket_true_values() -> None:
    """Y₁^L ≤ Y₁ i e₁^U ≥ e₁ (oszacowania od bezpiecznej strony)."""
    for length_km in (0.0, 40.0, 100.0):
        eta = GYS.eta(length_km)
        args = (eta, GYS.background, GYS.misalignment)
        q_mu, e_mu = gain(0.48, eta, GYS.background), qber(0.48, *args)
        q_nu, e_nu = gain(0.05, eta, GYS.background), qber(0.05, *args)
        y1_lo, e1_hi = vacuum_weak_bounds(0.48, 0.05, q_mu, e_mu, q_nu, e_nu, GYS.background)
        y1, e1 = infinite_decoy_singles(*args)
        assert y1_lo <= y1
        assert e1_hi >= e1
