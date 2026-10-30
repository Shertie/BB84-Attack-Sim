"""Moduł qkdsim.analysis vs wartości kontrolne z publikacji (ROADMAP 3.2, F1.9)."""

import pytest
from scipy.optimize import brentq

import golden as g
from qkdsim.analysis.key_rate import (
    GYS,
    best_over_mu,
    infinite_decoy_singles,
    max_distance,
    optimal_mu,
    rate_gllp_no_decoy,
    rate_infinite_decoy,
    rate_vacuum_weak,
    shor_preskill_rate,
    vacuum_weak_bounds,
)
from qkdsim.analysis.wcp import gain, qber


def test_gys_parameters_match_golden() -> None:
    assert GYS.alpha_db_per_km == g.GYS_ALPHA_DB_PER_KM
    assert GYS.eta_bob == g.GYS_ETA_BOB
    assert GYS.background == g.GYS_Y0_PER_PULSE
    assert GYS.misalignment == g.GYS_E_DETECTOR
    assert GYS.f_ec == g.GYS_F_EC
    assert GYS.e0 == g.E0
    assert GYS.q_sift == g.Q_SIFT


def test_infinite_decoy_max_distance() -> None:
    got = max_distance(lambda length: rate_infinite_decoy(GYS, length, g.VACUUM_WEAK_MU))
    assert got == pytest.approx(g.L_MAX_INFINITE_DECOY_KM, abs=g.L_MAX_TOLERANCE_KM)


def test_vacuum_weak_max_distance() -> None:
    got = max_distance(
        lambda length: rate_vacuum_weak(GYS, length, g.VACUUM_WEAK_MU, g.VACUUM_WEAK_NU)
    )
    assert got == pytest.approx(g.L_MAX_VACUUM_WEAK_KM, abs=g.L_MAX_TOLERANCE_KM)


def test_gllp_without_decoy_max_distance() -> None:
    got = max_distance(
        lambda length: best_over_mu(lambda mu: rate_gllp_no_decoy(GYS, length, mu))[0]
    )
    assert got == pytest.approx(g.L_MAX_GLLP_NO_DECOY_KM, abs=g.L_MAX_TOLERANCE_KM)


@pytest.mark.parametrize(
    ("f_ec", "expected"), [(g.GYS_F_EC, g.MU_OPT_F_EC), (1.0, g.MU_OPT_SHANNON)]
)
def test_optimal_mu(f_ec: float, expected: float) -> None:
    assert optimal_mu(g.GYS_E_DETECTOR, f_ec) == pytest.approx(expected, abs=g.MU_OPT_TOLERANCE)


def test_vacuum_weak_deviation_at_40km() -> None:
    mu, nu = g.VACUUM_WEAK_MU, 0.25 * g.VACUUM_WEAK_MU
    eta = GYS.eta(40.0)
    args = (eta, GYS.background, GYS.misalignment)
    y1, e1 = vacuum_weak_bounds(
        mu, nu, gain(mu, eta, GYS.background), qber(mu, *args),
        gain(nu, eta, GYS.background), qber(nu, *args), GYS.background,
    )  # fmt: skip
    y_inf, e_inf = infinite_decoy_singles(*args)
    tol = g.DEVIATION_TOLERANCE_PERCENT
    assert 100 * (y_inf - y1) / y_inf == pytest.approx(g.DEVIATION_40KM_Y1_PERCENT, abs=tol)
    assert 100 * (e1 - e_inf) / e_inf == pytest.approx(g.DEVIATION_40KM_E1_PERCENT, abs=tol)


def test_shor_preskill_threshold() -> None:
    root = brentq(shor_preskill_rate, 0.01, 0.3)
    assert root == pytest.approx(g.SHOR_PRESKILL_QBER, abs=g.ANALYTIC_TOLERANCE)
