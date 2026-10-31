"""Testy krzyżowe: wartości kontrolne z publikacji vs. niezależna implementacja qkd.

Sprawdzają, że qkd odtwarza liczby z golden.py; te same funkcje z qkd_oracle
porównują się z modułem qkdsim.analysis (test_analysis_vs_qkd.py).
"""

import math

import pytest

pytest.importorskip("qkd", reason="testy krzyżowe wymagają qkd: pip install -e '.[crosscheck]'")

import golden as g
import qkd_oracle as oracle


def test_infinite_decoy_max_distance() -> None:
    got = oracle.max_distance(oracle.rate_infinite_decoy)
    assert got == pytest.approx(g.L_MAX_INFINITE_DECOY_KM, abs=g.L_MAX_TOLERANCE_KM)


def test_vacuum_weak_max_distance() -> None:
    got = oracle.max_distance(oracle.rate_vacuum_weak)
    assert got == pytest.approx(g.L_MAX_VACUUM_WEAK_KM, abs=g.L_MAX_TOLERANCE_KM)
    assert got < oracle.max_distance(oracle.rate_infinite_decoy)


def test_gllp_without_decoy_max_distance() -> None:
    got = oracle.max_distance(oracle.best_rate_gllp_no_decoy)
    assert got == pytest.approx(g.L_MAX_GLLP_NO_DECOY_KM, abs=g.L_MAX_TOLERANCE_KM)
    # Stany decoy wydłużają zasięg kilkukrotnie.
    assert oracle.max_distance(oracle.rate_vacuum_weak) > 3.0 * got


def test_vacuum_weak_deviation_at_40km() -> None:
    mu = g.VACUUM_WEAK_MU
    y1, e1, _q1 = oracle.vacuum_weak_bounds(40.0, mu=mu, nu=0.25 * mu)
    y_inf, e_inf = oracle.single_photon_ideal(40.0)
    assert 100.0 * (y_inf - y1) / y_inf == pytest.approx(
        g.DEVIATION_40KM_Y1_PERCENT, abs=g.DEVIATION_TOLERANCE_PERCENT
    )
    assert 100.0 * (e1 - e_inf) / e_inf == pytest.approx(
        g.DEVIATION_40KM_E1_PERCENT, abs=g.DEVIATION_TOLERANCE_PERCENT
    )


@pytest.mark.parametrize("length_km", range(0, 201, 10))
def test_gain_approximation_is_negligible(length_km: int) -> None:
    """Wzór (10) Ma et al. pomija korektę podwójnego liczenia, którą ma qkd."""
    mu = g.VACUUM_WEAK_MU
    q_exact, e_exact = oracle.gain_and_qber(mu, length_km)
    detected = 1.0 - math.exp(-oracle.transmittance(length_km) * mu)
    q_approx = g.GYS_Y0_PER_PULSE + detected
    e_approx = (g.E0 * g.GYS_Y0_PER_PULSE + g.GYS_E_DETECTOR * detected) / q_approx
    assert q_approx == pytest.approx(q_exact, rel=1e-5)
    assert e_approx == pytest.approx(e_exact, rel=1e-5)


def test_lydersen_thresholds_allow_perfect_blinding() -> None:
    ratio = oracle.blinding_ratio(g.LYDERSEN_P_ALWAYS_W, g.LYDERSEN_P_NEVER_W)
    assert ratio == pytest.approx(max(g.LYDERSEN_P_ALWAYS_W) / min(g.LYDERSEN_P_NEVER_W))
    assert ratio < 2.0
