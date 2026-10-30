"""qkdsim.analysis vs niezależna implementacja qkd w każdym punkcie L (ADR 0007, F1.9)."""

import pytest

pytest.importorskip("qkd", reason="testy krzyżowe wymagają qkd: pip install -e '.[crosscheck]'")

import golden as g
import qkd_oracle as oracle
from qkdsim.analysis.key_rate import (
    GYS,
    infinite_decoy_singles,
    rate_infinite_decoy,
    rate_vacuum_weak,
    vacuum_weak_bounds,
)
from qkdsim.analysis.wcp import gain, qber

LENGTHS = [float(length) for length in range(0, 201, 10)]
REL = 1e-9


@pytest.mark.parametrize("length_km", LENGTHS)
def test_gain_and_qber(length_km: float) -> None:
    for mu in (g.VACUUM_WEAK_MU, g.VACUUM_WEAK_NU):
        eta = GYS.eta(length_km)
        q_ref, e_ref = oracle.gain_and_qber(mu, length_km)
        assert gain(mu, eta, GYS.background) == pytest.approx(q_ref, rel=REL)
        assert qber(mu, eta, GYS.background, GYS.misalignment) == pytest.approx(e_ref, rel=REL)


@pytest.mark.parametrize("length_km", LENGTHS)
def test_infinite_decoy_singles(length_km: float) -> None:
    y1, e1 = infinite_decoy_singles(GYS.eta(length_km), GYS.background, GYS.misalignment)
    y_ref, e_ref = oracle.single_photon_ideal(length_km)
    assert y1 == pytest.approx(y_ref, rel=REL)
    assert e1 == pytest.approx(e_ref, rel=REL)


@pytest.mark.parametrize("length_km", [length for length in LENGTHS if length <= 140])
def test_vacuum_weak_bounds(length_km: float) -> None:
    mu, nu = g.VACUUM_WEAK_MU, g.VACUUM_WEAK_NU
    q_mu, e_mu = oracle.gain_and_qber(mu, length_km)
    q_nu, e_nu = oracle.gain_and_qber(nu, length_km)
    y1, e1 = vacuum_weak_bounds(mu, nu, q_mu, e_mu, q_nu, e_nu, GYS.background)
    y_ref, e_ref, _q1 = oracle.vacuum_weak_bounds(length_km, mu, nu)
    assert y1 == pytest.approx(y_ref, rel=REL)
    assert e1 == pytest.approx(e_ref, rel=REL)


@pytest.mark.parametrize("length_km", LENGTHS)
def test_key_rates(length_km: float) -> None:
    mu, nu = g.VACUUM_WEAK_MU, g.VACUUM_WEAK_NU
    assert rate_infinite_decoy(GYS, length_km, mu) == pytest.approx(
        oracle.rate_infinite_decoy(length_km, mu), rel=1e-7, abs=1e-15
    )
    if length_km <= 140:
        assert rate_vacuum_weak(GYS, length_km, mu, nu) == pytest.approx(
            oracle.rate_vacuum_weak(length_km, mu, nu), rel=1e-7, abs=1e-15
        )
