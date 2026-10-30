"""Własności wzorów analitycznych dla dowolnych parametrów (F1.13, hypothesis)."""

import math

from hypothesis import given
from hypothesis import strategies as st

from qkdsim.analysis.key_rate import GYS, rate_infinite_decoy
from qkdsim.analysis.wcp import gain, matched_basis_clicks, qber

mus = st.floats(min_value=1e-4, max_value=2.0)
lengths = st.floats(min_value=0.0, max_value=300.0)
etas = st.floats(min_value=1e-12, max_value=1.0)
backgrounds = st.floats(min_value=0.0, max_value=1e-3)
misalignments = st.floats(min_value=0.0, max_value=0.5)


@given(mu=mus, eta=etas, p_d=st.floats(min_value=0.0, max_value=0.1), e_d=misalignments)
def test_simulator_qber_is_a_valid_error_rate(
    mu: float, eta: float, p_d: float, e_d: float
) -> None:
    c = matched_basis_clicks(mu, eta, p_d, e_d)
    if c.detected > 0.0:
        assert 0.0 <= c.qber <= 0.5 + 1e-12


@given(mu=mus, eta=etas, y0=backgrounds, e_d=misalignments)
def test_paper_qber_is_valid_below_its_corner(mu: float, eta: float, y0: float, e_d: float) -> None:
    """Wzór Ma et al. liczy błędy bez odjęcia nakładania sygnału i tła, a zysk z odjęciem.

    Dlatego E_μ ≤ 1/2 tylko dla e_d ≤ (1 − Y₀)/2; powyżej (np. e_d = 1/2 i duże Y₀) wzór daje
    E_μ nieco ponad 1/2. Znalezione przez hypothesis; fizyczne parametry są daleko od tego
    narożnika.
    """
    if gain(mu, eta, y0) > 0.0 and e_d <= 0.5 * (1.0 - y0):
        assert 0.0 <= qber(mu, eta, y0, e_d) <= 0.5 + 1e-12


@given(mu=mus, step=st.floats(min_value=1e-3, max_value=1.0), length=lengths)
def test_gain_grows_with_mu(mu: float, step: float, length: float) -> None:
    eta = GYS.eta(length)
    assert gain(mu + step, eta, GYS.background) >= gain(mu, eta, GYS.background)


@given(mu=mus, length=lengths, step=st.floats(min_value=0.1, max_value=50.0))
def test_gain_falls_with_distance(mu: float, length: float, step: float) -> None:
    assert gain(mu, GYS.eta(length + step), GYS.background) <= gain(
        mu, GYS.eta(length), GYS.background
    )


@given(mu=mus, eta=etas, p_d=st.floats(min_value=0.0, max_value=0.1), e_d=misalignments)
def test_click_model_is_a_probability_distribution(
    mu: float, eta: float, p_d: float, e_d: float
) -> None:
    c = matched_basis_clicks(mu, eta, p_d, e_d)
    parts = (c.right_only, c.wrong_only, c.double, c.none)
    assert all(0.0 <= p <= 1.0 for p in parts)
    assert math.isclose(sum(parts), 1.0, abs_tol=1e-12)


@given(length=lengths, mu=st.floats(min_value=0.05, max_value=1.0))
def test_key_rate_is_finite(length: float, mu: float) -> None:
    assert math.isfinite(rate_infinite_decoy(GYS, length, mu))
