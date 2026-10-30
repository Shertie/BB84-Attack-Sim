"""Skończony klucz (Tomamichel et al. 2012) i porównanie z wersją z planu (F4.1–F4.3)."""

import math

import pytest

from qkdsim.analysis.entropy import h2
from qkdsim.postprocessing.finite_key import (
    abort_probability,
    exchanged_signals,
    key_length,
    key_length_plan_hoeffding,
    key_length_tight,
    optimal_key_rate,
    statistical_deviation,
)

EPS_SEC, EPS_COR = 1e-10, 1e-12


def test_statistical_deviation_formula() -> None:
    n, k = 10_000, 1_000
    expected = math.sqrt(11_000 / 10_000_000 * 1001 / 1000 * math.log(4 / EPS_SEC))
    assert statistical_deviation(n, k, math.log(4 / EPS_SEC)) == pytest.approx(expected)
    with pytest.raises(ValueError):
        statistical_deviation(0, 10, 1.0)


def test_key_length_matches_printed_theorem() -> None:
    n, k, q_tol, leak = 100_000, 10_000, 0.02, 15_000.0
    mu = statistical_deviation(n, k, math.log(4 / EPS_SEC))
    expected = math.floor(n * (1 - h2(q_tol + mu)) - leak - math.log2(2 / (EPS_SEC**2 * EPS_COR)))
    assert key_length(n, k, q_tol, leak, EPS_SEC, EPS_COR) == expected


@pytest.mark.parametrize(
    ("n", "k", "q"), [(10**4, 2_000, 0.01), (10**5, 10**4, 0.02), (10**6, 3 * 10**4, 0.03)]
)
def test_tight_theorem_is_not_worse_and_plan_version_overestimates(
    n: int, k: int, q: float
) -> None:
    leak = 1.1 * n * h2(q)
    main = key_length(n, k, q, leak, EPS_SEC, EPS_COR)
    assert key_length_tight(n, k, q, leak, EPS_SEC, EPS_COR) >= main
    assert key_length_plan_hoeffding(n, k, q, leak, EPS_SEC, EPS_COR) > main


def test_key_length_vanishes_when_correction_reaches_half() -> None:
    assert key_length(1000, 10, 0.4, 0.0, EPS_SEC, EPS_COR) == 0


def test_key_length_grows_with_block_and_falls_with_tolerance() -> None:
    lengths = [
        key_length(n, n // 10, 0.02, 1.1 * n * h2(0.02), EPS_SEC, EPS_COR)
        for n in (10**4, 10**5, 10**6)
    ]
    assert lengths == sorted(lengths)
    assert key_length(10**5, 10**4, 0.03, 0.0, EPS_SEC, EPS_COR) < key_length(
        10**5, 10**4, 0.02, 0.0, EPS_SEC, EPS_COR
    )


def test_abort_probability_is_binomial_tail() -> None:
    k, q = 1000, 0.02
    tail = sum(math.comb(k, e) * q**e * (1 - q) ** (k - e) for e in range(21, k + 1))
    assert abort_probability(k, q, 0.02) == pytest.approx(tail, rel=1e-9)
    assert abort_probability(1000, 0.02, 0.05) < 1e-7
    assert abort_probability(1000, 0.10, 0.05) > 0.999


def test_exchanged_signals_balances_both_bases() -> None:
    n, k = 10**6, 2 * 10**4
    p_x = 1 / (1 + math.sqrt(k / n))
    assert exchanged_signals(n, k) == pytest.approx(n / p_x**2)
    assert exchanged_signals(n, k) == pytest.approx(k / (1 - p_x) ** 2)


def test_optimal_rate_approaches_asymptote() -> None:
    rates = [optimal_key_rate(n, 0.01, EPS_SEC, EPS_COR)[0] for n in (10**4, 10**6, 10**8)]
    assert rates == sorted(rates)
    assert 0.8 < rates[-1] / (1 - 2 * h2(0.01)) < 1.0


@pytest.mark.parametrize(
    ("n", "qber", "table"),
    [(10**5, 0.01, 0.304), (10**5, 0.025, 0.215), (10**6, 0.01, 0.478), (10**6, 0.025, 0.357)],
)
def test_optimized_rate_close_to_paper_table(n: int, qber: float, table: float) -> None:
    """Tabela 1 Tomamichela et al.: optymalizacja po k i Q_tol daje stopy w granicach 10%.

    Optymalne parametry różnią się od tabeli (ROADMAP, K18), więc sprawdzamy tylko stopę.
    """
    rate, _k, _q_tol = optimal_key_rate(n, qber, EPS_SEC, EPS_COR)
    assert rate == pytest.approx(table, rel=0.1)
