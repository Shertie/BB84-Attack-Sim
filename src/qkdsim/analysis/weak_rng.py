"""Model analityczny ataku 8: słaby generator Alice (F6.9–F6.10).

- Obciążone bazy: Ewa robi intercept-resend zawsze w częstszej bazie, więc trafia z p-stwem
  max(p_X, 1 − p_X) zamiast ½ (`intercept_resend_model` z `right_basis`). Znane ziarno: 1.
- Obciążone albo skorelowane bity: wzmacnianie prywatności liczone dla bitów jednostajnych
  zostawia klucz dłuższy niż jego entropia. Deficyt na bit to 1 − h(p) (obciążenie) albo
  1 − h(s) (łańcuch Markowa z p-stwem powtórzenia s, entropia na symbol); asymptotycznie
  tyle Ewa wie o kluczu końcowym. Dla skończonego bloku liczy się min-entropia:
  deficyt 1 + log₂ max(p, 1 − p).
- Sifting rozrzedza korelacje Markowa: zostaje co r-ty impuls w odstępach geometrycznych,
  więc sąsiednie bity klucza są równe z p-stwem ½(1 + E[λ^K]) = ½(1 + rλ/(1 − (1 − r)λ)),
  λ = 2s − 1.
- Moc testów losowości: przybliżenie normalne statystyk z defenses/randomness_tests.py.
"""

import math

from scipy.stats import norm

from qkdsim.analysis.entropy import h2


def shannon_deficit(p: float) -> float:
    """1 − h(p): o tyle bitów na bit klucza za mało entropii przy obciążeniu p."""
    return 1.0 - h2(p)


def min_entropy_deficit(p: float) -> float:
    """1 + log₂ max(p, 1 − p): deficyt min-entropii, istotny dla skończonego bloku."""
    return 1.0 + math.log2(max(p, 1.0 - p))


def sifted_stay(stay: float, sifted_fraction: float) -> float:
    """P-stwo równości sąsiednich bitów klucza po siftingu (Markow, odstępy geometryczne)."""
    lam, r = 2.0 * stay - 1.0, sifted_fraction
    return 0.5 * (1.0 + r * lam / (1.0 - (1.0 - r) * lam))


def _two_sided_power(mean: float, sd: float, alpha: float) -> float:
    """P(|X| > z_{1 − α/2}) dla X ~ N(mean, sd²)."""
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return float(norm.cdf((-z - mean) / sd) + norm.sf((z - mean) / sd))


def _check(p_one: float, stay: float) -> None:
    if p_one != 0.5 and stay != 0.5:
        raise ValueError("jedna alternatywa naraz: obciążenie (p_one) albo korelacja (stay)")


def monobit_power(n: int, p_one: float = 0.5, stay: float = 0.5, alpha: float = 0.01) -> float:
    """Moc testu monobit: S_n/√n ~ N(√n(2p − 1), 4p(1 − p)); Markow: wariancja (1 + λ)/(1 − λ)."""
    _check(p_one, stay)
    lam = 2.0 * stay - 1.0
    variance = 4.0 * p_one * (1.0 - p_one) * (1.0 + lam) / (1.0 - lam)
    return _two_sided_power(math.sqrt(n) * (2.0 * p_one - 1.0), math.sqrt(variance), alpha)


def runs_power(n: int, p_one: float = 0.5, stay: float = 0.5, alpha: float = 0.01) -> float:
    """Moc testu serii.

    Obciążenie łapie tylko warunek wstępny |π̂ − ½| ≥ 2/√n, bo statystyka serii centruje się
    na obserwowanym π̂. Korelację łapie statystyka Z = 2√n(c − ½)/(…), c = 1 − s.
    """
    _check(p_one, stay)
    sigma = math.sqrt(p_one * (1.0 - p_one) / n)
    tau = 2.0 / math.sqrt(n)
    failed = float(norm.sf((0.5 + tau - p_one) / sigma) + norm.cdf((0.5 - tau - p_one) / sigma))
    change = 1.0 - stay
    if p_one == 0.5:
        rejected = _two_sided_power(
            2.0 * math.sqrt(n) * (change - 0.5), 2.0 * math.sqrt(change * (1.0 - change)), alpha
        )
    else:
        rejected = alpha
    return failed + (1.0 - failed) * rejected


def autocorrelation_power(
    n: int, p_one: float = 0.5, stay: float = 0.5, alpha: float = 0.01
) -> float:
    """Moc testu autokorelacji (d = 1): Z = 2(A − (n − 1)/2)/√(n − 1), A = liczba zmian.

    Markow: zmiany są niezależne z p-stwem c = 1 − s. Obciążenie: c = 2p(1 − p), a sąsiednie
    wskaźniki zmian są skorelowane (kowariancja p(1 − p)(1 − 2c)).
    """
    _check(p_one, stay)
    if p_one == 0.5:
        change = 1.0 - stay
        variance = change * (1.0 - change)
    else:
        change = 2.0 * p_one * (1.0 - p_one)
        variance = change * (1.0 - change) + 2.0 * p_one * (1.0 - p_one) * (1.0 - 2.0 * change)
    mean = math.sqrt(n - 1) * (2.0 * change - 1.0)
    return _two_sided_power(mean, 2.0 * math.sqrt(variance), alpha)
