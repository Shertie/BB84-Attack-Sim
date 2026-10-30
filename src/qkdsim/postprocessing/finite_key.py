"""Długość klucza przy skończonej próbce (Tomamichel, Lim, Gisin, Renner 2012), F4.1–F4.2.

Protokół asymetryczny: n bitów klucza z jednej bazy, k bitów testu z drugiej. Alice i Bob
przerywają, gdy QBER z próbki przekroczy próg Q_tol ustalony z góry; ℓ zależy od Q_tol,
a nie od wyniku pomiaru. Źródło jednofotonowe (dla WCP z decoy: Lim et al. 2014, F7).

- `key_length`: twierdzenie z tekstu głównego (wzór (2)):
  ℓ = ⌊n(q − h(Q_tol + μ)) − leak_EC − log₂(2/(ε_sec²·ε_cor))⌋,
  μ = √[(n + k)/(nk) · (k + 1)/k · ln(4/ε_sec)];
- `key_length_tight`: twierdzenie z sekcji Methods, z optymalizacją po ε i ε̄ (2ε + ε̄ ≤ ε_sec),
  μ(ε) z ln(1/ε) i członem 2·log₂(1/(2ε̄));
- `key_length_plan_hoeffding`: uproszczenie z planu projektu, Q^U = Q̂ + √(ln(1/ε)/(2k)) na samej
  próbce. Nie uwzględnia przeniesienia błędu z próbki na klucz (losowanie bez zwracania),
  więc zawyża ℓ (ROADMAP, K3).

Uwaga (ROADMAP, K18): oba wydrukowane twierdzenia dają stopy o 8–14% niższe niż Tabela 1
artykułu przy jej parametrach. Implementacja trzyma się wzorów; rozbieżność jest otwarta.
"""

import math

import numpy as np
from scipy.stats import binom

from qkdsim.analysis.entropy import h2


def _h(x: float) -> float:
    """Obcięta entropia binarna: h(x) dla x ≤ ½, 1 powyżej (przypis do twierdzenia)."""
    return 1.0 if x >= 0.5 else h2(max(x, 0.0))


def statistical_deviation(n: int, k: int, log_term: float) -> float:
    """μ = √[(n + k)/(nk) · (k + 1)/k · log_term], log_term = ln(4/ε_sec) albo ln(1/ε)."""
    if n <= 0 or k <= 0:
        raise ValueError("n i k muszą być dodatnie")
    return math.sqrt((n + k) / (n * k) * (k + 1) / k * log_term)


def key_length(
    n: int,
    k: int,
    q_tol: float,
    leak_ec: float,
    eps_sec: float,
    eps_cor: float,
    preparation_quality: float = 1.0,
) -> int:
    mu = statistical_deviation(n, k, math.log(4.0 / eps_sec))
    secret = n * (preparation_quality - _h(q_tol + mu))
    return max(0, math.floor(secret - leak_ec - math.log2(2.0 / (eps_sec**2 * eps_cor))))


def key_length_tight(
    n: int,
    k: int,
    q_tol: float,
    leak_ec: float,
    eps_sec: float,
    eps_cor: float,
    preparation_quality: float = 1.0,
    grid: int = 199,
) -> int:
    best = -math.inf
    for fraction in np.linspace(0.005, 0.995, grid):
        eps = float(fraction) * eps_sec / 2.0
        eps_bar = eps_sec - 2.0 * eps
        mu = statistical_deviation(n, k, math.log(1.0 / eps))
        value = (
            n * (preparation_quality - _h(q_tol + mu))
            - 2.0 * math.log2(1.0 / (2.0 * eps_bar))
            - leak_ec
            - math.log2(2.0 / eps_cor)
        )
        best = max(best, value)
    return max(0, math.floor(best))


def key_length_plan_hoeffding(
    n: int, k: int, qber_sample: float, leak_ec: float, eps_sec: float, eps_cor: float
) -> int:
    """Wersja z planu projektu (ROADMAP, K3): zawyża ℓ, do porównania."""
    upper = qber_sample + math.sqrt(math.log(1.0 / eps_sec) / (2.0 * k))
    secret = n * (1.0 - _h(upper))
    return max(0, math.floor(secret - leak_ec - math.log2(2.0 / (eps_sec**2 * eps_cor))))


def abort_probability(k: int, qber: float, q_tol: float) -> float:
    """P(QBER z k bitów próbki > Q_tol) przy prawdziwym QBER: ε_rob dla uczciwego kanału,
    p-stwo wykrycia, gdy QBER podnosi Ewa (dokładny ogon rozkładu dwumianowego)."""
    return float(binom.sf(math.floor(q_tol * k), k, qber))


def exchanged_signals(n: int, k: int) -> float:
    """M(n, k) = (√n + √k)²: oczekiwana liczba wykrytych sygnałów przy p_x = (1 + √(k/n))⁻¹."""
    return (math.sqrt(n) + math.sqrt(k)) ** 2


def expected_key_rate(
    n: int,
    k: int,
    qber: float,
    q_tol: float,
    eps_sec: float,
    eps_cor: float,
    xi: float = 1.1,
    tight: bool = False,
) -> float:
    """(1 − ε_rob)·ℓ/M(n, k) z leak_EC = ξ·n·h(Q_tol), jak w ocenie numerycznej artykułu."""
    length = (key_length_tight if tight else key_length)(
        n, k, q_tol, xi * n * h2(q_tol), eps_sec, eps_cor
    )
    return (1.0 - abort_probability(k, qber, q_tol)) * length / exchanged_signals(n, k)


def optimal_key_rate(
    n: int,
    qber: float,
    eps_sec: float,
    eps_cor: float,
    xi: float = 1.1,
    tight: bool = False,
) -> tuple[float, int, float]:
    """Maksimum `expected_key_rate` po k i Q_tol (siatka); zwraca (stopa, k, Q_tol)."""
    best = (0.0, 1, qber)
    for ratio in np.geomspace(1e-3, 1.0, 40):
        k = max(10, round(float(ratio) * n))
        for q_tol in np.linspace(qber, min(qber + 0.06, 0.2), 40):
            rate = expected_key_rate(n, k, qber, float(q_tol), eps_sec, eps_cor, xi, tight)
            if rate > best[0]:
                best = (rate, k, float(q_tol))
    return best
