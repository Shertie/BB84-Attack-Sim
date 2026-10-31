"""Długość klucza: Shor–Preskill, GLLP, stany decoy (Ma, Qi, Zhao, Lo 2005).

Wszystkie stopy są w bitach na wysłany impuls. Wartości kontrolne: tests/golden.py.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from qkdsim.analysis.entropy import h2
from qkdsim.analysis.wcp import gain, qber


@dataclass(frozen=True)
class LinkParameters:
    """Parametry łącza w notacji Ma et al. 2005."""

    alpha_db_per_km: float
    eta_bob: float  # sprawność Boba łącznie z detektorem
    background: float  # Y₀, łącznie dla obu detektorów, na impuls
    misalignment: float  # e_d
    f_ec: float  # sprawność korekcji błędów
    e0: float = 0.5
    q_sift: float = 0.5

    def eta(self, length_km: float) -> float:
        """η = η_Bob · 10^(−αL/10)."""
        return self.eta_bob * float(10.0 ** (-self.alpha_db_per_km * length_km / 10.0))


# Ma et al. 2005, Tabela I; f_EC = 1,22 z tekstu artykułu (docs/model.md, rozdz. 3).
GYS = LinkParameters(
    alpha_db_per_km=0.21, eta_bob=0.045, background=1.7e-6, misalignment=0.033, f_ec=1.22
)
GYS_REPETITION_RATE_HZ = 2e6  # kolumna „f” Tabeli I


def shor_preskill_rate(error_rate: float, f_ec: float = 1.0) -> float:
    """r = 1 − f h(Q) − h(Q): źródło jednofotonowe, granica asymptotyczna."""
    return 1.0 - f_ec * h2(error_rate) - h2(error_rate)


def gllp_rate(
    q_mu: float, e_mu: float, q1: float, e1: float, f_ec: float, q_sift: float = 0.5
) -> float:
    """R = q{−Q_μ f h(E_μ) + Q₁[1 − h(e₁)]} (GLLP; Ma et al., wzór (1)).

    e₁ powyżej 1/2 jest obcinane do 1/2, bo h jest symetryczna i inaczej „poprawiałaby” stopę.
    """
    return q_sift * (-q_mu * f_ec * h2(e_mu) + q1 * (1.0 - h2(min(e1, 0.5))))


def infinite_decoy_singles(
    eta: float, background: float, misalignment: float, e0: float = 0.5
) -> tuple[float, float]:
    """Y₁ = Y₀ + η − Y₀η i e₁ = (e₀Y₀ + e_dη)/Y₁: granica nieskończenie wielu stanów decoy."""
    y1 = background + eta - background * eta
    return y1, (e0 * background + misalignment * eta) / y1


def vacuum_weak_bounds(
    mu: float,
    nu: float,
    q_mu: float,
    e_mu: float,
    q_nu: float,
    e_nu: float,
    background: float,
    e0: float = 0.5,
) -> tuple[float, float]:
    """Y₁^L i e₁^U z protokołu vacuum+weak (Ma et al., oszacowania VWY1Bound i VWe1Bound)."""
    y1 = (mu / (mu * nu - nu * nu)) * (
        q_nu * math.exp(nu)
        - q_mu * math.exp(mu) * nu * nu / (mu * mu)
        - (mu * mu - nu * nu) / (mu * mu) * background
    )
    e1 = (e_nu * q_nu * math.exp(nu) - e0 * background) / (y1 * nu)
    return y1, e1


def _gain_qber(p: LinkParameters, length_km: float, mu: float) -> tuple[float, float]:
    eta = p.eta(length_km)
    return gain(mu, eta, p.background), qber(mu, eta, p.background, p.misalignment, p.e0)


def rate_infinite_decoy(p: LinkParameters, length_km: float, mu: float) -> float:
    """Stopa GLLP z Y₁ i e₁ dla nieskończenie wielu stanów decoy (Ma et al. 2005)."""
    q_mu, e_mu = _gain_qber(p, length_km, mu)
    y1, e1 = infinite_decoy_singles(p.eta(length_km), p.background, p.misalignment, p.e0)
    return gllp_rate(q_mu, e_mu, y1 * mu * math.exp(-mu), e1, p.f_ec, p.q_sift)


def rate_vacuum_weak(p: LinkParameters, length_km: float, mu: float, nu: float) -> float:
    """Stopa GLLP z granicami Y₁ i e₁ z decoy vacuum + weak (Ma et al. 2005)."""
    q_mu, e_mu = _gain_qber(p, length_km, mu)
    q_nu, e_nu = _gain_qber(p, length_km, nu)
    y1, e1 = vacuum_weak_bounds(mu, nu, q_mu, e_mu, q_nu, e_nu, p.background, p.e0)
    return gllp_rate(q_mu, e_mu, y1 * mu * math.exp(-mu), e1, p.f_ec, p.q_sift)


def rate_gllp_no_decoy(p: LinkParameters, length_km: float, mu: float) -> float:
    """GLLP bez decoy: najgorszy przypadek, w którym Ewa zna wszystkie impulsy wielofotonowe."""
    q_mu, e_mu = _gain_qber(p, length_km, mu)
    multi = (1.0 - math.exp(-mu) * (1.0 + mu)) / q_mu
    if multi >= 1.0:
        return -q_mu  # nic nie zostaje dla klucza; wartość ujemna, ciągła z sąsiedztwem
    single = 1.0 - multi
    return gllp_rate(q_mu, e_mu, q_mu * single, e_mu / single, p.f_ec, p.q_sift)


_MU_GRID = np.geomspace(1e-4, 1.5, 400)


def best_over_mu(rate: Callable[[float], float]) -> tuple[float, float]:
    """(max stopa, μ_opt). Siatka logarytmiczna, potem dokładna optymalizacja wokół maksimum.

    Sama optymalizacja lokalna gubi optimum, gdy dodatnia stopa istnieje tylko w wąskim
    zakresie małych μ (GLLP bez decoy na długim dystansie).
    """
    values = [rate(float(mu)) for mu in _MU_GRID]
    i = int(np.argmax(values))
    lo = float(_MU_GRID[max(i - 1, 0)])
    hi = float(_MU_GRID[min(i + 1, len(_MU_GRID) - 1)])
    res = minimize_scalar(
        lambda mu: -rate(mu), bounds=(lo, hi), method="bounded", options={"xatol": 1e-9}
    )
    if -float(res.fun) >= values[i]:
        return -float(res.fun), float(res.x)
    return values[i], float(_MU_GRID[i])


def optimal_mu(misalignment: float, f_ec: float) -> float:
    """Ma et al., wzór (12): (1 − μ)e^(−μ) = f h(e_d) / (1 − h(e_d))."""
    target = f_ec * h2(misalignment) / (1.0 - h2(misalignment))
    return float(brentq(lambda mu: (1.0 - mu) * math.exp(-mu) - target, 0.0, 1.0))


def max_distance(rate: Callable[[float], float], hi_km: float = 400.0) -> float:
    """Odległość, przy której stopa klucza spada do zera (bisekcja)."""
    lo = 0.0
    if rate(lo) <= 0.0:
        raise ValueError("stopa klucza w 0 km musi być dodatnia")
    if rate(hi_km) > 0.0:
        raise ValueError(f"stopa klucza nadal dodatnia w {hi_km} km; zwiększ hi_km")
    for _ in range(60):
        mid = 0.5 * (lo + hi_km)
        if rate(mid) > 0.0:
            lo = mid
        else:
            hi_km = mid
    return lo
