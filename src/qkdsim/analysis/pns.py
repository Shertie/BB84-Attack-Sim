"""Model analityczny ataku PNS (photon-number splitting), F3.11–F3.12.

Ewa stoi tuż za Alice i mierzy liczbę fotonów n bez naruszania polaryzacji (pomiar QND):
- n = 1: blokuje impuls z p-stwem `block`, inaczej przepuszcza foton;
- n ≥ 2: zabiera 1 foton do pamięci (zmierzy go po ujawnieniu baz, więc zna bit na pewno)
  i z p-stwem `forward` wysyła resztę, n − 1 fotonów;
- wszystko, co przepuszcza, idzie do Boba linią bezstratną, więc Bob widzi tylko η_Bob.

Ewa nie widzi intensywności, tylko n, więc ta sama strategia działa na sygnał i na decoy.
Dobiera ją tak, by zysk sygnału był jak bez ataku; zysk decoy wychodzi wtedy niespójny.
"""

import math
from dataclasses import dataclass

from qkdsim.analysis.wcp import ClickProbabilities

MAX_PHOTONS = 40  # ogon rozkładu Poissona powyżej jest pomijalny dla μ ≤ 2


def _poisson(mu: float, n: int) -> float:
    return math.exp(-mu + n * math.log(mu) - math.lgamma(n + 1)) if mu > 0 else float(n == 0)


def fock_clicks(k: int, eta: float, dark_count: float, misalignment: float) -> ClickProbabilities:
    """Stan k fotonów w zgodnej bazie: każdy wykryty z p-stwem η, w złym detektorze z p. e_d."""
    silent = 1.0 - dark_count
    right, wrong = eta * (1.0 - misalignment), eta * misalignment
    none = silent**2 * (1.0 - eta) ** k
    right_only = silent * (1.0 - wrong) ** k - none
    wrong_only = silent * (1.0 - right) ** k - none
    return ClickProbabilities(
        right_only=right_only,
        wrong_only=wrong_only,
        double=1.0 - right_only - wrong_only - none,
        none=none,
    )


@dataclass(frozen=True)
class PNSModel:
    """P-stwa kliknięć przy PNS i detekcji, których bit Ewa zna na pewno."""

    clicks: ClickProbabilities  # zgodna baza, na impuls danej intensywności
    known: float  # detekcja z impulsu wielofotonowego: Ewa zna bit na pewno

    @property
    def known_fraction(self) -> float:
        return self.known / self.clicks.detected


def pns_model(
    mu: float, block: float, forward: float, eta_bob: float, dark_count: float, misalignment: float
) -> PNSModel:
    """Wartość oczekiwana symulatora dla impulsu o średniej μ pod atakiem PNS."""
    weights: dict[int, float] = {}  # liczba fotonów u Boba → p-stwo
    known = 0.0
    for n in range(MAX_PHOTONS):
        p = _poisson(mu, n)
        if n == 0:
            weights[0] = weights.get(0, 0.0) + p
        elif n == 1:
            weights[1] = weights.get(1, 0.0) + p * (1.0 - block)
            weights[0] = weights.get(0, 0.0) + p * block
        else:
            weights[n - 1] = weights.get(n - 1, 0.0) + p * forward
            weights[0] = weights.get(0, 0.0) + p * (1.0 - forward)
            known += p * (
                forward * fock_clicks(n - 1, eta_bob, dark_count, misalignment).detected
                + (1.0 - forward) * fock_clicks(0, eta_bob, dark_count, misalignment).detected
            )
    parts = [(w, fock_clicks(k, eta_bob, dark_count, misalignment)) for k, w in weights.items()]
    clicks = ClickProbabilities(
        right_only=sum(w * c.right_only for w, c in parts),
        wrong_only=sum(w * c.wrong_only for w, c in parts),
        double=sum(w * c.double for w, c in parts),
        none=sum(w * c.none for w, c in parts),
    )
    return PNSModel(clicks, known)


def pns_strategy(
    target_gain: float, mu: float, eta_bob: float, dark_count: float
) -> tuple[float, float]:
    """(block, forward), przy których zysk impulsu μ wynosi `target_gain`.

    Najpierw Ewa blokuje impulsy jednofotonowe (z nich nic by nie wiedziała). Jeśli same impulsy
    wielofotonowe dają za duży zysk, przepuszcza tylko ich część; jeśli za mały, przepuszcza
    część jednofotonowych.
    """
    silent = (1.0 - dark_count) ** 2
    needed = 1.0 - (1.0 - target_gain) / silent  # p-stwo wykrycia przepuszczonych fotonów
    multi = sum(_poisson(mu, n) * (1.0 - (1.0 - eta_bob) ** (n - 1)) for n in range(2, MAX_PHOTONS))
    if multi >= needed:
        return 1.0, max(0.0, needed) / multi
    single = _poisson(mu, 1) * eta_bob
    return max(0.0, 1.0 - (needed - multi) / single), 1.0


def critical_transmittance(mu: float, eta_bob: float) -> float:
    """Transmisja kanału, od której Ewa może zablokować wszystkie impulsy jednofotonowe.

    Zysk bez ataku 1 − (1 − Y₀)e^(−η_Bob·T·μ) zrównuje się wtedy z zyskiem samych impulsów
    wielofotonowych przepuszczonych bezstratnie, 1 − (1 − Y₀)(1 − M), gdzie M to p-stwo
    wykrycia ich fotonów. Stąd T = −ln(1 − M)/(η_Bob·μ), niezależnie od ciemnych zliczeń.
    """
    multi = sum(_poisson(mu, n) * (1.0 - (1.0 - eta_bob) ** (n - 1)) for n in range(2, MAX_PHOTONS))
    return -math.log1p(-multi) / (eta_bob * mu)
