"""Oczekiwane liczniki i optymalna stopa dla skończonego klucza z decoy (Lim et al. 2014).

Dwa modele detekcji w zgodnej bazie (na impuls o średniej μ):
- `LimChannel`: model z rozdziału „Evaluation” artykułu, z afterpulsami:
  D = 1 − (1 − 2p_dc)e^(−η_sys μ), R = D(1 + p_ap), e = p_dc + e_mis(1 − e^(−η_sys μ)) + p_ap D/2.
  W e jest η_sys, a nie wydrukowane η_ch; tak odtwarzają się liczby z tekstu;
- `SimulatorChannel`: dokładny model detektorów symulatora (analysis/wcp.py), bez afterpulsów.

Lim et al. ustalają rozmiar bloku n_X (detekcje w bazie klucza) i optymalizują stopę R = ℓ/N
po (q, p₁, p₂, μ₁, μ₂) przy stałym μ₃. ε_sec = 10⁻¹⁰ i ε_cor = 10⁻¹⁵ odtwarzają liczby z tekstu
(135 km przy n_X = 10⁴, stosunek ≈ 1,75 stóp dla 10⁹ i 10⁷ przy 100 km); arXiv v1 opisuje
zamiast tego ε_sec = κℓ z κ = 10⁻¹⁵ (to daje ok. 128 km zamiast 135 km z tekstu).
"""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import differential_evolution

from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.postprocessing.finite_key_decoy import (
    BasisCounts,
    DecoyKeyLength,
    DecoySettings,
    key_length_decoy,
)

LIM_MU3 = 2e-4
LIM_EPS_SEC = 1e-10
LIM_EPS_COR = 1e-15
LIM_F_EC = 1.16


class DetectionModel(Protocol):
    def __call__(self, mu: float) -> tuple[float, float]:
        """(p-stwo detekcji, p-stwo detekcji z błędem) na impuls w zgodnej bazie."""
        ...


@dataclass(frozen=True)
class LimChannel:
    """Światłowód i detektory z oceny Lim et al. 2014."""

    length_km: float
    eta_bob: float = 0.10
    dark_count: float = 6e-7
    afterpulse: float = 4e-2
    misalignment: float = 5e-3
    alpha_db_per_km: float = 0.2

    def __call__(self, mu: float) -> tuple[float, float]:
        eta = 10.0 ** (-self.alpha_db_per_km * self.length_km / 10.0) * self.eta_bob
        clicked = -math.expm1(-eta * mu)
        detection = 1.0 - (1.0 - 2.0 * self.dark_count) * math.exp(-eta * mu)
        error = self.dark_count + self.misalignment * clicked + self.afterpulse * detection / 2.0
        return detection * (1.0 + self.afterpulse), error


@dataclass(frozen=True)
class SimulatorChannel:
    """Dokładny model detektorów symulatora; `eta` = transmisja kanału × η_Bob."""

    eta: float
    dark_count: float  # na detektor
    misalignment: float

    def __call__(self, mu: float) -> tuple[float, float]:
        clicks = matched_basis_clicks(mu, self.eta, self.dark_count, self.misalignment)
        return clicks.detected, clicks.wrong_only + 0.5 * clicks.double


@dataclass(frozen=True)
class ProtocolChoice:
    """Parametry protokołu: p-stwo bazy klucza q i ustawienia decoy."""

    key_basis: float
    settings: DecoySettings

    @classmethod
    def from_vector(cls, x: np.ndarray, mu3: float) -> "ProtocolChoice | None":
        """(q, p₁, p₂, μ₁, μ₂) albo None, gdy wektor łamie warunki protokołu."""
        q, p1, p2, mu1, mu2 = (float(v) for v in x)
        p3 = 1.0 - p1 - p2
        if not 0.0 < q < 1.0 or min(p1, p2, p3) <= 0.0 or not (mu1 > mu2 + mu3 and mu2 > mu3):
            return None
        return cls(q, DecoySettings((mu1, mu2, mu3), (p1, p2, p3)))


def expected_counts(
    choice: ProtocolChoice, channel: DetectionModel, n_key: float
) -> tuple[BasisCounts, BasisCounts, float]:
    """Oczekiwane liczniki w bazie klucza i testowej oraz liczba impulsów N dla bloku `n_key`."""
    q = choice.key_basis
    mus, ps = choice.settings.intensities, choice.settings.probabilities
    rates = [channel(mu) for mu in mus]
    pulses = n_key / (q**2 * sum(p * d for p, (d, _) in zip(ps, rates, strict=True)))

    def basis(weight: float) -> BasisCounts:
        (d1, e1), (d2, e2), (d3, e3) = (
            (pulses * weight * p * d, pulses * weight * p * e)
            for p, (d, e) in zip(ps, rates, strict=True)
        )
        return BasisCounts((d1, d2, d3), (e1, e2, e3))

    return basis(q**2), basis((1.0 - q) ** 2), pulses


def key_rate(
    choice: ProtocolChoice,
    channel: DetectionModel,
    n_key: float,
    f_ec: float = LIM_F_EC,
    eps_sec: float = LIM_EPS_SEC,
    eps_cor: float = LIM_EPS_COR,
    finite: bool = True,
) -> tuple[float, DecoyKeyLength]:
    """R = ℓ/N (bez zaokrąglenia ℓ) i składniki wzoru (1) dla oczekiwanych liczników."""
    key, test, pulses = expected_counts(choice, channel, n_key)
    length = key_length_decoy(key, test, choice.settings, f_ec, eps_sec, eps_cor, finite)
    return length.value / pulses, length


BOUNDS = ((0.01, 0.995), (0.01, 0.97), (0.01, 0.9), (0.05, 1.2), (0.002, 0.6))


def optimal_rate(
    channel: DetectionModel,
    n_key: float,
    f_ec: float = LIM_F_EC,
    eps_sec: float = LIM_EPS_SEC,
    eps_cor: float = LIM_EPS_COR,
    mu3: float = LIM_MU3,
    finite: bool = True,
    seed: int = 1,
) -> tuple[float, ProtocolChoice | None]:
    """Najlepsza stopa po (q, p₁, p₂, μ₁, μ₂) (ewolucja różnicowa); ≤ 0 oznacza brak klucza."""

    def loss(x: np.ndarray) -> float:
        choice = ProtocolChoice.from_vector(x, mu3)
        if choice is None:
            return 1.0
        return -max(key_rate(choice, channel, n_key, f_ec, eps_sec, eps_cor, finite)[0], -1.0)

    result = differential_evolution(loss, BOUNDS, seed=seed, tol=1e-12, maxiter=300, polish=True)
    return -float(result.fun), ProtocolChoice.from_vector(result.x, mu3)
