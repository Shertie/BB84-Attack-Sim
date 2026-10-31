"""Skończony klucz dla źródła WCP ze stanami decoy (Lim, Curty, Walenta, Xu, Zbinden 2014), F7.

Protokół asymetryczny: baza klucza z p-stwem q (u obu stron), baza testowa z 1 − q, trzy
intensywności μ₁ > μ₂ + μ₃, μ₂ > μ₃ ≥ 0. Z liczb detekcji n_k i błędów m_k w obu bazach:
- wzór (2): dolna granica liczby detekcji z próżni s₀;
- wzór (3): dolna granica liczby detekcji jednofotonowych s₁ (w bazie klucza i testowej);
- wzór (4): górna granica błędów jednofotonowych v₁ w bazie testowej;
- wzór (5): górna granica błędu fazowego φ = v₁/s₁ᵗ + γ(ε_sec, v₁/s₁ᵗ, s₁ᵗ, s₁);
- wzór (1): ℓ = ⌊s₀ + s₁ − s₁h(φ) − leak_EC − 6 log₂(21/ε_sec) − log₂(2/ε_cor)⌋.

Poprawki statystyczne: n±_k = (e^{μ_k}/p_k)(n_k ± √(n/2 · ln(21/ε_sec))), n = Σ_k n_k (Hoeffding;
każda z 21 granic zawodzi z p-stwem ε_sec/21). leak_EC = n f h(e) liczymy w bitach, jak `qkd`;
wydrukowane „f_EC h(e_obs)” to błąd druku (ROADMAP, K20).

Zastrzeżenie: dokumentacja `qkd` 0.2.0 powołuje się na Tupkary et al. (arXiv:2502.10340), według
których oszacowanie błędu fazowego (5) nie jest ścisłą granicą. Nie weryfikowaliśmy tego.
"""

import math
from dataclasses import dataclass

from qkdsim.analysis.entropy import h2
from qkdsim.pipeline import Counts


@dataclass(frozen=True)
class DecoySettings:
    """Intensywności (μ₁, μ₂, μ₃) i p-stwa ich wyboru."""

    intensities: tuple[float, float, float]
    probabilities: tuple[float, float, float]

    def __post_init__(self) -> None:
        mu1, mu2, mu3 = self.intensities
        if not (mu1 > mu2 + mu3 and mu2 > mu3 >= 0.0):
            raise ValueError("wymagane μ₁ > μ₂ + μ₃ i μ₂ > μ₃ ≥ 0 (Lim et al. 2014)")
        if min(self.probabilities) <= 0.0 or not math.isclose(sum(self.probabilities), 1.0):
            raise ValueError("p-stwa intensywności muszą być dodatnie i sumować się do 1")

    def tau(self, photons: int) -> float:
        """τ_n: p-stwo, że Alice wyśle stan n-fotonowy."""
        return sum(
            p * math.exp(-mu) * mu**photons / math.factorial(photons)
            for mu, p in zip(self.intensities, self.probabilities, strict=True)
        )


@dataclass(frozen=True)
class BasisCounts:
    """Detekcje i błędy w jednej bazie, osobno dla każdej intensywności."""

    detections: tuple[float, float, float]
    errors: tuple[float, float, float]

    @property
    def total(self) -> float:
        return sum(self.detections)

    @property
    def error_rate(self) -> float:
        return sum(self.errors) / self.total


def _corrected(
    counts: tuple[float, float, float], settings: DecoySettings, delta: float, sign: float
) -> list[float]:
    """n±_k = (e^{μ_k}/p_k)(n_k ± δ)."""
    return [
        math.exp(mu) / p * (n + sign * delta)
        for n, mu, p in zip(counts, settings.intensities, settings.probabilities, strict=True)
    ]


def hoeffding(total: float, eps_sec: float) -> float:
    """δ = √(n/2 · ln(21/ε_sec))."""
    return math.sqrt(total / 2.0 * math.log(21.0 / eps_sec))


def vacuum_lower(
    counts: tuple[float, float, float], settings: DecoySettings, delta: float
) -> float:
    """Wzór (2): s₀ ≥ τ₀(μ₂n⁻_{μ₃} − μ₃n⁺_{μ₂})/(μ₂ − μ₃), obcięte do zera."""
    _, mu2, mu3 = settings.intensities
    low, high = _corrected(counts, settings, delta, -1.0), _corrected(counts, settings, delta, 1.0)
    return max(0.0, settings.tau(0) * (mu2 * low[2] - mu3 * high[1]) / (mu2 - mu3))


def single_photon_lower(
    counts: tuple[float, float, float], settings: DecoySettings, delta: float, vacuum: float
) -> float:
    """Wzór (3): dolna granica detekcji jednofotonowych, obcięta do zera."""
    mu1, mu2, mu3 = settings.intensities
    low, high = _corrected(counts, settings, delta, -1.0), _corrected(counts, settings, delta, 1.0)
    bracket = low[1] - high[2] - (mu2**2 - mu3**2) / mu1**2 * (high[0] - vacuum / settings.tau(0))
    value = settings.tau(1) * mu1 * bracket / (mu1 * (mu2 - mu3) - mu2**2 + mu3**2)
    return max(0.0, value)


def single_photon_errors_upper(
    errors: tuple[float, float, float], settings: DecoySettings, delta: float
) -> float:
    """Wzór (4): v₁ ≤ τ₁(m⁺_{μ₂} − m⁻_{μ₃})/(μ₂ − μ₃)."""
    _, mu2, mu3 = settings.intensities
    low, high = _corrected(errors, settings, delta, -1.0), _corrected(errors, settings, delta, 1.0)
    return settings.tau(1) * (high[1] - low[2]) / (mu2 - mu3)


def gamma(a: float, b: float, c: float, d: float) -> float:
    """γ(a, b, c, d) = √[(c + d)(1 − b)b/(cd ln 2) · log₂((c + d)/(cd(1 − b)b) · 21²/a²)]."""
    spread = (c + d) / (c * d * (1.0 - b) * b)
    return math.sqrt(
        (c + d) * (1.0 - b) * b / (c * d * math.log(2.0)) * math.log2(spread * 441.0 / a**2)
    )


def phase_error_upper(
    errors_test: float, singles_test: float, singles_key: float, eps_sec: float, finite: bool = True
) -> float:
    """Wzór (5), obcięty do ½; bez próbki testowej albo przy b ≥ ½ zwraca ½."""
    if singles_test <= 0.0 or singles_key <= 0.0:
        return 0.5
    b = errors_test / singles_test
    if b <= 0.0:
        return 0.0
    if b >= 0.5:
        return 0.5
    spread = gamma(eps_sec, b, singles_test, singles_key) if finite else 0.0
    return min(0.5, b + spread)


@dataclass(frozen=True)
class DecoyKeyLength:
    """Składniki wzoru (1); `value` bez zaokrąglenia i obcięcia (do optymalizacji)."""

    vacuum: float
    singles: float
    singles_test: float
    errors_test: float
    phase_error: float
    n_key: float
    error_rate: float
    value: float

    @property
    def length(self) -> int:
        return max(0, math.floor(self.value))


def key_length_decoy(
    key: BasisCounts,
    test: BasisCounts,
    settings: DecoySettings,
    f_ec: float,
    eps_sec: float,
    eps_cor: float,
    finite: bool = True,
) -> DecoyKeyLength:
    """Wzór (1) z liczb detekcji i błędów. `finite` = False: bez poprawek skończonego bloku."""
    delta_key = hoeffding(key.total, eps_sec) if finite else 0.0
    delta_test = hoeffding(test.total, eps_sec) if finite else 0.0
    vacuum = vacuum_lower(key.detections, settings, delta_key)
    singles = single_photon_lower(key.detections, settings, delta_key, vacuum)
    vacuum_test = vacuum_lower(test.detections, settings, delta_test)
    singles_test = single_photon_lower(test.detections, settings, delta_test, vacuum_test)
    errors_test = single_photon_errors_upper(
        test.errors, settings, hoeffding(sum(test.errors), eps_sec) if finite else 0.0
    )
    phase = phase_error_upper(errors_test, singles_test, singles, eps_sec, finite)
    error_rate = min(0.5, key.error_rate)
    value = singles * (1.0 - h2(phase)) + vacuum - key.total * f_ec * h2(error_rate)
    if finite:
        value -= 6.0 * math.log2(21.0 / eps_sec) + math.log2(2.0 / eps_cor)
    return DecoyKeyLength(
        vacuum, singles, singles_test, errors_test, phase, key.total, error_rate, value
    )


def counts_from_simulation(counts: Counts, key_basis: int = 0) -> tuple[BasisCounts, BasisCounts]:
    """Liczniki Lima z wyniku `run_counts`: zgodne bazy, intensywności 0, 1, 2 = μ₁, μ₂, μ₃."""
    if counts.detected.shape[0] != 3:
        raise ValueError("rachunek Lima wymaga trzech intensywności")

    def basis(b: int) -> BasisCounts:
        d = [float(counts.detected[k, b, b]) for k in range(3)]
        e = [float(counts.errors[k, b, b]) for k in range(3)]
        return BasisCounts((d[0], d[1], d[2]), (e[0], e[1], e[2]))

    return basis(key_basis), basis(1 - key_basis)
