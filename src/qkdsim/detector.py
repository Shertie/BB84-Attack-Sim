"""Para progowych detektorów SPAD za PBS Boba (F1.7, tryb liniowy F3.5).

Tryb Geigera (docs/model.md, rozdz. 4): fotony rozdzielają się na detektory według reguły
Borna, każdy jest wykryty z p-stwem `efficiency`, a każdy detektor ma niezależne ciemne
zliczenia z p-stwem `dark_count` na bramkę. Impuls klasyczny o mocy P to średnio
P·`gate_s`/hν fotonów, więc w trybie Geigera jasne światło po prostu wywołuje kliknięcie.

Tryb liniowy (oślepianie, Lydersen et al. 2010): ciągłe światło o mocy ≥ `blind_power_w`
zbija napięcie na diodzie poniżej przebicia. Oślepiony detektor nie widzi pojedynczych fotonów
ani nie ma ciemnych zliczeń, a klika tylko od impulsu klasycznego: nigdy dla P ≤ P_never,
zawsze dla P ≥ P_always, pomiędzy z p-stwem rosnącym liniowo (to założenie modelu; artykuł
podaje tylko oba progi).

Krzywe sprawności w czasie (atak 4, F6.1): opcjonalne `timing` zmienia sprawność każdego
detektora zależnie od czasu przyjścia impulsu `Pulses.t`. Bez `timing` czas jest ignorowany.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qkdsim.optics import analyzer_angle, prob_detector0
from qkdsim.pulses import Pulses
from qkdsim.rng import Bits

Clicks = npt.NDArray[np.bool_]
Floats = npt.NDArray[np.float64]

PHOTON_ENERGY_J = 6.62607015e-34 * 299_792_458.0 / 1550e-9  # hν przy 1550 nm


@dataclass(frozen=True)
class LinearMode:
    """Progi trybu liniowego dla detektora 0 i 1 [W, moc na wejściu Boba]."""

    blind_power_w: tuple[float, float]  # ciągłe światło, które oślepia detektor
    never_w: tuple[float, float]  # impuls nie wywołuje kliknięcia, gdy P ≤ P_never
    always_w: tuple[float, float]  # impuls zawsze wywołuje kliknięcie, gdy P ≥ P_always

    def __post_init__(self) -> None:
        for i in (0, 1):
            if not 0.0 < self.never_w[i] <= self.always_w[i]:
                raise ValueError("wymagane 0 < P_never ≤ P_always dla obu detektorów")
            if self.blind_power_w[i] <= 0.0:
                raise ValueError("moc oślepiania musi być dodatnia")

    def response(self, power: Floats, detector: int) -> Floats:
        """P-stwo kliknięcia oślepionego detektora od impulsu o mocy `power`."""
        never, always = self.never_w[detector], self.always_w[detector]
        if always == never:
            return (power >= always).astype(np.float64)
        return np.clip((power - never) / (always - never), 0.0, 1.0)

    @property
    def perfect_window_w(self) -> tuple[float, float]:
        """[max P_always, 2·min P_never]: moc impulsu wyzwalającego dla doskonałego ataku.

        Pusty (dolna granica > górna), gdy nie zachodzi warunek Lydersena (wzór (1)):
        max P_always < 2·min P_never.
        """
        return max(self.always_w), 2.0 * min(self.never_w)


@dataclass(frozen=True)
class EfficiencyMismatch:
    """Sprawność detektorów w funkcji czasu przyjścia impulsu t [ns] (atak 4, F6.1).

    η_j(t) = η·exp(−[(t − c_j)² − c_j²]/(2σ_j²)): Gauss o środku c_j i szerokości σ_j,
    unormowany tak, że w nominalnym czasie przyjścia t = 0 oba detektory mają sprawność η
    (Bob skalibrował bramki). Rozsunięcie środków c₁ − c₀ to niedopasowanie Δ z Zhao et al. 2008.
    """

    centers_ns: tuple[float, float]
    widths_ns: tuple[float, float]

    def __post_init__(self) -> None:
        if min(self.widths_ns) <= 0.0:
            raise ValueError("szerokości krzywych muszą być dodatnie")

    def relative(self, t: Floats, detector: int) -> Floats:
        """η_j(t)/η: 1 w t = 0, może przekroczyć 1 bliżej środka krzywej."""
        c, sigma = self.centers_ns[detector], self.widths_ns[detector]
        return np.exp(-(t * (t - 2.0 * c)) / (2.0 * sigma**2))


# Lydersen et al. 2010, Clavis2: oślepienie przy 397 µW (detektor 0) i 765 µW (detektor 1)
# światła ciągłego; impuls wyzwalający: P_never = 647 / 697 µW, P_always = 808 / 932 µW.
CLAVIS2 = LinearMode(
    blind_power_w=(397e-6, 765e-6), never_w=(647e-6, 697e-6), always_w=(808e-6, 932e-6)
)


@dataclass(frozen=True)
class DetectorPair:
    """Dwa detektory progowe; `linear` = None oznacza detektory, których nie da się oślepić."""

    efficiency: float  # p-stwo wykrycia fotonu, ze stratami wewnątrz Boba (η_Bob dla GYS)
    dark_count: float  # p-stwo ciemnego zliczenia na detektor i bramkę
    linear: LinearMode | None = None
    gate_s: float = 2.5e-9  # długość bramki i impulsu wyzwalającego [s]
    timing: EfficiencyMismatch | None = None  # krzywe sprawności w czasie (atak 4)

    def __post_init__(self) -> None:
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError(f"efficiency musi być w (0, 1], jest {self.efficiency}")
        if not 0.0 <= self.dark_count < 1.0:
            raise ValueError(f"dark_count musi być w [0, 1), jest {self.dark_count}")

    def blinded(self, cw_power: Floats, detector: int) -> npt.NDArray[np.bool_]:
        if self.linear is None:
            return np.zeros(cw_power.shape, dtype=np.bool_)
        return cw_power >= self.linear.blind_power_w[detector]

    def silent_probability(
        self,
        photons: npt.NDArray[np.int64],
        classical_power: Floats,
        blinded: npt.NDArray[np.bool_],
        detector: int,
        efficiency: float | Floats | None = None,
    ) -> Floats:
        """P-stwo, że detektor nie kliknie: `photons` fotonów kwantowych plus impuls klasyczny.

        `efficiency`: sprawność dla każdego impulsu (krzywe w czasie); None = `self.efficiency`.
        """
        eta = self.efficiency if efficiency is None else efficiency
        classical_photons = classical_power * self.gate_s / PHOTON_ENERGY_J
        geiger: Floats = np.asarray(
            (1.0 - self.dark_count) * (1.0 - eta) ** photons * np.exp(-eta * classical_photons),
            dtype=np.float64,
        )
        if self.linear is None or not blinded.any():
            return geiger
        return np.where(blinded, 1.0 - self.linear.response(classical_power, detector), geiger)

    def efficiencies(self, t: Floats) -> tuple[float | Floats, float | Floats]:
        """Sprawność detektora 0 i 1 dla impulsów przychodzących w czasie `t`."""
        if self.timing is None:
            return self.efficiency, self.efficiency
        eta0 = np.minimum(self.efficiency * self.timing.relative(t, 0), 1.0)
        eta1 = np.minimum(self.efficiency * self.timing.relative(t, 1), 1.0)
        return eta0, eta1

    def clicks(
        self,
        pulses: Pulses,
        bob_bases: Bits,
        nature: np.random.Generator,
        swap: Bits | None = None,
    ) -> tuple[Clicks, Clicks]:
        """Kliknięcia detektora 0 i detektora 1 dla każdego impulsu.

        `swap`: Bob obraca polaryzację o 90° przed PBS tam, gdzie swap = 1 (obrona F6.4),
        więc bit 0 trafia wtedy do detektora 1.
        """
        analyzer = analyzer_angle(bob_bases)
        if swap is not None:
            analyzer = analyzer + swap * (np.pi / 2)
        p0 = prob_detector0(pulses.theta, analyzer)
        to0 = nature.binomial(pulses.n, p0)
        to1 = pulses.n - to0
        classical = np.nan_to_num(pulses.power, nan=0.0)
        eta0, eta1 = self.efficiencies(pulses.t)
        silent0 = self.silent_probability(
            to0, classical * p0, self.blinded(pulses.cw_power, 0), 0, eta0
        )
        silent1 = self.silent_probability(
            to1, classical * (1.0 - p0), self.blinded(pulses.cw_power, 1), 1, eta1
        )
        # Porównanie z p-stwem ciszy (u ≥ cisza), jak w F1: te same ziarna dają te same wyniki.
        u = nature.random((2, len(pulses)))
        return u[0] >= silent0, u[1] >= silent1


def dark_count_per_detector(background_per_pulse: float) -> float:
    """p_d jednego detektora z łącznego tła Y₀ obu detektorów: 1 − (1 − p_d)² = Y₀."""
    if not 0.0 <= background_per_pulse < 1.0:
        raise ValueError(f"tło musi być w [0, 1), jest {background_per_pulse}")
    return float(1.0 - np.sqrt(1.0 - background_per_pulse))
