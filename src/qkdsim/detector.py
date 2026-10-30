"""Para progowych detektorów SPAD za PBS Boba (F1.7).

Model (docs/model.md, rozdz. 4): fotony rozdzielają się na detektory według reguły Borna,
każdy jest wykryty z prawdopodobieństwem `efficiency`, a każdy detektor ma niezależne ciemne
zliczenia z prawdopodobieństwem `dark_count` na bramkę.

Na razie bez trybu liniowego (oślepianie, F3), krzywych sprawności w czasie (time-shift, F6)
i czasu martwego. Pole `Pulses.t` jest ignorowane, a jasne impulsy są odrzucane z błędem
zamiast liczone po cichu źle.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qkdsim.optics import analyzer_angle, prob_detector0
from qkdsim.pulses import Pulses
from qkdsim.rng import Bits

Clicks = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class DetectorPair:
    """Dwa identyczne detektory progowe."""

    efficiency: float  # p-stwo wykrycia fotonu, ze stratami wewnątrz Boba (η_Bob dla GYS)
    dark_count: float  # p-stwo ciemnego zliczenia na detektor i bramkę

    def __post_init__(self) -> None:
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError(f"efficiency musi być w (0, 1], jest {self.efficiency}")
        if not 0.0 <= self.dark_count < 1.0:
            raise ValueError(f"dark_count musi być w [0, 1), jest {self.dark_count}")

    def clicks(
        self, pulses: Pulses, bob_bases: Bits, nature: np.random.Generator
    ) -> tuple[Clicks, Clicks]:
        """Kliknięcia detektora 0 i detektora 1 dla każdego impulsu."""
        if not np.all(np.isnan(pulses.power)):
            raise NotImplementedError("jasne impulsy wymagają trybu liniowego detektora (F3)")
        p0 = prob_detector0(pulses.theta, analyzer_angle(bob_bases))
        to0 = nature.binomial(pulses.n, p0)
        to1 = pulses.n - to0
        # Detektor nie klika, gdy nie wykrył żadnego fotonu i nie miał ciemnego zliczenia.
        miss = 1.0 - self.efficiency
        silent = 1.0 - self.dark_count
        u = nature.random((2, len(pulses)))
        return u[0] >= silent * miss**to0, u[1] >= silent * miss**to1


def dark_count_per_detector(background_per_pulse: float) -> float:
    """p_d jednego detektora z łącznego tła Y₀ obu detektorów: 1 − (1 − p_d)² = Y₀."""
    if not 0.0 <= background_per_pulse < 1.0:
        raise ValueError(f"tło musi być w [0, 1), jest {background_per_pulse}")
    return float(1.0 - np.sqrt(1.0 - background_per_pulse))
