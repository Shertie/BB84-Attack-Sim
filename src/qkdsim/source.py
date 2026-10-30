"""Źródło słabych impulsów koherentnych z losową fazą (F1.5).

Liczba fotonów w impulsie ~ Poisson(μ_i), gdzie i to intensywność wylosowana przez Alice
(sygnał, decoy, próżnia). Wiele intensywności jest obsługiwanych od początku, żeby stany decoy
w F3 nie wymagały przebudowy.
"""

import math
from dataclasses import dataclass

import numpy as np

from qkdsim.optics import polarization_angle
from qkdsim.protocol import AliceRecord
from qkdsim.pulses import Pulses
from qkdsim.rng import RandomSource


@dataclass(frozen=True)
class WCPSource:
    """Średnie liczby fotonów i prawdopodobieństwa ich wyboru."""

    intensities: tuple[float, ...]
    probabilities: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        if len(self.intensities) != len(self.probabilities):
            raise ValueError("intensities i probabilities muszą mieć tę samą długość")
        if not 1 <= len(self.intensities) <= 256:
            raise ValueError("liczba intensywności musi być w [1, 256]")
        if any(mu < 0.0 for mu in self.intensities):
            raise ValueError("intensywności muszą być nieujemne")
        if any(p < 0.0 for p in self.probabilities) or not math.isclose(
            sum(self.probabilities), 1.0, abs_tol=1e-9
        ):
            raise ValueError("probabilities muszą być nieujemne i sumować się do 1")

    def emit(
        self, n_pulses: int, rng: RandomSource, nature: np.random.Generator
    ) -> tuple[AliceRecord, Pulses]:
        """Alice losuje bit, bazę i intensywność; przyroda losuje liczbę fotonów."""
        bits = rng.bits(n_pulses)
        bases = rng.bases(n_pulses)
        if len(self.intensities) == 1:
            intensity = np.zeros(n_pulses, dtype=np.uint8)
        else:
            intensity = rng.choice(n_pulses, self.probabilities)
        mu = np.asarray(self.intensities, dtype=np.float64)[intensity]
        pulses = Pulses.build(
            n=nature.poisson(mu),
            bit=bits,
            basis=bases,
            intensity=intensity,
            theta=polarization_angle(bits, bases),
        )
        return AliceRecord(bits=bits, bases=bases, intensity=intensity), pulses
