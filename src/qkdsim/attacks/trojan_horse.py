"""Atak 5: Trojan horse (Gisin et al. 2006), F6.5–F6.6.

Ewa wpuszcza do Alice sondę o średnio μ_in fotonach; wraca do niej stan koherentny
o μ_out = μ_in·10^(−A/10) z polaryzacją ustawioną przez modulator Alice (analysis/trojan_horse.py).
Sygnału nie dotyka, więc QBER i zysk się nie zmieniają. Sondę trzyma do ujawnienia baz i wtedy
mierzy ją optymalnie; wynik tego pomiaru losujemy z jego dokładnego rozkładu:
- bez losowej fazy (pomiar Helstroma): trafia bit z p-stwem ½(1 + √(1 − e^(−2μ_out)));
  w `EveLog` zapis jak dla zgadywania: `measured` = True, `basis` = NO_BASIS;
- z losową fazą (licznik fotonów w bazie Alice): wraca n ~ Poisson(μ_out) fotonów, a przy n ≥ 1
  Ewa zna bit na pewno, więc `measured` = n ≥ 1 i `basis` = baza Alice.
"""

from dataclasses import dataclass

import numpy as np

from qkdsim.analysis.trojan_horse import bit_guess_probability, returned_photons
from qkdsim.eve import NO_BASIS, EveLog
from qkdsim.pulses import Pulses
from qkdsim.rng import Bits


def modulator_bit(pulses: Pulses) -> Bits:
    """Bit zapisany w ustawieniu modulatora: θ = bπ/4 + xπ/2, a bazę b Ewa pozna po siftingu."""
    return (np.rint(2.0 * pulses.theta / np.pi - 0.5 * pulses.basis).astype(np.int64) % 2).astype(
        np.uint8
    )


@dataclass(frozen=True)
class TrojanHorse:
    """Atak 5: Ewa sonduje modulator Alice jasnym impulsem; sygnału nie dotyka."""

    probe_photons: float  # μ_in: średnia liczba fotonów sondy na wejściu Alice
    attenuation_db: float  # A: tłumienie toru sondy tam i z powrotem, z izolacją i odbiciem
    phase_randomized: bool = False  # Alice losuje fazę światła, które może się odbić
    position: float = 0.0  # Ewa tuż za Alice...
    lossless_after: bool = False  # ...a sygnał idzie dalej zwykłym światłowodem

    def __post_init__(self) -> None:
        if self.probe_photons < 0.0:
            raise ValueError("probe_photons musi być nieujemne")

    @property
    def returned_photons(self) -> float:
        return returned_photons(self.probe_photons, self.attenuation_db)

    def intercept(self, pulses: Pulses, rng: np.random.Generator) -> tuple[Pulses, EveLog]:
        size, mu = len(pulses), self.returned_photons
        bits = modulator_bit(pulses)
        if self.phase_randomized:
            caught = rng.poisson(mu, size) > 0
            log = EveLog(
                measured=caught,
                basis=np.where(caught, pulses.basis, NO_BASIS).astype(np.uint8),
                bit=np.where(caught, bits, 0).astype(np.uint8),
            )
        else:
            correct = rng.random(size) < bit_guess_probability(mu)
            log = EveLog(
                measured=np.ones(size, dtype=np.bool_),
                basis=np.full(size, NO_BASIS, dtype=np.uint8),
                bit=np.where(correct, bits, 1 - bits).astype(np.uint8),
            )
        return pulses, log
