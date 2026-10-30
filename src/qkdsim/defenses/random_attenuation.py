"""Obrona (b) przed oślepianiem: losowe tłumienie przed detektorami (F3.9).

Bob w losowym, tajnym ułamku `fraction` bramek wstawia tłumik o transmisji `factor`.
- Bez ataku (tryb Geigera) odsetek kliknięć w tych bramkach spada o znany czynnik
  ρ = Q(μ, factor·η) / Q(μ, η), bliski `factor` przy małym ημ.
- Przy oślepianiu impuls wyzwalający po stłumieniu spada poniżej P_never i kliknięć prawie
  nie ma, albo (gdy Ewa podbije moc) cała moc pozostaje nad progiem i spadku nie ma wcale.

Test: z-score różnicy między odsetkiem kliknięć w bramkach tłumionych a ρ razy odsetek
w pozostałych. Tłumik nie zmienia stanu oślepienia, bo oślepienie zależy od średniego
fotoprądu, a nie od mocy w jednej bramce.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qkdsim.pulses import Pulses
from qkdsim.rng import RandomSource


@dataclass(frozen=True)
class AttenuationTally:
    """Wysłane i wykryte impulsy: wszystkie oraz w bramkach z tłumikiem."""

    sent: int
    detected: int
    sent_attenuated: int
    detected_attenuated: int


@dataclass(frozen=True)
class RandomAttenuator:
    fraction: float = 0.1  # ułamek bramek z tłumikiem
    factor: float = 0.5  # transmisja tłumika

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction < 1.0 or not 0.0 <= self.factor < 1.0:
            raise ValueError("wymagane 0 < fraction < 1 i 0 ≤ factor < 1")

    def apply(
        self, pulses: Pulses, rng: RandomSource, nature: np.random.Generator
    ) -> tuple[Pulses, npt.NDArray[np.bool_]]:
        """Tłumi wybrane bramki; wybór należy do Boba, przerzedzanie fotonów do przyrody."""
        mask = rng.choice(len(pulses), (1.0 - self.fraction, self.fraction)).astype(np.bool_)
        attenuated = pulses.replace(
            n=np.where(mask, nature.binomial(pulses.n, self.factor), pulses.n),
            power=np.where(mask, pulses.power * self.factor, pulses.power),
        )
        return attenuated, mask


def attenuation_z(counts: AttenuationTally, expected_ratio: float) -> float:
    """z-score: odsetek kliknięć w bramkach tłumionych vs `expected_ratio` × pozostałe."""
    sent_att, det_att = counts.sent_attenuated, counts.detected_attenuated
    sent_rest, det_rest = counts.sent - sent_att, counts.detected - det_att
    if sent_att == 0 or sent_rest == 0:
        raise ValueError("test wymaga bramek tłumionych i nietłumionych")
    rate_rest = det_rest / sent_rest
    expected = expected_ratio * rate_rest
    variance = (
        expected * (1.0 - expected) / sent_att
        + expected_ratio**2 * rate_rest * (1.0 - rate_rest) / sent_rest
    )
    if variance <= 0.0:
        return 0.0 if det_att == 0 else math.inf
    return (det_att / sent_att - expected) / math.sqrt(variance)
