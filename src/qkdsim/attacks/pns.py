"""Atak 3: photon-number splitting (Brassard, Lütkenhaus, Mor, Sanders 2000).

Opis strategii i model analityczny: analysis/pns.py. Ewa nie zmienia polaryzacji, więc QBER
zostaje bez zmian; zabiera po jednym fotonie z impulsów wielofotonowych i po ujawnieniu baz
zna te bity na pewno. Bez stanów decoy GLLP musi zakładać najgorszy przypadek (wszystkie
impulsy wielofotonowe zna Ewa); kto tego nie zrobi i policzy klucz jak dla źródła
jednofotonowego, odda Ewie prawie cały klucz.
"""

from dataclasses import dataclass

import numpy as np

from qkdsim.eve import NO_BASIS, EveLog
from qkdsim.pulses import Pulses


@dataclass(frozen=True)
class PhotonNumberSplitting:
    """Atak 3: Ewa blokuje impulsy jednofotonowe i zatrzymuje foton z reszty."""

    block: float  # p-stwo zablokowania impulsu jednofotonowego
    forward: float = 1.0  # p-stwo przepuszczenia reszty impulsu wielofotonowego
    position: float = 0.0  # Ewa tuż za Alice...
    lossless_after: bool = True  # ...i bezstratna linia do Boba

    def __post_init__(self) -> None:
        for name in ("block", "forward"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} musi być w [0, 1]")

    def intercept(self, pulses: Pulses, rng: np.random.Generator) -> tuple[Pulses, EveLog]:
        n = pulses.n
        single, multi = n == 1, n >= 2
        passed_single = single & (rng.random(len(n)) >= self.block)
        passed_multi = multi & (rng.random(len(n)) < self.forward)
        forwarded = np.where(passed_single, 1, np.where(passed_multi, n - 1, 0)).astype(np.int64)
        # Foton z pamięci Ewa mierzy po ujawnieniu baz, w bazie Alice: zna bit na pewno.
        log = EveLog(
            measured=multi,
            basis=np.where(multi, pulses.basis, NO_BASIS).astype(np.uint8),
            bit=np.where(multi, pulses.bit, 0).astype(np.uint8),
        )
        return pulses.replace(n=forwarded), log
