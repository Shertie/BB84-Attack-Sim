"""Wiedza Ewy i interfejs ataków w kanale kwantowym (ADR 0001, ADR 0003).

Atak w kanale to obiekt z polem `position` (0 = tuż za Alice, 1 = tuż przed Bobem) i metodą
`intercept`, która przekształca impulsy i zapisuje, co Ewa wie, w `EveLog`. Tor dzieli
światłowód w punkcie `position`; błąd optyki (obrót polaryzacji) leży po stronie Boba, za Ewą.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from qkdsim.pulses import Pulses
from qkdsim.rng import Bits

NO_BASIS = np.uint8(255)


@dataclass(frozen=True, eq=False)
class EveLog:
    """Tablice wyrównane indeksem z impulsami.

    `measured`: Ewa zmierzyła impuls (miał co najmniej jeden foton w jej punkcie);
    `basis`, `bit`: baza i wynik jej pomiaru (NO_BASIS i 0, gdy nie mierzyła).
    Po siftingu Ewa zna bazę Alice, więc wie, które swoje bity może uznać za pewne.
    """

    measured: npt.NDArray[np.bool_]
    basis: Bits
    bit: Bits

    @classmethod
    def empty(cls, n: int) -> "EveLog":
        return cls(
            measured=np.zeros(n, dtype=np.bool_),
            basis=np.full(n, NO_BASIS, dtype=np.uint8),
            bit=np.zeros(n, dtype=np.uint8),
        )


class ChannelAttack(Protocol):
    """Atak wpinany w światłowód (punkt P4/P5 z docs/architecture.md)."""

    @property
    def position(self) -> float:
        """Ułamek długości światłowodu między Alice a Ewą."""
        ...

    @property
    def lossless_after(self) -> bool:
        """True, gdy Ewa zastępuje dalszy odcinek kanałem bezstratnym (atak PNS, F3)."""
        ...

    def intercept(self, pulses: Pulses, rng: np.random.Generator) -> tuple[Pulses, EveLog]: ...
