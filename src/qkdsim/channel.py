"""Światłowód: straty i błąd optyki (F1.6)."""

import math
from dataclasses import dataclass
from typing import Self

import numpy as np

from qkdsim.pulses import Pulses


@dataclass(frozen=True)
class FiberChannel:
    """Każdy foton przechodzi z p-stwem 10^(−αL/10); polaryzacja obraca się o stały kąt."""

    length_km: float
    alpha_db_per_km: float = 0.21
    misalignment: float = 0.0  # e_d: p-stwo, że foton trafi do złego detektora w zgodnej bazie

    def __post_init__(self) -> None:
        if self.length_km < 0.0:
            raise ValueError(f"length_km musi być nieujemne, jest {self.length_km}")
        if self.alpha_db_per_km < 0.0:
            raise ValueError(f"alpha_db_per_km musi być nieujemne, jest {self.alpha_db_per_km}")
        if not 0.0 <= self.misalignment <= 0.5:
            raise ValueError(f"misalignment musi być w [0, 0.5], jest {self.misalignment}")

    @property
    def transmittance(self) -> float:
        return float(10.0 ** (-self.alpha_db_per_km * self.length_km / 10.0))

    @property
    def rotation(self) -> float:
        """Kąt θ_e z sin²θ_e = e_d [rad]."""
        return math.asin(math.sqrt(self.misalignment))

    def transmit(self, pulses: Pulses, nature: np.random.Generator) -> Pulses:
        return pulses.replace(
            n=nature.binomial(pulses.n, self.transmittance),
            theta=pulses.theta + self.rotation,
        )

    def split(self, position: float, lossless_after: bool = False) -> tuple[Self, Self]:
        """Dwa odcinki: do Ewy (bez błędu optyki) i od Ewy do Boba (z błędem optyki).

        Złożenie obu odcinków ma tę samą transmisję co cały kanał, bo przerzedzanie
        dwumianowe się składa: Binomial(Binomial(n, T₁), T₂) ~ Binomial(n, T₁T₂).
        """
        if not 0.0 <= position <= 1.0:
            raise ValueError(f"position musi być w [0, 1], jest {position}")
        before = type(self)(self.length_km * position, self.alpha_db_per_km, 0.0)
        after = type(self)(
            self.length_km * (1.0 - position),
            0.0 if lossless_after else self.alpha_db_per_km,
            self.misalignment,
        )
        return before, after
