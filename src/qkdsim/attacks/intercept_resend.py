"""Atak 1: intercept-resend (F3.1–F3.2).

Ewa przechwytuje ułamek `fraction` impulsów w punkcie `position` światłowodu, mierzy każdy
w losowej bazie idealnym detektorem i wysyła Bobowi jeden foton w stanie, który zmierzyła.

- Impuls bez fotonów w jej punkcie nie daje wyniku; Ewa nic nie wysyła (Bob widzi próżnię).
- Impuls wielofotonowy w zgodnej bazie daje wynik deterministyczny, a w niezgodnej
  przypadkowy, więc jedno losowanie z reguły Borna jest dokładnym modelem jej detektora
  (przy podwójnym kliknięciu losuje bit).
- Ewa mierzy stan przed obrotem polaryzacji, bo błąd optyki leży po stronie Boba (K14).

Oczekiwany wynik: QBER ≈ e_d + p(¼ − e_d/2) dla Ewy tuż przed Bobem; po siftingu Ewa
zgaduje 75% bitów z impulsów, które zmierzyła, a połowę zna na pewno. Ewa tuż za Alice
zmienia dodatkowo zysk, bo wysyła 1 foton zamiast średnio μ (drugi detektor ataku).
"""

from dataclasses import dataclass

import numpy as np

from qkdsim.eve import NO_BASIS, EveLog
from qkdsim.optics import analyzer_angle, polarization_angle, prob_detector0
from qkdsim.pulses import Pulses


@dataclass(frozen=True)
class InterceptResend:
    fraction: float  # p: ułamek przechwytywanych impulsów
    position: float = 1.0  # 0 = tuż za Alice, 1 = tuż przed Bobem
    lossless_after: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"fraction musi być w [0, 1], jest {self.fraction}")
        if not 0.0 <= self.position <= 1.0:
            raise ValueError(f"position musi być w [0, 1], jest {self.position}")

    def intercept(self, pulses: Pulses, rng: np.random.Generator) -> tuple[Pulses, EveLog]:
        n = len(pulses)
        chosen = rng.random(n) < self.fraction
        measured = chosen & (pulses.n > 0)
        basis = rng.integers(0, 2, size=n, dtype=np.uint8)
        p0 = prob_detector0(pulses.theta, analyzer_angle(basis))
        bit = (rng.random(n) >= p0).astype(np.uint8)
        resent = pulses.replace(
            n=np.where(chosen, measured.astype(np.int64), pulses.n),
            theta=np.where(measured, polarization_angle(bit, basis), pulses.theta),
        )
        log = EveLog(
            measured=measured,
            basis=np.where(measured, basis, NO_BASIS).astype(np.uint8),
            bit=np.where(measured, bit, 0).astype(np.uint8),
        )
        return resent, log
