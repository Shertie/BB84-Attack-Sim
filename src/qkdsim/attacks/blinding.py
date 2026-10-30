"""Atak 2: oślepianie detektorów (Lydersen et al. 2010), F3.7.

1. Ewa świeci na detektory Boba ciągłym światłem `blind_power_w`; detektory przechodzą
   w tryb liniowy (detector.py).
2. Tuż za Alice przechwytuje impulsy i mierzy je w losowej bazie idealnym detektorem.
3. Wysyła Bobowi bezstratną linią impuls wyzwalający (faked state) o mocy `trigger_power_w`
   w swojej bazie i ze swoim bitem. Przy zgodnej bazie cała moc trafia w jeden detektor
   i wywołuje kliknięcie z bitem Ewy; przy niezgodnej każdy detektor dostaje P/2, poniżej
   progu, i nic nie klika. Te zdarzenia i tak wypadają przy siftingu.

Dopasowanie zysku: Ewa nie płaci strat kanału, więc wysyła impuls wyzwalający tylko w części
`send_fraction` bramek, w których coś zmierzyła, tak by zysk u Boba był taki jak bez ataku.
`mimic_qber` to odsetek bitów, które Ewa celowo odwraca: QBER = 0 byłby podejrzany, więc
Ewa odtwarza QBER spodziewany przez Boba. Zna przy tym bit Alice, bo wie, co odwróciła.

`compensation_rad`: błąd optyki w modelu to stały obrót polaryzacji o θ_e (sin²θ_e = e_d).
Dla jasnego impulsu w bazie sprzężonej daje on podział mocy ½ ± √(e_d(1 − e_d)) zamiast ½,
czyli 32/68 dla GYS, co przy progach Clavis2 likwiduje okno doskonałego ataku. Skalibrowana
Ewa obraca swoje impulsy o −θ_e i przywraca podział ½ (oraz 100/0 w zgodnej bazie).
"""

import math
from dataclasses import dataclass

import numpy as np

from qkdsim.eve import NO_BASIS, EveLog
from qkdsim.optics import analyzer_angle, polarization_angle, prob_detector0
from qkdsim.pulses import Pulses


@dataclass(frozen=True)
class DetectorBlinding:
    """Atak 2: ciągłe światło oślepia detektory, a jasny impuls je wyzwala."""

    trigger_power_w: float  # moc szczytowa impulsu wyzwalającego na wejściu Boba
    send_fraction: float = 1.0  # w ilu bramkach z wynikiem Ewa wysyła impuls wyzwalający
    blind_power_w: float = 1.08e-3  # ciągłe światło oślepiające (Lydersen: 1,08 mW)
    mimic_qber: float = 0.0
    compensation_rad: float = 0.0  # obrót, który Ewa odejmuje od swoich impulsów
    position: float = 0.0  # Ewa mierzy tuż za Alice...
    lossless_after: bool = True  # ...i wysyła do Boba bezstratną linią

    def __post_init__(self) -> None:
        if self.trigger_power_w < 0.0 or self.blind_power_w < 0.0:
            raise ValueError("moce muszą być nieujemne")
        for name in ("send_fraction", "mimic_qber"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} musi być w [0, 1]")

    def intercept(self, pulses: Pulses, rng: np.random.Generator) -> tuple[Pulses, EveLog]:
        n = len(pulses)
        measured = pulses.n > 0
        basis = rng.integers(0, 2, size=n, dtype=np.uint8)
        p0 = prob_detector0(pulses.theta, analyzer_angle(basis))
        bit = (rng.random(n) >= p0).astype(np.uint8)
        send = measured & (rng.random(n) < self.send_fraction)
        sent_bit = bit ^ (rng.random(n) < self.mimic_qber).astype(np.uint8)
        faked = pulses.replace(
            n=np.zeros(n, dtype=np.int64),
            theta=np.where(
                send,
                polarization_angle(sent_bit, basis) - self.compensation_rad,
                pulses.theta,
            ),
            power=np.where(send, self.trigger_power_w, 0.0),
            cw_power=np.full(n, self.blind_power_w),
        )
        log = EveLog(
            measured=measured,
            basis=np.where(measured, basis, NO_BASIS).astype(np.uint8),
            bit=np.where(measured, bit, 0).astype(np.uint8),
        )
        return faked, log


def matching_send_fraction(target_gain: float, mu: float) -> float:
    """Ułamek bramek z impulsem wyzwalającym, przy którym zysk u Boba = `target_gain`.

    W oknie doskonałego ataku impuls wyzwalający daje kliknięcie tylko przy zgodnej bazie
    Boba i Ewy (p-stwo ½), a Ewa ma wynik z p-stwem 1 − e^(−μ).
    """
    return min(1.0, target_gain / (0.5 * -math.expm1(-mu)))
