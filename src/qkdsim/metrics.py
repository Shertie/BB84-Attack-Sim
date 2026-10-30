"""Metryki przebiegu (F1.11).

Warstwa bitów: QBER i udział bitów po siftingu z `SiftedKey`.
Warstwa statystyk: estymaty z liczników `Counts` z odchyleniem standardowym i z-score
względem wartości oczekiwanej (rozdz. 7.2 ROADMAP: porównanie MC ze wzorem przez z-score).
"""

import math
from dataclasses import dataclass

import numpy as np

from qkdsim.pipeline import Counts, Table
from qkdsim.protocol import SiftedKey


def qber(key: SiftedKey) -> float:
    """Odsetek bitów po siftingu, w których Alice i Bob się różnią."""
    if len(key) == 0:
        raise ValueError("QBER z pustego klucza po siftingu jest nieokreślony")
    return float(np.mean(key.alice != key.bob))


def sifted_fraction(key: SiftedKey, n_pulses: int) -> float:
    """Udział wysłanych impulsów, które przeszły sifting."""
    if n_pulses <= 0:
        raise ValueError("liczba impulsów musi być dodatnia")
    return len(key) / n_pulses


@dataclass(frozen=True)
class Estimate:
    """Odsetek k/n z próby dwumianowej."""

    successes: int
    trials: int

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("estymata wymaga co najmniej jednej próby")
        if not 0 <= self.successes <= self.trials:
            raise ValueError("liczba sukcesów musi być w [0, trials]")

    @property
    def value(self) -> float:
        return self.successes / self.trials

    @property
    def sigma(self) -> float:
        """Odchylenie standardowe estymaty (z wartości zmierzonej)."""
        p = self.value
        return math.sqrt(p * (1.0 - p) / self.trials)

    def z_score(self, expected: float) -> float:
        """(k − np) / √(np(1 − p)) z wariancją z wartości oczekiwanej p."""
        if not 0.0 < expected < 1.0:
            raise ValueError("wartość oczekiwana musi być w (0, 1)")
        n = self.trials
        return (self.successes - n * expected) / math.sqrt(n * expected * (1.0 - expected))


def _matched(table: Table, basis: int | None) -> int:
    if basis is None:
        return int(table[0, 0] + table[1, 1])
    return int(table[basis, basis])


def gain(counts: Counts, intensity: int = 0) -> Estimate:
    """Q: odsetek wysłanych impulsów, które dały kliknięcie (wszystkie bazy)."""
    return Estimate(int(counts.detected[intensity].sum()), int(counts.sent[intensity].sum()))


def matched_qber(counts: Counts, intensity: int = 0, basis: int | None = None) -> Estimate:
    """E: odsetek błędnych bitów wśród wykrytych w zgodnych bazach (opcjonalnie tylko Z lub X)."""
    return Estimate(
        _matched(counts.errors[intensity], basis), _matched(counts.detected[intensity], basis)
    )


def double_click_rate(counts: Counts, intensity: int = 0, matched: bool = True) -> Estimate:
    """Odsetek wysłanych impulsów z podwójnym kliknięciem, w zgodnych albo niezgodnych bazach."""
    double, sent = counts.double[intensity], counts.sent[intensity]
    if matched:
        return Estimate(_matched(double, None), _matched(sent, None))
    return Estimate(int(double[0, 1] + double[1, 0]), int(sent[0, 1] + sent[1, 0]))
