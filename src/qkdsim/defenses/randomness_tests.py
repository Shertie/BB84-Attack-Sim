"""Obrona przed atakiem 8: testy losowości ciągu bitów.

Podzbiór w stylu NIST SP 800-22 (rev. 1a): test częstości (monobit, rozdz. 2.1) i test serii
(runs, rozdz. 2.3, z warunkiem wstępnym |π − ½| < 2/√n), plus test autokorelacji z przesunięciem
d: A = Σ x_i ⊕ x_{i+d}, Z = 2(A − (n − d)/2)/√(n − d). Każdy zwraca p-wartość; ciąg odrzucamy,
gdy p < α (NIST: α = 0,01).

Testy wykrywają obciążenie i korelacje, ale nie przewidywalność: MT19937 o znanym ziarnie
przechodzi je jak prawdziwy generator (e13).
"""

import math

import numpy as np

from qkdsim.rng import Bits


def monobit_pvalue(bits: Bits) -> float:
    """p = erfc(|S_n|/√(2n)), S_n = Σ(2x − 1)."""
    n = len(bits)
    s = 2.0 * float(np.count_nonzero(bits)) - n
    return math.erfc(abs(s) / math.sqrt(2.0 * n))


def runs_pvalue(bits: Bits) -> float:
    """p = erfc(|V − 2nπ(1 − π)| / (2√(2n)·π(1 − π))); 0, gdy nie zachodzi warunek wstępny."""
    n = len(bits)
    pi = float(np.count_nonzero(bits)) / n
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0
    runs = 1 + int(np.count_nonzero(bits[1:] != bits[:-1]))
    expected = 2.0 * n * pi * (1.0 - pi)
    return math.erfc(abs(runs - expected) / (2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)))


def autocorrelation_pvalue(bits: Bits, lag: int = 1) -> float:
    """p = erfc(|Z|/√2) dla Z = 2(A − (n − d)/2)/√(n − d)."""
    pairs = len(bits) - lag
    differences = int(np.count_nonzero(bits[lag:] != bits[:-lag]))
    z = 2.0 * (differences - pairs / 2.0) / math.sqrt(pairs)
    return math.erfc(abs(z) / math.sqrt(2.0))


TESTS = {"monobit": monobit_pvalue, "runs": runs_pvalue, "autokorelacja": autocorrelation_pvalue}
