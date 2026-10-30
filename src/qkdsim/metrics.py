"""Metryki przebiegu.

Etap 1: QBER i udział bitów po siftingu. Zyski per intensywność, QBER osobno w bazach Z i X,
podwójne kliknięcia i z-score dochodzą w F1.11.
"""

import numpy as np

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
