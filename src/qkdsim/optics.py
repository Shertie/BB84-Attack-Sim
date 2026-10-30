"""Kąty polaryzacji i reguła Borna (docs/model.md, rozdz. 2 i 4).

Wspólne dla Alice (przygotowanie stanu), kanału (obrót polaryzacji) i Boba (PBS).
"""

import numpy as np
import numpy.typing as npt

from qkdsim.rng import Bits

Angles = npt.NDArray[np.float64]


def polarization_angle(bits: Bits, bases: Bits) -> Angles:
    """Kąt stanu Alice: H = 0, V = π/2, D = π/4, A = 3π/4."""
    return bases.astype(np.float64) * (np.pi / 4) + bits.astype(np.float64) * (np.pi / 2)


def analyzer_angle(bases: Bits) -> Angles:
    """Kąt PBS Boba: 0 dla bazy Z, π/4 dla bazy X."""
    return bases.astype(np.float64) * (np.pi / 4)


def prob_detector0(theta: Angles, phi: Angles) -> Angles:
    """Reguła Borna cos²(θ − φ): p-stwo, że foton trafi do detektora 0 (bit 0).

    Liczone jako (1 + cos 2(θ − φ)) / 2, co daje dokładnie 0 i 1 dla stanów z bazy analizatora.
    Wprost liczone cos² zostawiłoby resztę rzędu 10⁻³³ zamiast zera.
    """
    return 0.5 * (1.0 + np.cos(2.0 * (theta - phi)))
