"""Testy zdrowia źródła entropii z NIST SP 800-90B (rozdz. 4.4): obrona generatora QRNG (F7).

- Repetition Count Test (4.4.1): alarm, gdy ta sama wartość pojawia się C razy z rzędu,
  C = 1 + ⌈−log₂ α / H⌉;
- Adaptive Proportion Test (4.4.2): w kolejnych oknach W = 1024 próbek (dane binarne) alarm,
  gdy pierwsza wartość okna występuje w nim co najmniej C razy, C = 1 + CRITBINOM(W, 2^(−H), 1 − α).
  Progi zgadzają się z Tabelą 2 normy (test jednostkowy).

H to min-entropia na próbkę z kalibracji, α = 2⁻²⁰ jak w normie. Testy wykrywają awarię źródła
(oślepiony detektor, dryf sprawności), a nie przewidywalność.
"""

import math

import numpy as np
import numpy.typing as npt
from scipy.stats import binom

from qkdsim.rng import Bits

FALSE_ALARM = 2.0**-20
WINDOW = 1024


def repetition_cutoff(entropy: float, false_alarm: float = FALSE_ALARM) -> int:
    """C = 1 + ⌈−log₂ α / H⌉."""
    return 1 + math.ceil(-math.log2(false_alarm) / entropy)


def adaptive_cutoff(entropy: float, window: int = WINDOW, false_alarm: float = FALSE_ALARM) -> int:
    """C = 1 + CRITBINOM(W, 2^(−H), 1 − α): najmniejsze k z P(B ≤ k) ≥ 1 − α, plus 1."""
    return 1 + int(binom.ppf(1.0 - false_alarm, window, 2.0**-entropy))


def first_repetition_alarm(bits: Bits, cutoff: int) -> int | None:
    """Indeks próbki, przy której RCT sygnalizuje awarię (seria osiąga C), albo None."""
    if len(bits) == 0:
        return None
    starts = np.flatnonzero(np.r_[True, bits[1:] != bits[:-1]])
    lengths = np.diff(np.r_[starts, len(bits)])
    long_runs = np.flatnonzero(lengths >= cutoff)
    return int(starts[long_runs[0]] + cutoff - 1) if len(long_runs) else None


def adaptive_alarms(bits: Bits, cutoff: int, window: int = WINDOW) -> npt.NDArray[np.bool_]:
    """Alarm APT dla każdego pełnego okna: liczba wystąpień pierwszej wartości okna ≥ C."""
    windows = bits[: len(bits) // window * window].reshape(-1, window)
    same: npt.NDArray[np.intp] = np.count_nonzero(windows == windows[:, :1], axis=1)
    return same >= cutoff


def adaptive_detection_probability(p_one: float, cutoff: int, window: int = WINDOW) -> float:
    """P(alarm w oknie) dla niezależnych bitów z P(1) = p: pierwsza próbka plus C − 1 z W − 1."""
    return float(
        p_one * binom.sf(cutoff - 2, window - 1, p_one)
        + (1.0 - p_one) * binom.sf(cutoff - 2, window - 1, 1.0 - p_one)
    )
