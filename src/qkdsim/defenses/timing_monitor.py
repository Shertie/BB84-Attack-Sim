"""Obrona przed atakiem 4: monitor czasu przyjścia kliknięć.

Bob znakuje czas każdego kliknięcia z rozmyciem Gaussa o odchyleniu `jitter_ns` i liczy kliknięcia
poza oknem ±`window_ns` wokół nominalnego przyjścia. Bez ataku to ułamek p₀ = 2Φ(−w/σ); Ewa, która
przesuwa impulsy o setki ps, wypycha poza okno większość kliknięć. Liczba kliknięć poza oknem
wśród n to Bin(n, p), więc test i krzywa ROC są jak w qber_test.py.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from qkdsim.defenses.qber_test import error_limit

Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TimingMonitor:
    jitter_ns: float
    window_ns: float

    def outside_probability(self, offset_ns: float) -> float:
        """P(|t + N(0, σ²)| > w) dla impulsu przesuniętego o `offset_ns`."""
        sigma = self.jitter_ns
        return float(
            norm.sf((self.window_ns - offset_ns) / sigma)
            + norm.cdf((-self.window_ns - offset_ns) / sigma)
        )

    def outside(self, arrival_ns: Floats, rng: np.random.Generator) -> int:
        """Liczba kliknięć poza oknem po dodaniu rozmycia detektora."""
        measured = arrival_ns + self.jitter_ns * rng.standard_normal(len(arrival_ns))
        return int(np.count_nonzero(np.abs(measured) > self.window_ns))

    def alarm_limit(self, clicks: int, false_alarm: float = 0.01) -> int:
        """Największa liczba kliknięć poza oknem bez alarmu (uczciwe p₀, fałszywe alarmy ≤ α)."""
        return error_limit(clicks, self.outside_probability(0.0), false_alarm)
