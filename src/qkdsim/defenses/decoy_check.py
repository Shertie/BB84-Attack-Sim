"""Metryka wykrycia PNS: spójność zysków sygnału i decoy.

Bez ataku Q_i = 1 − (1 − Y₀)e^(−ημ_i) dla każdej intensywności z tym samym η. Z Q_μ i Y₀
można więc wyliczyć η̂ i przewidzieć Q_ν. Ewa widzi tylko liczbę fotonów, nie intensywność:
blokując impulsy jednofotonowe, uderza mocniej w słaby decoy (u którego prawie każdy
niepusty impuls jest jednofotonowy) niż w sygnał. Statystyka: z = (Q̂_ν − Q_ν przewidziane)/σ,
gdzie σ uwzględnia niepewność Q̂_ν i Q̂_μ.
"""

import math

from qkdsim.metrics import gain
from qkdsim.pipeline import Counts
from qkdsim.postprocessing.decoy import DECOY, SIGNAL, VACUUM


def decoy_consistency_z(counts: Counts, mu: float, nu: float) -> float:
    """z = (Q̂_ν − Q_ν przewidziane z Q_μ i Y₀)/σ; silnie ujemne przy PNS."""
    signal, decoy, vacuum = gain(counts, SIGNAL), gain(counts, DECOY), gain(counts, VACUUM)
    silent = 1.0 - vacuum.value
    if not 0.0 < signal.value < 1.0 or silent <= 0.0:
        raise ValueError("test wymaga zysku sygnału w (0, 1) i tła < 1")
    eta = -math.log((1.0 - signal.value) / silent) / mu
    predicted = 1.0 - silent * math.exp(-eta * nu)
    # dQ_ν/dQ_μ z η̂ = −ln((1 − Q_μ)/(1 − Y₀))/μ.
    slope = silent * nu * math.exp(-eta * nu) / (mu * (1.0 - signal.value))
    variance = (
        predicted * (1.0 - predicted) / decoy.trials
        + slope**2 * signal.value * (1.0 - signal.value) / signal.trials
    )
    return (decoy.value - predicted) / math.sqrt(variance)
