"""Obrona przed atakami 1 i 6: próg QBER na losowej próbce (test jednostronny), F5.1 / rozdz. 7.1.

Alice i Bob ujawniają k bitów i przerywają, gdy błędów jest więcej niż próg ustalony z góry tak,
by uczciwy przebieg przerywał się z p-stwem ≤ α. Liczba błędów w próbce to Bin(k, Q), więc
krzywa ROC jest dokładna: dla każdego progu t FPR = P(Bin(k, Q₀) > t), TPR = P(Bin(k, Q₁) > t).
"""

import numpy as np
import numpy.typing as npt
from scipy.stats import binom

Floats = npt.NDArray[np.float64]


def error_limit(sample: int, honest_qber: float, false_alarm: float = 0.01) -> int:
    """Największa liczba błędów bez przerwania: kwantyl 1 − α rozkładu Bin(k, Q₀)."""
    return int(binom.ppf(1.0 - false_alarm, sample, honest_qber))


def tolerance(sample: int, honest_qber: float, false_alarm: float = 0.01) -> float:
    """Próg Q_tol jako ułamek próbki (e07: Q_tol z 1% fałszywych przerwań)."""
    return error_limit(sample, honest_qber, false_alarm) / sample


def detection_probability(
    sample: int, honest_qber: float, attacked_qber: float, false_alarm: float = 0.01
) -> float:
    """P-stwo przerwania pod atakiem przy progu z `error_limit`."""
    return float(binom.sf(error_limit(sample, honest_qber, false_alarm), sample, attacked_qber))


def binomial_roc(trials: int, p_honest: float, p_attacked: float) -> tuple[Floats, Floats]:
    """(FPR, TPR) dla wszystkich progów t = −1, 0, …, n testu „liczba zdarzeń > t”."""
    t = np.arange(-1, trials + 1)
    return binom.sf(t, trials, p_honest), binom.sf(t, trials, p_attacked)


def binomial_auc(trials: int, p_honest: float, p_attacked: float) -> float:
    """Pole pod krzywą ROC: P(X₁ > X₀) + ½P(X₁ = X₀) dla niezależnych X ~ Bin."""
    k = np.arange(trials + 1)
    honest = binom.pmf(k, trials, p_honest)
    attacked = binom.pmf(k, trials, p_attacked)
    below = np.concatenate([[0.0], np.cumsum(honest)[:-1]])  # P(X₀ < k)
    return float(np.sum(attacked * (below + 0.5 * honest)))
