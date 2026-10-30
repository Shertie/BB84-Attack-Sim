"""Krzywa ROC dla obrony z progiem: odsetek wykrytych ataków vs odsetek fałszywych alarmów."""

import numpy as np
import numpy.typing as npt

Scores = npt.NDArray[np.float64]


def roc_curve(honest: Scores, attacked: Scores) -> tuple[Scores, Scores]:
    """(FPR, TPR) dla wszystkich progów; większy wynik = bardziej podejrzany."""
    thresholds = np.unique(np.concatenate([honest, attacked, [np.inf]]))[::-1]
    fpr = np.array([np.mean(honest >= t) for t in thresholds])
    tpr = np.array([np.mean(attacked >= t) for t in thresholds])
    return fpr, tpr


def auc(honest: Scores, attacked: Scores) -> float:
    """Pole pod krzywą ROC = P(wynik ataku > wynik uczciwy), remisy liczone po połowie."""
    greater = (attacked[:, None] > honest[None, :]).mean()
    ties = (attacked[:, None] == honest[None, :]).mean()
    return float(greater + 0.5 * ties)
