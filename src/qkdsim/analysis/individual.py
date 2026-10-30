"""Granice informacyjne: atak indywidualny (Fuchs et al. 1997) vs ataki ogólne (F4.9–F4.10).

Przy błędzie D wprowadzanym u Boba:
- I_AB = 1 − h(D);
- I_AE = 1 − h(½ − √(D(1 − D))) dla optymalnego ataku indywidualnego, równoważnie
  ½φ(2√(D(1 − D))), φ(z) = (1 + z)log₂(1 + z) + (1 − z)log₂(1 − z) (zapis artykułu);
- intercept-resend: D = p/4, Ewa zna na pewno p/2 bitów, więc I_AE = 2D (do D = ¼).

I_AB = I_AE przy D* = (1 − 1/√2)/2 ≈ 14,64%; Shor–Preskill (ataki ogólne) daje 11,00%.
"""

import math

from qkdsim.analysis.entropy import h2

FUCHS_CROSSING = (1.0 - 1.0 / math.sqrt(2.0)) / 2.0


def bob_information(error: float) -> float:
    return 1.0 - h2(error)


def eve_error_individual(error: float) -> float:
    """Błąd Ewy w optymalnym ataku indywidualnym: ½ − √(D(1 − D))."""
    return 0.5 - math.sqrt(error * (1.0 - error))


def eve_information_individual(error: float) -> float:
    return 1.0 - h2(eve_error_individual(error))


def eve_information_intercept_resend(error: float) -> float:
    """I_AE = 2D dla intercept-resend na źródle jednofotonowym (D ≤ ¼)."""
    if not 0.0 <= error <= 0.25:
        raise ValueError("intercept-resend daje D ≤ ¼")
    return 2.0 * error
