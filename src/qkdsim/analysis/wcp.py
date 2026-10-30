"""Zysk i QBER źródła WCP: model Ma, Qi, Zhao, Lo 2005 i dokładny model detektorów symulatora.

Dwa modele, bo pełnią różne role:

- `gain`, `qber` to model z artykułu (sumy wzorów (7) i (8)); na nim opierają się stopy klucza
  i wartości kontrolne z ROADMAP 3.2;
- `click_probabilities`, `qber_squashed` to dokładna wartość oczekiwana dla modelu Monte Carlo
  z detector.py (dwa detektory progowe, niezależne ciemne zliczenia, podwójne kliknięcia
  z losowym bitem). Z nią porównuje się wyniki MC w bramce G1.

Oba modele różnią się tylko traktowaniem podwójnych kliknięć; różnica jest rzędu (ημ)²·e_d
(test w tests/unit/test_analysis.py).
"""

import math
from dataclasses import dataclass


def _one_minus_exp(x: float) -> float:
    """1 − e^(−x) bez utraty precyzji dla małych x (np. ημ ≈ 10⁻¹⁶ daje inaczej 10⁻⁴ błędu)."""
    return -math.expm1(-x)


def gain(mu: float, eta: float, background: float) -> float:
    """Q_μ = Σ Y_n P(n) = 1 − (1 − Y₀) e^(−ημ) (Ma et al., wzór (7) zsumowany dokładnie)."""
    return background + (1.0 - background) * _one_minus_exp(eta * mu)


def gain_eq10(mu: float, eta: float, background: float) -> float:
    """Q_μ ≈ Y₀ + 1 − e^(−ημ): przybliżenie z wzoru (10), użyte w planie projektu."""
    return background + _one_minus_exp(eta * mu)


def qber(mu: float, eta: float, background: float, misalignment: float, e0: float = 0.5) -> float:
    """E_μ = [e₀Y₀ + e_d(1 − e^(−ημ))] / Q_μ (Ma et al., wzory (8) i (11))."""
    errors = e0 * background + misalignment * _one_minus_exp(eta * mu)
    return errors / gain(mu, eta, background)


@dataclass(frozen=True)
class ClickProbabilities:
    """P-stwa wyników pomiaru jednego impulsu w zgodnej bazie."""

    right_only: float  # kliknął tylko detektor zgodny z bitem Alice
    wrong_only: float  # kliknął tylko drugi detektor (błąd)
    double: float  # kliknęły oba; bit losowy, błąd z p-stwem 1/2
    none: float

    @property
    def detected(self) -> float:
        # Suma małych składników zamiast 1 − none: stabilne, gdy detekcje są rzadkie.
        return self.right_only + self.wrong_only + self.double

    @property
    def qber(self) -> float:
        return (self.wrong_only + 0.5 * self.double) / self.detected


def click_probabilities(
    mu: float, eta: float, dark_count: float, fraction_right: float
) -> ClickProbabilities:
    """Dokładny model detektorów symulatora dla stanu koherentnego o średniej μ.

    Stan koherentny rozdziela się na dwa niezależne stany koherentne, więc liczby wykrytych
    fotonów w detektorach to niezależne zmienne Poissona o średnich ημ·f i ημ·(1 − f),
    gdzie f = `fraction_right`. Ciemne zliczenia są niezależne z p-stwem `dark_count`.
    """
    silent = 1.0 - dark_count
    click_right = dark_count + silent * _one_minus_exp(eta * mu * fraction_right)
    click_wrong = dark_count + silent * _one_minus_exp(eta * mu * (1.0 - fraction_right))
    return ClickProbabilities(
        right_only=click_right * (1.0 - click_wrong),
        wrong_only=click_wrong * (1.0 - click_right),
        double=click_right * click_wrong,
        none=(1.0 - click_right) * (1.0 - click_wrong),
    )


def matched_basis_clicks(
    mu: float, eta: float, dark_count: float, misalignment: float
) -> ClickProbabilities:
    """Zgodna baza: do złego detektora trafia ułamek e_d fotonów."""
    return click_probabilities(mu, eta, dark_count, 1.0 - misalignment)


def mismatched_basis_clicks(
    mu: float, eta: float, dark_count: float, misalignment: float
) -> ClickProbabilities:
    """Niezgodna baza: obrót o θ_e daje podział (1 ± sin 2θ_e)/2, sin 2θ_e = 2√(e_d(1 − e_d))."""
    return click_probabilities(
        mu, eta, dark_count, 0.5 - math.sqrt(misalignment * (1.0 - misalignment))
    )


def qber_squashed(mu: float, eta: float, dark_count: float, misalignment: float) -> float:
    """QBER modelu Monte Carlo (podwójne kliknięcie = losowy bit)."""
    return matched_basis_clicks(mu, eta, dark_count, misalignment).qber
