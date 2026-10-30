"""Model analityczny ataku intercept-resend na źródło WCP (F3.3).

Dokładna wartość oczekiwana dla symulatora (attacks/intercept_resend.py), w zgodnej bazie:

- impuls nieprzechwycony (1 − p): stan koherentny przez cały kanał;
- przechwycony, ale Ewa nie dostała fotonu (p·e^(−μT₁)): Bob widzi tylko ciemne zliczenia;
- przechwycony i zmierzony (p·(1 − e^(−μT₁))): Ewa wysyła jeden foton; w połowie przypadków
  w bazie Alice (błąd e_d u Boba), w połowie w bazie sprzężonej (błąd ½).

T₁ to transmisja do Ewy, T₂ od Ewy do Boba. Uproszczony wynik z planu, QBER = e_d + p(¼ − e_d/2)
(`qber_line`), jest dokładny dla p = 0 i p = 1. Pomiędzy zakłada, że impulsy przechwycone
i nieprzechwycone są wykrywane równie często. Dla WCP tak nie jest: Ewa wysyła 1 foton
z p-stwem 1 − e^(−μT₁), a niezaatakowany impuls daje detekcję z p-stwem ≈ μT₁·ηT₂, więc
QBER(p) leży poniżej prostej (do −0,33 pp przy 25 km dla GYS). Prosta jest dobra, gdy Ewa stoi
tuż przed Bobem, μT₁ ≪ 1 i ciemne zliczenia są pomijalne (około 50 km dla GYS: ±0,07 pp).
"""

import math
from dataclasses import dataclass

from qkdsim.analysis.key_rate import LinkParameters
from qkdsim.analysis.wcp import ClickProbabilities, matched_basis_clicks


def single_photon_clicks(eta: float, dark_count: float, error: float) -> ClickProbabilities:
    """Jeden foton: wykryty z p-stwem η, w złym detektorze z p-stwem `error`; plus ciemne."""
    silent = 1.0 - dark_count
    missed = 1.0 - eta
    return ClickProbabilities(
        right_only=missed * dark_count * silent + eta * (1.0 - error) * silent,
        wrong_only=missed * dark_count * silent + eta * error * silent,
        double=missed * dark_count**2 + eta * dark_count,
        none=missed * silent**2,
    )


def _mix(parts: list[tuple[float, ClickProbabilities]]) -> ClickProbabilities:
    return ClickProbabilities(
        right_only=sum(w * c.right_only for w, c in parts),
        wrong_only=sum(w * c.wrong_only for w, c in parts),
        double=sum(w * c.double for w, c in parts),
        none=sum(w * c.none for w, c in parts),
    )


@dataclass(frozen=True)
class InterceptResendModel:
    """P-stwa na wysłany impuls w zgodnej bazie."""

    clicks: ClickProbabilities
    measured_detected: float  # Ewa zmierzyła impuls i Bob go wykrył
    known_detected: float  # do tego Ewa mierzyła w bazie Alice (zna bit na pewno)

    @property
    def known_fraction(self) -> float:
        """I_AE na bit po siftingu: Ewa zna bit na pewno albo nie wie nic."""
        return self.known_detected / self.clicks.detected

    @property
    def guess_accuracy(self) -> float:
        """Trafność najlepszego zgadywania Ewy na bitach po siftingu."""
        unknown = self.clicks.detected - self.known_detected
        return (self.known_detected + 0.5 * unknown) / self.clicks.detected


def intercept_resend_model(
    p: LinkParameters,
    length_km: float,
    mu: float,
    fraction: float,
    position: float,
    dark_count: float,
) -> InterceptResendModel:
    t_before = 10.0 ** (-p.alpha_db_per_km * length_km * position / 10.0)
    t_after = 10.0 ** (-p.alpha_db_per_km * length_km * (1.0 - position) / 10.0)
    eta_resent = p.eta_bob * t_after
    measured = -math.expm1(-mu * t_before)  # Ewa dostała co najmniej jeden foton
    right_basis = single_photon_clicks(eta_resent, dark_count, p.misalignment)
    wrong_basis = single_photon_clicks(eta_resent, dark_count, 0.5)
    clicks = _mix(
        [
            (
                1.0 - fraction,
                matched_basis_clicks(
                    mu, p.eta_bob * t_before * t_after, dark_count, p.misalignment
                ),
            ),
            (fraction * (1.0 - measured), single_photon_clicks(0.0, dark_count, 0.5)),
            (fraction * measured * 0.5, right_basis),
            (fraction * measured * 0.5, wrong_basis),
        ]
    )
    measured_detected = fraction * measured * right_basis.detected
    return InterceptResendModel(clicks, measured_detected, 0.5 * measured_detected)


def qber_line(fraction: float, misalignment: float) -> float:
    """QBER = e_d + p(¼ − e_d/2): uproszczony wynik z planu (warunki: docstring modułu)."""
    return misalignment + fraction * (0.25 - 0.5 * misalignment)
