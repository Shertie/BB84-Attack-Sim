"""Model analityczny ataku oślepiania.

Dokładna wartość oczekiwana dla symulatora (attacks/blinding.py + detector.py), na impuls
i w zgodnej bazie Alice–Boba. Odpowiedź pojedynczego detektora bierze się z
`DetectorPair.silent_probability`, więc model sprawdza złożenie przypadków i losowanie w MC,
a odpowiedź detektora mają osobne testy jednostkowe.

Przypadki (M = 1 − e^(−μT₁): Ewa ma wynik, s: ułamek bramek z impulsem wyzwalającym):
- A, M·s/2: baza Ewy = baza Boba = baza Alice; detektor bitu Ewy dostaje (1 − ε)P, drugi ε·P,
  gdzie ε = sin² resztkowego obrotu (θ_e minus kompensacja Ewy);
- B, M·s/2: baza Ewy ≠ baza Boba; podział mocy ½ ± √(ε(1 − ε)), bit Boba losowy względem Alice;
- C, 1 − M·s: brak impulsu wyzwalającego; oślepiony detektor milczy, nieoślepiony ma ciemne
  zliczenia.
"""

import math
from dataclasses import dataclass

import numpy as np

from qkdsim.attacks.blinding import DetectorBlinding
from qkdsim.detector import DetectorPair


@dataclass(frozen=True)
class BlindingModel:
    """P-stwa na impuls przy oślepianiu: detekcja, błąd i detekcja znana Ewie."""

    detected: float  # p-stwo detekcji na impuls (równe w zgodnej i niezgodnej bazie)
    errors: float  # p-stwo detekcji z błędem w zgodnej bazie
    known: float  # p-stwo detekcji, której bit Ewa zna na pewno

    @property
    def qber(self) -> float:
        return self.errors / self.detected

    @property
    def known_fraction(self) -> float:
        return self.known / self.detected


def blinding_model(
    detectors: DetectorPair,
    attack: DetectorBlinding,
    mu: float,
    misalignment: float,
    transmittance_to_eve: float = 1.0,
) -> BlindingModel:
    """Wartość oczekiwana ataku 2 dla symulatora (przypadki A–C z opisu modułu)."""
    cw = np.array([attack.blind_power_w])
    blinded = [detectors.blinded(cw, i) for i in (0, 1)]

    def click(detector: int, power: float) -> float:
        silent = detectors.silent_probability(
            np.zeros(1, dtype=np.int64), np.array([power]), blinded[detector], detector
        )
        return 1.0 - float(silent[0])

    # Resztkowy obrót impulsów Ewy u Boba: błąd optyki minus kompensacja Ewy.
    residual = math.asin(math.sqrt(misalignment)) - attack.compensation_rad
    p, eps, m = attack.trigger_power_w, math.sin(residual) ** 2, attack.mimic_qber
    measured = -math.expm1(-mu * transmittance_to_eve)
    sent = measured * attack.send_fraction

    # A: bit Ewy r ∈ {0, 1} równie prawdopodobny; detektor r dostaje (1 − e_d)P.
    det_a = err_a = 0.0
    for r in (0, 1):
        c_right, c_other = click(r, (1.0 - eps) * p), click(1 - r, eps * p)
        both = c_right * c_other
        to_r = c_right * (1.0 - c_other) + 0.5 * both
        to_other = c_other * (1.0 - c_right) + 0.5 * both
        det_a += 0.5 * (1.0 - (1.0 - c_right) * (1.0 - c_other))
        err_a += 0.5 * ((1.0 - m) * to_other + m * to_r)

    # B: detektor 0 dostaje ułamek ½ − k albo ½ + k mocy, równie prawdopodobnie.
    k = math.sqrt(eps * (1.0 - eps))
    det_b = 0.0
    for share in (0.5 - k, 0.5 + k):
        c0, c1 = click(0, share * p), click(1, (1.0 - share) * p)
        det_b += 0.5 * (1.0 - (1.0 - c0) * (1.0 - c1))

    # C: brak impulsu wyzwalającego.
    c0, c1 = click(0, 0.0), click(1, 0.0)
    det_c = 1.0 - (1.0 - c0) * (1.0 - c1)

    detected = 0.5 * sent * det_a + 0.5 * sent * det_b + (1.0 - sent) * det_c
    errors = 0.5 * sent * err_a + 0.5 * sent * 0.5 * det_b + (1.0 - sent) * 0.5 * det_c
    known = 0.5 * sent * det_a + 0.5 * measured * (1.0 - attack.send_fraction) * det_c
    return BlindingModel(detected, errors, known)
