"""Model analityczny MDI-QKD (Lo, Curty, Qi 2012), F7: dokładna wartość oczekiwana dla mdi.py.

- `coherent_gains`: zysk Q i QBER E w danej bazie dla stanów koherentnych o średnich μ_A, μ_B
  (po kanałach), średnio po bitach i po fazie względnej (całka liczona metodą trapezów; funkcja
  podcałkowa jest gładka i okresowa, więc zbieżność jest wykładnicza);
- `single_photon_pair`: Y₁₁ i e₁₁ dla pary pojedynczych fotonów: interferencja dwufotonowa,
  gdy dotrą oba (amplituda α_oβ_o' + α_o'β_o, dla tego samego trybu √2·α_oβ_o), jeden foton,
  gdy dotrze jeden, i niezależne ciemne zliczenia. Sprawność detektorów, równa dla czterech,
  jest wliczona w transmisję wejść, bo jednorodne straty komutują z optyką liniową;
- `key_rate`: wzór (1) artykułu z Q¹¹ = μ²e^(−2μ)Y¹¹ w bazie prostej i e¹¹ w diagonalnej
  (nieskończenie wiele stanów decoy, granica asymetrycznego wyboru baz).
"""

import itertools
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from qkdsim.analysis.entropy import h2
from qkdsim.mdi import FAIL, PSI_MINUS, PSI_PLUS, Relay, bell_outcome, detector_means

BITS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _expected_error(basis: int, same_bits: bool, outcome: int) -> bool:
    """Tabela 1: w bazie prostej Bob zawsze odwraca; w diagonalnej poza |ψ⁺⟩."""
    flipped = not (basis == 1 and outcome == PSI_PLUS)
    return same_bits == flipped  # po odwróceniu bity Boba i Alice mają być równe


def _outcome_probabilities(click: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(|ψ⁻⟩), P(|ψ⁺⟩) z niezależnych p-stw kliknięć czterech detektorów (wiersze)."""
    silent = 1.0 - click
    h1, v1, h2_, v2 = click
    sh1, sv1, sh2, sv2 = silent
    minus = h1 * v2 * sv1 * sh2 + v1 * h2_ * sh1 * sv2
    plus = h1 * v1 * sh2 * sv2 + h2_ * v2 * sh1 * sv1
    return minus, plus


def coherent_gains(
    mu_a: float, mu_b: float, relay: Relay, basis: int, phases: int = 64
) -> tuple[float, float]:
    """(Q, E) na parę impulsów w bazie `basis` (0 = prosta, 1 = diagonalna); μ po kanałach."""
    phi = 2.0 * np.pi * np.arange(phases) / phases
    gain = errors = 0.0
    for bit_a, bit_b in BITS:
        theta_a = basis * math.pi / 4 + bit_a * math.pi / 2
        theta_b = basis * math.pi / 4 + bit_b * math.pi / 2
        ones = np.ones(phases)
        means = detector_means(mu_a * ones, mu_b * ones, theta_a * ones, theta_b * ones, phi, relay)
        click = 1.0 - (1.0 - relay.dark_count) * np.exp(-means)
        minus, plus = (float(np.mean(p)) for p in _outcome_probabilities(click))
        gain += 0.25 * (minus + plus)
        for outcome, prob in ((PSI_MINUS, minus), (PSI_PLUS, plus)):
            if _expected_error(basis, bit_a == bit_b, outcome):
                errors += 0.25 * prob
    return gain, errors / gain


def _photon_modes(theta: float, sign: float, relay: Relay, rotate_input: bool) -> np.ndarray:
    """Amplituda jednego fotonu w trybach 1H, 1V, 2H, 2V po dzielniku i obrocie wyjścia 1."""
    angle = theta + (relay.rotation_in if rotate_input else 0.0)
    h, v = math.cos(angle) / math.sqrt(2.0), math.sin(angle) / math.sqrt(2.0)
    c, s = math.cos(relay.rotation_out), math.sin(relay.rotation_out)
    return np.array([c * h - s * v, s * h + c * v, sign * h, sign * v])


def _click_patterns(occupied: frozenset[int], dark: float) -> list[tuple[np.ndarray, float]]:
    """Wszystkie zbiory kliknięć ⊇ `occupied` i ich p-stwa (ciemne zliczenia w pustych trybach)."""
    empty = [o for o in range(4) if o not in occupied]
    patterns = []
    for extra in itertools.product((False, True), repeat=len(empty)):
        clicks = np.zeros(4, dtype=bool)
        clicks[list(occupied)] = True
        prob = 1.0
        for mode, fires in zip(empty, extra, strict=True):
            clicks[mode] = fires
            prob *= dark if fires else 1.0 - dark
        patterns.append((clicks, prob))
    return patterns


def single_photon_pair(t_a: float, t_b: float, relay: Relay, basis: int) -> tuple[float, float]:
    """(Y₁₁, e₁₁): p-stwo udanego pomiaru i błędu, gdy Alice i Bob wysłali po jednym fotonie.

    `t_*`: transmisja kanału; sprawność detektorów przekaźnika dochodzi tutaj.
    """
    arrive_a, arrive_b = t_a * relay.efficiency, t_b * relay.efficiency
    gain = errors = 0.0
    for bit_a, bit_b in BITS:
        alpha = _photon_modes(basis * math.pi / 4 + bit_a * math.pi / 2, 1.0, relay, False)
        beta = _photon_modes(basis * math.pi / 4 + bit_b * math.pi / 2, -1.0, relay, True)
        cases: list[tuple[float, dict[frozenset[int], float]]] = [
            ((1 - arrive_a) * (1 - arrive_b), {frozenset(): 1.0}),
            (arrive_a * (1 - arrive_b), {frozenset({o}): alpha[o] ** 2 for o in range(4)}),
            ((1 - arrive_a) * arrive_b, {frozenset({o}): beta[o] ** 2 for o in range(4)}),
        ]
        both: dict[frozenset[int], float] = {}
        for o, o2 in itertools.combinations_with_replacement(range(4), 2):
            amp = alpha[o] * beta[o2] + alpha[o2] * beta[o] if o != o2 else alpha[o] * beta[o]
            both[frozenset({o, o2})] = amp**2 if o != o2 else 2.0 * amp**2
        cases.append((arrive_a * arrive_b, both))
        for weight, photons in cases:
            for occupied, p_photons in photons.items():
                for clicks, p_dark in _click_patterns(occupied, relay.dark_count):
                    outcome = int(bell_outcome(clicks[:, None])[0])
                    if outcome == FAIL:
                        continue
                    prob = 0.25 * weight * p_photons * p_dark
                    gain += prob
                    if _expected_error(basis, bit_a == bit_b, outcome):
                        errors += prob
    return gain, errors / gain


def key_rate(
    mu: float, length_km: float, relay: Relay, f_ec: float, alpha_db_per_km: float = 0.2
) -> float:
    """Wzór (1): R = Q¹¹_Z[1 − h(e¹¹_X)] − Q_Z f h(E_Z) na parę impulsów, przekaźnik w połowie,
    μ_A = μ_B = μ."""
    t = 10.0 ** (-alpha_db_per_km * length_km / 20.0)
    q_z, e_z = coherent_gains(mu * t, mu * t, relay, 0)
    y11, _ = single_photon_pair(t, t, relay, 0)
    _, e11_x = single_photon_pair(t, t, relay, 1)
    q11 = mu * mu * math.exp(-2.0 * mu) * y11
    return q11 * (1.0 - h2(min(e11_x, 0.5))) - q_z * f_ec * h2(min(e_z, 0.5))


def optimal_rate(
    length_km: float, relay: Relay, f_ec: float, alpha_db_per_km: float = 0.2
) -> tuple[float, float]:
    """(R, μ) przy najlepszym μ_A = μ_B."""
    result = minimize_scalar(
        lambda mu: -key_rate(mu, length_km, relay, f_ec, alpha_db_per_km),
        bounds=(0.005, 1.5),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return -float(result.fun), float(result.x)


def max_distance(
    relay: Relay, f_ec: float, alpha_db_per_km: float = 0.2, hi_km: float = 400.0
) -> float:
    """Największa odległość Alice–Bob z R > 0 (bisekcja po optymalnej stopie)."""

    def rate(length: float) -> float:
        return optimal_rate(length, relay, f_ec, alpha_db_per_km)[0]

    return float(brentq(rate, 1.0, hi_km))
