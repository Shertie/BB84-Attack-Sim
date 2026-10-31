"""Model analityczny ataku Trojan horse (Gisin et al. 2006).

Ewa wpuszcza do modułu Alice jasny impuls sondujący. Odbija się on od modulatora polaryzacji
i wraca do Ewy z polaryzacją ustawioną tak jak stan Alice (najgorszy przypadek: modulator
nakłada swoje ustawienie na sondę tak samo jak na sygnał). Po drodze tłumią go izolator
(raz, w kierunku zaporowym), tłumik Alice (dwa razy) i samo odbicie, więc wraca słaby stan
koherentny o średniej μ_out = μ_in·10^(−A/10). Sygnału Ewa nie dotyka, więc QBER
i zysk się nie zmieniają.

Wiedza Ewy (Helstrom dotyczy dwóch hipotez, więc bit i bazę liczymy osobno):
- bit przy znanej bazie (Ewa trzyma sondę do ujawnienia baz): dwa stany czyste |α_θ⟩
  i |α_{θ+π/2}⟩, Helstrom P = ½(1 + √(1 − e^(−2μ))), informacja 1 − h(P) (wzór (1) artykułu);
- z losową fazą (Alice losuje fazę światła, które może się odbić): Ewa ma mieszaninę stanów
  Focka i zna bit, gdy wróci choć jeden foton: I = 1 − e^(−μ) (wzór (7));
- baza w czasie rzeczywistym (pozwala na intercept-resend bez błędów): Helstrom dla stanów
  mieszanych ρ_Z = ½(ρ_H + ρ_V) i ρ_X = ½(ρ_D + ρ_A) w obciętej przestrzeni Focka dwóch modów
  polaryzacji. Weryfikacja w QuTiP: tests/crosscheck/test_trojan_qutip.py;
- cały stan (bit i baza naraz, stała faza): cztery stany czyste |α_θ⟩, θ = 0, π/4, π/2, 3π/4,
  o macierzy Grama G_jk = exp(−μ(1 − cos(θ_j − θ_k))). Stany są liniowo niezależne, więc
  optymalny pomiar jest rzutowy w ich rozpięciu (Kennedy 1973); szukamy go numerycznie po
  bazach ortonormalnych, startując z pomiaru pierwiastkowego, a optymalność potwierdzają
  warunki Holevo–Yuena–Kennedy’ego–Laxa (`state_guess_certificate`).

Granica sondy (Lucamarini i in. 2015): Ewa nie wpuści więcej niż N fotonów na sekundę bez
uszkodzenia światłowodu (próg LIDT, N = 10²⁰), więc na jedno ustawienie modulatora o częstości
f_A przypada najwyżej μ_in = N/f_A fotonów (ich wzór (4): μ_out = Nγ/f_A).
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm
from scipy.optimize import brentq, minimize

from qkdsim.analysis.entropy import h2

Floats = npt.NDArray[np.float64]

# Gisin et al. 2006, rozdz. II: odbicie wsteczne typowych elementów [dB tłumienia].
GISIN_REFLECTION_DB = {"łącznik ze skośnym szlifem": 40.0, "modulator fazy": 20.0, "lustro": 1.0}
BASIS_ANGLES = {"Z": (0.0, math.pi / 2), "X": (math.pi / 4, 3 * math.pi / 4)}
# Lucamarini et al. 2015, rozdz. III.B: próg uszkodzenia (LIDT) w fotonach na sekundę.
LIDT_PHOTONS_PER_S = 1e20


def round_trip_attenuation_db(
    isolation_db: float, reflection_db: float, attenuator_db: float = 0.0
) -> float:
    """A: izolator raz (w kierunku zaporowym), odbicie raz, tłumik Alice tam i z powrotem."""
    return isolation_db + reflection_db + 2.0 * attenuator_db


def lidt_probe_photons(clock_hz: float, lidt_photons_per_s: float = LIDT_PHOTONS_PER_S) -> float:
    """μ_in = N/f_A: najwięcej fotonów sondy na ustawienie modulatora poniżej progu LIDT."""
    return lidt_photons_per_s / clock_hz


def returned_photons(probe_photons: float, attenuation_db: float) -> float:
    """μ_out = μ_in·10^(−A/10)."""
    return float(probe_photons * 10.0 ** (-attenuation_db / 10.0))


def bit_guess_probability(mu: float, phase_randomized: bool = False) -> float:
    """P-stwo, że Ewa trafi bit przy znanej bazie (pomiar Helstroma albo licznik fotonów)."""
    if phase_randomized:
        return 1.0 - 0.5 * math.exp(-mu)  # bez fotonu zgaduje, z fotonem wie na pewno
    return 0.5 * (1.0 + math.sqrt(-math.expm1(-2.0 * mu)))


def bit_information(mu: float, phase_randomized: bool = False) -> float:
    """Informacja Ewy o bicie na impuls: wzór (1) albo (7) Gisina et al. 2006."""
    if phase_randomized:
        return -math.expm1(-mu)
    return 1.0 - h2(bit_guess_probability(mu))


def _photon_cutoff(mu: float) -> int:
    """Liczba fotonów, powyżej której ogon rozkładu Poissona jest pomijalny (< 10⁻¹²)."""
    return max(12, math.ceil(mu + 12.0 * math.sqrt(mu) + 12.0))


def fock_states(theta: float, n_max: int) -> Floats:
    """Wiersz n: |n_θ⟩ = (cos θ a_H† + sin θ a_V†)ⁿ/√n! |0⟩ w bazie |k, n − k⟩, k = 0..n.

    Stany o różnym n leżą w osobnych blokach; indeks bloku n zaczyna się od n(n + 1)/2.
    """
    dim = (n_max + 1) * (n_max + 2) // 2
    states = np.zeros((n_max + 1, dim))
    c, s = math.cos(theta), math.sin(theta)
    for n in range(n_max + 1):
        start = n * (n + 1) // 2
        for k in range(n + 1):
            states[n, start + k] = math.sqrt(math.comb(n, k)) * c**k * s ** (n - k)
    return states


def probe_density(mu: float, theta: float, phase_randomized: bool, n_max: int) -> Floats:
    """Sonda o polaryzacji θ: stan koherentny albo (losowa faza) mieszanina Poissona |n_θ⟩."""
    n = np.arange(n_max + 1)
    log_weights = -mu + n * math.log(mu) - np.array([math.lgamma(k + 1) for k in n])
    states = fock_states(theta, n_max)
    if phase_randomized:
        return np.asarray((states.T * np.exp(log_weights)) @ states)
    vector = np.exp(0.5 * log_weights) @ states
    return np.outer(vector, vector)


def helstrom_probability(rho: Floats, sigma: Floats) -> float:
    """½ + ¼‖ρ − σ‖₁: najlepsze p-stwo rozróżnienia dwóch stanów o równych p-stwach a priori."""
    return 0.5 + 0.25 * float(np.abs(np.linalg.eigvalsh(rho - sigma)).sum())


def basis_guess_probability(mu: float, phase_randomized: bool = False) -> float:
    """Helstrom dla baz Z i X: ρ_Z = ½(ρ_H + ρ_V), ρ_X = ½(ρ_D + ρ_A)."""
    if mu <= 0.0:
        return 0.5
    n_max = _photon_cutoff(mu)
    rho = {
        name: 0.5
        * (
            probe_density(mu, angles[0], phase_randomized, n_max)
            + probe_density(mu, angles[1], phase_randomized, n_max)
        )
        for name, angles in BASIS_ANGLES.items()
    }
    return helstrom_probability(rho["Z"], rho["X"])


@dataclass(frozen=True)
class StateGuess:
    """Optymalny pomiar czterech stanów sondy: trafność i p-stwa wyników."""

    probability: float  # P(wynik = stan), stany równie prawdopodobne
    confusion: Floats  # confusion[j, k] = P(wynik k | stan j), kolejność H, D, V, A
    states: Floats  # kolumny: stany we współrzędnych ortonormalnych rozpięcia
    basis: Floats  # kolumny: wektory pomiaru w tych samych współrzędnych


PROBE_ANGLES = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)  # H, D, V, A


def _rotation(params: Floats, size: int) -> Floats:
    skew = np.zeros((size, size))
    skew[np.triu_indices(size, 1)] = params
    result: Floats = expm(skew - skew.T)
    return result


def state_guess(mu: float) -> StateGuess:
    """Najlepsze rozpoznanie jednego z czterech stanów BB84 sondy (stała faza)."""
    angles = np.asarray(PROBE_ANGLES)
    gram = np.exp(-mu * (1.0 - np.cos(angles[:, None] - angles[None, :])))
    values, vectors = np.linalg.eigh(gram)
    states: Floats = vectors @ np.diag(np.sqrt(np.clip(values, 0.0, None))) @ vectors.T
    size = len(angles)

    def failure(params: Floats) -> float:
        amplitudes = _rotation(params, size).T @ states
        return 1.0 - float(np.mean(np.diag(amplitudes) ** 2))

    start = np.zeros(size * (size - 1) // 2)  # pomiar pierwiastkowy
    best = minimize(failure, start, method="BFGS", options={"gtol": 1e-12})
    basis = _rotation(best.x, size)
    amplitudes = basis.T @ states
    return StateGuess(1.0 - float(best.fun), (amplitudes**2).T, states, basis)


def state_guess_certificate(guess: StateGuess) -> float:
    """Najmniejsza wartość własna Γ − ¼|ψ_j⟩⟨ψ_j| (≥ 0 dla pomiaru optymalnego).

    Γ = ¼ Σ_j |ψ_j⟩⟨ψ_j|Π_j; pomiar jest optymalny wtedy i tylko wtedy, gdy Γ jest hermitowska
    i Γ − ¼|ψ_j⟩⟨ψ_j| ⪰ 0 dla każdego j. Zwraca też −‖Γ − Γᵀ‖, jeśli to mniejsze.
    """
    psi, w = guess.states, guess.basis
    size = psi.shape[1]
    gamma = sum(np.outer(psi[:, j], psi[:, j]) @ np.outer(w[:, j], w[:, j]) for j in range(size))
    gamma = gamma / size
    asymmetry = float(np.abs(gamma - gamma.T).max())
    symmetric = 0.5 * (gamma + gamma.T)
    smallest = min(
        float(np.linalg.eigvalsh(symmetric - np.outer(psi[:, j], psi[:, j]) / size)[0])
        for j in range(size)
    )
    return min(smallest, -asymmetry)


def resend_qber(guess: StateGuess, misalignment: float = 0.0) -> float:
    """QBER, gdy Ewa w każdym impulsie wysyła Bobowi rozpoznany stan (zgodne bazy).

    Stan trafiony: błąd optyki e_d; ta sama baza, inny bit: 1 − e_d; druga baza: ½.
    """
    errors = 0.0
    for j in range(4):
        for k in range(4):
            if j == k:
                error = misalignment
            elif j % 2 == k % 2:
                error = 1.0 - misalignment
            else:
                error = 0.5
            errors += guess.confusion[j, k] * error
    return errors / 4.0


def tolerable_photons(max_information: float, phase_randomized: bool = False) -> float:
    """Największe μ_out, przy którym Ewa wie o bicie nie więcej niż `max_information`."""
    if not 0.0 < max_information < 1.0:
        raise ValueError("dopuszczalna informacja musi być w (0, 1)")

    def gap(log_mu: float) -> float:
        return bit_information(math.exp(log_mu), phase_randomized) - max_information

    return math.exp(brentq(gap, math.log(1e-30), math.log(50.0)))


def required_isolation_db(
    probe_photons: float,
    max_information: float,
    reflection_db: float,
    attenuator_db: float = 0.0,
    phase_randomized: bool = False,
) -> float:
    """Izolacja, przy której sonda o μ_in fotonach daje Ewie najwyżej `max_information` bitu."""
    total = 10.0 * math.log10(probe_photons / tolerable_photons(max_information, phase_randomized))
    return total - reflection_db - 2.0 * attenuator_db
