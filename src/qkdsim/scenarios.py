"""Gotowe scenariusze ataków dla interfejsu interaktywnego (app/streamlit_app.py).

Każdy scenariusz buduje łącze GYS z wybranym atakiem i obroną tak samo jak eksperymenty
w experiments/, uruchamia Monte Carlo w warstwie liczników i zestawia wynik z modelem
analitycznym tego ataku. Wiedza Ewy to informacja na bit po siftingu: ułamek bitów znanych na
pewno (intercept-resend, oślepianie, PNS, słaby generator, Trojan horse z losową fazą),
1 − h(trafność) dla sondy o stałej fazie albo granica K_U Zhao et al. dla time-shift.
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from qkdsim.analysis.blinding import blinding_model
from qkdsim.analysis.entropy import h2
from qkdsim.analysis.intercept_resend import intercept_resend_model
from qkdsim.analysis.key_rate import (
    GYS,
    best_over_mu,
    infinite_decoy_singles,
    rate_gllp_no_decoy,
    rate_infinite_decoy,
    rate_vacuum_weak,
)
from qkdsim.analysis.pns import pns_model, pns_strategy
from qkdsim.analysis.time_shift import (
    eve_information,
    fit_zhao,
    gain_matching_transmission,
    time_shift_model,
)
from qkdsim.analysis.trojan_horse import bit_information
from qkdsim.analysis.wcp import gain as gain_model
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.attacks.blinding import DetectorBlinding, matching_send_fraction
from qkdsim.attacks.intercept_resend import InterceptResend
from qkdsim.attacks.pns import PhotonNumberSplitting
from qkdsim.attacks.time_shift import TimeShift, detection_table
from qkdsim.attacks.trojan_horse import TrojanHorse
from qkdsim.attacks.weak_rng import MajorityPredictor
from qkdsim.channel import FiberChannel
from qkdsim.defenses.random_attenuation import RandomAttenuator, attenuation_z
from qkdsim.detector import CLAVIS2, DetectorPair, dark_count_per_detector
from qkdsim.eve import ChannelAttack
from qkdsim.metrics import (
    attenuation_tally,
    eve_guess_accuracy,
    eve_known_fraction,
    gain,
    matched_qber,
)
from qkdsim.pipeline import Link, run_counts, run_sifted
from qkdsim.postprocessing.run import KeyBalance, PostprocessingConfig, postprocess
from qkdsim.rng import BiasedSource, RandomStreams
from qkdsim.source import WCPSource

P_D = dark_count_per_detector(GYS.background)
THETA_E = math.asin(math.sqrt(GYS.misalignment))  # obrót błędu optyki
TROJAN_PROBE = 1e9  # μ_in; tłumienie dobrane do żądanego μ_out
ZHAO = fit_zhao()
ATTENUATOR = RandomAttenuator(0.1, 0.5)

NONE = "Bez ataku"
INTERCEPT_RESEND = "1. Intercept-resend"
BLINDING = "2. Oślepianie detektorów"
PNS = "3. Photon-number splitting"
TIME_SHIFT = "4. Time-shift"
TROJAN = "5. Trojan horse"
WEAK_RNG = "8. Słaby generator (obciążone bazy)"
ATTACKS = (NONE, INTERCEPT_RESEND, BLINDING, PNS, TIME_SHIFT, TROJAN, WEAK_RNG)


@dataclass(frozen=True)
class Settings:
    """Łącze GYS i rozmiar przebiegu."""

    length_km: float = 25.0
    mu: float = 0.48
    pulses: int = 1_000_000
    seed: int = 1

    def __post_init__(self) -> None:
        if self.length_km < 0.0 or self.mu <= 0.0 or self.pulses <= 0:
            raise ValueError("wymagane L ≥ 0, μ > 0 i dodatnia liczba impulsów")


@dataclass(frozen=True)
class Attack:
    """Wybrany atak i jego parametry; pola nieużywane przez dany atak są ignorowane."""

    name: str = NONE
    fraction: float = 1.0  # intercept-resend: ułamek przechwytywanych impulsów
    position: float = 1.0  # intercept-resend: 0 = tuż za Alice, 1 = tuż przed Bobem
    trigger_uw: float = 1100.0  # oślepianie: moc impulsu wyzwalającego [µW]
    calibrated: bool = True  # oślepianie: Ewa kompensuje obrót polaryzacji
    attenuator: bool = False  # obrona przed oślepianiem: losowe tłumienie 3 dB w 10% bramek
    random_assignment: bool = False  # obrona przed time-shift
    returned_photons: float = 0.1  # Trojan horse: μ_out
    random_phase: bool = False  # Trojan horse: Alice losuje fazę
    basis_bias: float = 0.8  # słaby generator: P(baza X)

    def __post_init__(self) -> None:
        if self.name not in ATTACKS:
            raise ValueError(f"nieznany atak {self.name!r}")


@dataclass(frozen=True)
class Outcome:
    """Monte Carlo i model: zysk, QBER, informacja Ewy na bit po siftingu."""

    gain: float
    gain_sigma: float
    qber: float
    qber_sigma: float
    eve: float
    eve_sigma: float
    model_gain: float | None
    model_qber: float
    model_eve: float
    honest_gain: float
    honest_qber: float
    defense: str | None = None  # opis wyniku obrony, jeśli włączona
    defense_alarm: bool = False  # obrona przerywa protokół


def detectors(attack: Attack) -> DetectorPair:
    if attack.name == BLINDING:
        return DetectorPair(GYS.eta_bob, P_D, CLAVIS2)
    if attack.name == TIME_SHIFT:
        return DetectorPair(GYS.eta_bob, P_D, timing=ZHAO.mismatch)
    return DetectorPair(GYS.eta_bob, P_D)


def _transmittance(length_km: float) -> float:
    return float(10.0 ** (-GYS.alpha_db_per_km * length_km / 10.0))


def time_shift_min_length(mu: float) -> float:
    """Najkrótsze łącze [km], przy którym Ewa wyrówna zysk w ataku time-shift (bisekcja)."""

    def feasible(length_km: float) -> bool:
        return (
            gain_matching_transmission(
                mu,
                _transmittance(length_km),
                GYS.eta_bob,
                P_D,
                GYS.misalignment,
                ZHAO.mismatch,
                ZHAO.shifts_ns,
            )
            is not None
        )

    low, high = 0.0, 200.0
    if feasible(low):
        return 0.0
    for _ in range(30):
        middle = 0.5 * (low + high)
        low, high = (low, middle) if feasible(middle) else (middle, high)
    return high


def _time_shift_transmission(settings: Settings) -> float:
    transmission = gain_matching_transmission(
        settings.mu,
        _transmittance(settings.length_km),
        GYS.eta_bob,
        P_D,
        GYS.misalignment,
        ZHAO.mismatch,
        ZHAO.shifts_ns,
    )
    if transmission is None:
        minimum = time_shift_min_length(settings.mu)
        raise ValueError(
            f"Przy {settings.length_km:g} km Ewa nie wyrówna zysku Boba: time-shift obniża "
            f"sprawność detektorów, więc potrzebuje strat w kanale (co najmniej {minimum:.1f} km)."
        )
    return transmission


def _p_a(settings: Settings, transmission: float) -> float:
    return time_shift_model(
        settings.mu, transmission, GYS.eta_bob, P_D, GYS.misalignment, ZHAO.mismatch, ZHAO.shifts_ns
    ).p_a


def eve_attack(settings: Settings, attack: Attack) -> ChannelAttack | None:
    """Atak w kanale, skalibrowany jak w eksperymentach."""
    honest = matched_basis_clicks(settings.mu, GYS.eta(settings.length_km), P_D, GYS.misalignment)
    if attack.name == INTERCEPT_RESEND:
        return InterceptResend(attack.fraction, attack.position)
    if attack.name == BLINDING:
        return DetectorBlinding(
            attack.trigger_uw * 1e-6,
            send_fraction=matching_send_fraction(honest.detected, settings.mu),
            mimic_qber=honest.qber,
            compensation_rad=THETA_E if attack.calibrated else 0.0,
        )
    if attack.name == PNS:
        return PhotonNumberSplitting(*pns_strategy(honest.detected, settings.mu, GYS.eta_bob, P_D))
    if attack.name == TIME_SHIFT:
        transmission = _time_shift_transmission(settings)
        return TimeShift(
            *ZHAO.shifts_ns, p_a=_p_a(settings, transmission), transmission=transmission
        )
    if attack.name == TROJAN:
        attenuation = 10.0 * math.log10(TROJAN_PROBE / attack.returned_photons)
        return TrojanHorse(TROJAN_PROBE, attenuation, attack.random_phase)
    if attack.name == WEAK_RNG:
        return InterceptResend(1.0, predictor=MajorityPredictor(basis=int(attack.basis_bias > 0.5)))
    return None


def build(settings: Settings, attack: Attack) -> tuple[Link, RandomStreams]:
    """Łącze i strumienie losowe przebiegu."""
    link = Link(
        WCPSource((settings.mu,)),
        FiberChannel(settings.length_km, GYS.alpha_db_per_km, GYS.misalignment),
        detectors(attack),
        eve=eve_attack(settings, attack),
        attenuator=ATTENUATOR if attack.name == BLINDING and attack.attenuator else None,
        random_assignment=attack.name == TIME_SHIFT and attack.random_assignment,
    )
    streams = RandomStreams.from_seed(np.random.SeedSequence(settings.seed))
    if attack.name == WEAK_RNG:
        alice = BiasedSource(np.random.default_rng([settings.seed, 1]), p_x=attack.basis_bias)
        streams = dataclasses.replace(streams, alice=alice)
    return link, streams


def model(settings: Settings, attack: Attack) -> tuple[float | None, float, float]:
    """(zysk, QBER, informacja Ewy) z modelu analitycznego ataku."""
    eta = GYS.eta(settings.length_km)
    honest = matched_basis_clicks(settings.mu, eta, P_D, GYS.misalignment)
    eve = eve_attack(settings, attack)
    if isinstance(eve, InterceptResend):
        right = max(attack.basis_bias, 1.0 - attack.basis_bias) if attack.name == WEAK_RNG else 0.5
        m = intercept_resend_model(
            GYS, settings.length_km, settings.mu, eve.fraction, eve.position, P_D, right
        )
        return m.clicks.detected, m.clicks.qber, m.known_fraction
    if isinstance(eve, DetectorBlinding):
        b = blinding_model(detectors(attack), eve, settings.mu, GYS.misalignment)
        return (None if attack.attenuator else b.detected), b.qber, b.known_fraction
    if isinstance(eve, PhotonNumberSplitting):
        p = pns_model(settings.mu, eve.block, eve.forward, GYS.eta_bob, P_D, GYS.misalignment)
        return p.clicks.detected, p.clicks.qber, p.known_fraction
    if isinstance(eve, TimeShift):
        t = time_shift_model(
            settings.mu,
            eve.transmission,
            GYS.eta_bob,
            P_D,
            GYS.misalignment,
            ZHAO.mismatch,
            ZHAO.shifts_ns,
            eve.p_a,
            attack.random_assignment,
        )
        return t.gain, t.qber, t.eve_information
    if isinstance(eve, TrojanHorse):
        return (
            honest.detected,
            honest.qber,
            bit_information(attack.returned_photons, attack.random_phase),
        )
    return honest.detected, honest.qber, 0.0


def simulate(settings: Settings, attack: Attack) -> Outcome:
    """Monte Carlo scenariusza i jego model."""
    link, streams = build(settings, attack)
    model_gain, model_qber, model_eve = model(settings, attack)
    honest = matched_basis_clicks(settings.mu, GYS.eta(settings.length_km), P_D, GYS.misalignment)
    if attack.name == TIME_SHIFT:
        table, matched = detection_table(link, settings.pulses, streams)
        detected = int(table.sum())
        if detected == 0:
            raise ValueError("Brak detekcji: zwiększ liczbę impulsów albo skróć łącze.")
        errors = int(table[..., 0, 1].sum() + table[..., 1, 0].sum())
        rate, error_rate = detected / matched, errors / detected
        return Outcome(
            rate,
            math.sqrt(rate * (1 - rate) / matched),
            error_rate,
            math.sqrt(error_rate * (1 - error_rate) / detected),
            eve_information(table),
            math.nan,  # estymator „plug-in”, przy małej liczbie detekcji obciążony w górę
            model_gain,
            model_qber,
            model_eve,
            honest.detected,
            honest.qber,
        )
    counts = run_counts(link, settings.pulses, streams)
    if counts.detected.sum() == 0:
        raise ValueError("Brak detekcji: zwiększ liczbę impulsów albo skróć łącze.")
    g, q = gain(counts), matched_qber(counts)
    if attack.name == TROJAN and not attack.random_phase:
        guess = eve_guess_accuracy(counts)
        p = min(max(guess.value, 1e-12), 1 - 1e-12)
        eve, eve_sigma = 1.0 - h2(p), abs(math.log2((1.0 - p) / p)) * guess.sigma
    elif attack.name == NONE:
        eve, eve_sigma = 0.0, 0.0
    else:
        known = eve_known_fraction(counts)
        eve, eve_sigma = known.value, known.sigma
    defense, alarm = None, False
    if link.attenuator is not None:
        eta = GYS.eta(settings.length_km)
        ratio = gain_model(settings.mu, ATTENUATOR.factor * eta, GYS.background) / gain_model(
            settings.mu, eta, GYS.background
        )
        z = attenuation_z(attenuation_tally(counts), ratio)
        alarm = abs(z) > 4.0
        verdict = "alarm: przerwanie protokołu" if alarm else "brak alarmu"
        defense = f"Test losowego tłumienia: z = {z:.1f} ({verdict}, próg |z| > 4)"
    return Outcome(
        g.value,
        g.sigma,
        q.value,
        q.sigma,
        eve,
        eve_sigma,
        model_gain,
        model_qber,
        model_eve,
        honest.detected,
        honest.qber,
        defense,
        alarm,
    )


def key_rates(mu: float, nu: float, lengths_km: list[float]) -> dict[str, list[float]]:
    """Stopy klucza GYS na impuls: GLLP bez decoy (z optymalnym μ dla każdego L, jak w e06),
    decoy vacuum+weak i granica nieskończenie wielu stanów decoy (przy podanym μ)."""

    def gllp(length_km: float) -> float:
        return best_over_mu(lambda m: rate_gllp_no_decoy(GYS, length_km, m))[0]

    return {
        "GLLP bez decoy (optymalne μ)": [gllp(x) for x in lengths_km],
        "decoy vacuum+weak": [rate_vacuum_weak(GYS, x, mu, nu) for x in lengths_km],
        "decoy (nieskończenie wiele stanów)": [rate_infinite_decoy(GYS, x, mu) for x in lengths_km],
    }


def final_key(settings: Settings, attack: Attack, reconciliation: str = "cascade") -> KeyBalance:
    """Pełny postprocessing klucza z przebiegu w warstwie bitów.

    Długość klucza: GLLP z udziałem detekcji jednofotonowych z granicy decoy dla uczciwego
    łącza i błędem fazowym równym QBER z próbki. Atak, który nie podnosi QBER (oślepianie,
    time-shift, Trojan horse), zostawia więc klucz, który Ewa zna.
    """
    link, streams = build(settings, attack)
    key = run_sifted(link, settings.pulses, streams)
    if len(key) < 100:
        raise ValueError("Za mało bitów po siftingu: zwiększ liczbę impulsów albo skróć łącze.")
    eta = GYS.eta(settings.length_km)
    y1, _ = infinite_decoy_singles(eta, GYS.background, GYS.misalignment)
    single = (
        y1 * settings.mu * math.exp(-settings.mu) / gain_model(settings.mu, eta, GYS.background)
    )
    config = PostprocessingConfig(
        reconciliation="ldpc" if reconciliation == "ldpc" else "cascade",
        single_photon_fraction=min(single, 1.0),
    )
    return postprocess(key, streams.public, config).balance
