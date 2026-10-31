"""MDI-QKD: Alice i Bob wysyłają stany BB84 do niezaufanego przekaźnika (Lo, Curty, Qi 2012), F7.

Przekaźnik (rys. 1 artykułu): dzielnik 50:50, za każdym wyjściem PBS (H/V) i dwa detektory
progowe. Udany pomiar Bella to dokładnie dwa kliknięcia o ortogonalnych polaryzacjach:
(1H, 2V) albo (1V, 2H) → |ψ⁻⟩, (1H, 1V) albo (2H, 2V) → |ψ⁺⟩. Po ogłoszeniu wyniku Bob odwraca
bit, z wyjątkiem bazy diagonalnej i wyniku |ψ⁺⟩ (Tabela 1). Klucz z bazy prostej (Z),
baza diagonalna (X) tylko do oszacowania błędu fazowego.

Fizyka jest dokładna dla stanów koherentnych z losową fazą: przy ustalonych fazach φ_A, φ_B
każdy z czterech detektorów dostaje stan koherentny, więc liczby fotonów są niezależne
(Poisson) o średnich η|amplituda|². Sprawność detektorów (równa dla czterech) mnoży wszystkie
średnie. Błąd optyki to obrót polaryzacji w ramieniu Boba przed dzielnikiem i w wyjściu 1 przed
PBS (model z podpisu rys. 2 artykułu).

Przekaźnik może należeć do Ewy (`RelayAttack`): wtedy ogłasza, co chce, a detektory przestają
mieć znaczenie (tak MDI usuwa ataki 2 i 4).
"""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from qkdsim.optics import polarization_angle
from qkdsim.rng import Bits, RandomStreams
from qkdsim.source import WCPSource

FAIL, PSI_MINUS, PSI_PLUS = 0, 1, 2
Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Relay:
    """Detektory przekaźnika i błąd optyki jako obroty polaryzacji."""

    efficiency: float  # transmisja optyki przekaźnika × sprawność detektora
    dark_count: float  # na detektor i bramkę
    rotation_in: float = 0.0  # obrót w ramieniu Boba przed dzielnikiem [rad]
    rotation_out: float = 0.0  # obrót w wyjściu 1 dzielnika przed PBS [rad]

    @classmethod
    def with_misalignment(cls, efficiency: float, dark_count: float, total: float) -> "Relay":
        """Błąd optyki `total` rozłożony po połowie na dwa obroty (Lo, Curty, Qi 2012, rys. 2)."""
        angle = math.asin(math.sqrt(total / 2.0))
        return cls(efficiency, dark_count, angle, angle)


def detector_means(
    mu_a: Floats,
    mu_b: Floats,
    theta_a: Floats,
    theta_b: Floats,
    relative_phase: Floats,
    relay: Relay,
) -> Floats:
    """Średnie liczby wykrytych fotonów w 1H, 1V, 2H, 2V (wiersze), `mu_*` po kanale."""
    amp_a = np.sqrt(mu_a)
    amp_b = np.sqrt(mu_b) * np.exp(1j * relative_phase)
    tb = theta_b + relay.rotation_in
    a_h, a_v = amp_a * np.cos(theta_a), amp_a * np.sin(theta_a)
    b_h, b_v = amp_b * np.cos(tb), amp_b * np.sin(tb)
    out1_h, out1_v = (a_h + b_h) / math.sqrt(2.0), (a_v + b_v) / math.sqrt(2.0)
    out2_h, out2_v = (a_h - b_h) / math.sqrt(2.0), (a_v - b_v) / math.sqrt(2.0)
    c, s = math.cos(relay.rotation_out), math.sin(relay.rotation_out)
    out1_h, out1_v = c * out1_h - s * out1_v, s * out1_h + c * out1_v
    return relay.efficiency * np.abs(np.stack([out1_h, out1_v, out2_h, out2_v])) ** 2


def bell_outcome(clicks: npt.NDArray[np.bool_]) -> npt.NDArray[np.int8]:
    """Wynik pomiaru Bella z kliknięć (wiersze 1H, 1V, 2H, 2V): FAIL, PSI_MINUS albo PSI_PLUS."""
    h1, v1, h2, v2 = clicks
    two = clicks.sum(axis=0) == 2
    minus = two & ((h1 & v2) | (v1 & h2))
    plus = two & ((h1 & v1) | (h2 & v2))
    return np.where(minus, PSI_MINUS, np.where(plus, PSI_PLUS, FAIL)).astype(np.int8)


def bob_flips(bases: Bits, outcome: npt.NDArray[np.int8]) -> npt.NDArray[np.bool_]:
    """Tabela 1: Bob odwraca bit zawsze, poza bazą diagonalną z wynikiem |ψ⁺⟩."""
    return np.asarray(~((bases == 1) & (outcome == PSI_PLUS)), dtype=np.bool_)


@dataclass(frozen=True, eq=False)
class MDIRecord:
    """Wybory Alice i Boba oraz ogłoszony wynik przekaźnika dla każdej pary impulsów."""

    alice_bits: Bits
    alice_bases: Bits
    alice_intensity: Bits
    bob_bits: Bits
    bob_bases: Bits
    bob_intensity: Bits
    outcome: npt.NDArray[np.int8]
    eve_bits: Bits | None = None  # bity Alice, które zna Ewa w przekaźniku (atak)


class RelayAttack(Protocol):
    """Ewa jako przekaźnik: dostaje stany Alice i Boba, ogłasza dowolny wynik."""

    def announce(
        self,
        mu_a: Floats,
        mu_b: Floats,
        theta_a: Floats,
        theta_b: Floats,
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.int8], Bits]: ...


@dataclass(frozen=True)
class MDILink:
    """Dwa źródła WCP, dwa odcinki światłowodu i przekaźnik pośrodku."""

    source_a: WCPSource
    source_b: WCPSource
    length_a_km: float
    length_b_km: float
    relay: Relay
    alpha_db_per_km: float = 0.2
    attack: RelayAttack | None = None

    def transmittance(self, length_km: float) -> float:
        return float(10.0 ** (-self.alpha_db_per_km * length_km / 10.0))


def simulate_mdi(link: MDILink, n_pairs: int, streams: RandomStreams) -> MDIRecord:
    """Jedna paczka par impulsów; Alice losuje ze `streams.alice`, Bob ze `streams.bob`."""
    choices = []
    for source, rng in ((link.source_a, streams.alice), (link.source_b, streams.bob)):
        bits, bases = rng.bits(n_pairs), rng.bases(n_pairs)
        if len(source.intensities) == 1:
            intensity = np.zeros(n_pairs, dtype=np.uint8)
        else:
            intensity = rng.choice(n_pairs, source.probabilities)
        mu = np.asarray(source.intensities)[intensity]
        choices.append((bits, bases, intensity, mu))
    (a_bits, a_bases, a_int, mu_a), (b_bits, b_bases, b_int, mu_b) = choices
    mu_a = mu_a * link.transmittance(link.length_a_km)
    mu_b = mu_b * link.transmittance(link.length_b_km)
    theta_a, theta_b = polarization_angle(a_bits, a_bases), polarization_angle(b_bits, b_bases)
    eve_bits = None
    if link.attack is None:
        phase = 2.0 * np.pi * streams.nature.random(n_pairs)  # φ_A − φ_B, obie fazy losowe
        means = detector_means(mu_a, mu_b, theta_a, theta_b, phase, link.relay)
        silent = (1.0 - link.relay.dark_count) * np.exp(-means)
        outcome = bell_outcome(streams.nature.random(means.shape) >= silent)
    else:
        outcome, eve_bits = link.attack.announce(mu_a, mu_b, theta_a, theta_b, streams.eve)
    return MDIRecord(a_bits, a_bases, a_int, b_bits, b_bases, b_int, outcome, eve_bits)


@dataclass(frozen=True, eq=False)
class MDICounts:
    """Liczniki o kształcie (intensywność Alice, intensywność Boba, baza) dla zgodnych baz."""

    sent: npt.NDArray[np.int64]
    detected: npt.NDArray[np.int64]
    errors: npt.NDArray[np.int64]
    eve_correct: npt.NDArray[np.int64]  # wykryte bity Alice, które Ewa zna (atak w przekaźniku)

    @classmethod
    def from_record(cls, record: MDIRecord, n_a: int, n_b: int) -> "MDICounts":
        same = record.alice_bases == record.bob_bases
        cell = (record.alice_intensity.astype(np.intp) * n_b + record.bob_intensity) * 2
        cell = cell + record.alice_bases
        detected = same & (record.outcome != FAIL)
        bob_key = record.bob_bits ^ bob_flips(record.bob_bases, record.outcome).astype(np.uint8)
        wrong = detected & (bob_key != record.alice_bits)
        known = (
            np.zeros_like(detected)
            if record.eve_bits is None
            else detected & (record.eve_bits == record.alice_bits)
        )
        size = n_a * n_b * 2

        def table(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
            counts = np.bincount(cell[mask], minlength=size).astype(np.int64)
            return counts.reshape(n_a, n_b, 2)

        return cls(table(same), table(detected), table(wrong), table(known))

    def __add__(self, other: "MDICounts") -> "MDICounts":
        return MDICounts(
            self.sent + other.sent,
            self.detected + other.detected,
            self.errors + other.errors,
            self.eve_correct + other.eve_correct,
        )


def run_mdi(
    link: MDILink, n_pairs: int, streams: RandomStreams, batch_size: int = 2**20
) -> MDICounts:
    """Liczniki z `n_pairs` par impulsów, liczone w paczkach."""
    n_a, n_b = len(link.source_a.intensities), len(link.source_b.intensities)
    total = MDICounts(*(np.zeros((n_a, n_b, 2), dtype=np.int64) for _ in range(4)))
    done = 0
    while done < n_pairs:
        size = min(batch_size, n_pairs - done)
        total = total + MDICounts.from_record(simulate_mdi(link, size, streams), n_a, n_b)
        done += size
    return total


@dataclass(frozen=True)
class MeasureAndAnnounce:
    """Ewa jako przekaźnik: mierzy oba impulsy w bazie Z i ogłasza |ψ⁻⟩, gdy wyniki się różnią.

    W bazie klucza Bob po odwróceniu dostaje dokładnie bit Alice, a Ewa zna go z pomiaru.
    W bazie diagonalnej jej wyniki są losowe, więc ogłoszenia nie zależą od bitów i połowa
    bitów po siftingu jest błędna. Pomiar impulsu wielofotonowego to jedno losowanie Borna.
    """

    def announce(
        self,
        mu_a: Floats,
        mu_b: Floats,
        theta_a: Floats,
        theta_b: Floats,
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.int8], Bits]:
        size = len(mu_a)
        results = []
        for mu, theta in ((mu_a, theta_a), (mu_b, theta_b)):
            arrived = rng.random(size) >= np.exp(-mu)
            bit = (rng.random(size) >= np.cos(theta) ** 2).astype(np.uint8)
            results.append((arrived, bit))
        (arrived_a, bit_a), (arrived_b, bit_b) = results
        announce = arrived_a & arrived_b & (bit_a != bit_b)
        return np.where(announce, PSI_MINUS, FAIL).astype(np.int8), bit_a
