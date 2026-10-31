"""Alice, Bob i sifting.

Etap 1 (F1.3): idealny BB84, jeden foton na impuls, kanał bez strat. Etap 2 (F1.7): pomiar
parą detektorów progowych z ciemnymi zliczeniami i podwójnymi kliknięciami.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qkdsim.detector import DetectorPair
from qkdsim.optics import analyzer_angle, polarization_angle, prob_detector0
from qkdsim.pulses import Pulses
from qkdsim.rng import Bits, RandomSource, RandomStreams

BASIS_Z = 0
BASIS_X = 1


@dataclass(frozen=True, eq=False)
class AliceRecord:
    """To, co Alice wie o swoich impulsach."""

    bits: Bits
    bases: Bits
    intensity: Bits


@dataclass(frozen=True, eq=False)
class BobRecord:
    """To, co Bob wie o swoich pomiarach."""

    bases: Bits
    bits: Bits
    detected: npt.NDArray[np.bool_]
    double: npt.NDArray[np.bool_]  # kliknęły oba detektory; bit wylosowany (squashing)
    attenuated: npt.NDArray[np.bool_] | None = None  # bramki z tłumikiem Boba (obrona F3.9)


@dataclass(frozen=True, eq=False)
class SiftedKey:
    """Bity po siftingu i pozycje impulsów, z których pochodzą."""

    index: npt.NDArray[np.intp]
    alice: Bits
    bob: Bits

    def __len__(self) -> int:
        return int(self.index.shape[0])


def alice_prepare_single_photons(n_pulses: int, rng: RandomSource) -> tuple[AliceRecord, Pulses]:
    """Alice losuje bity i bazy i wysyła po jednym fotonie na impuls."""
    bits = rng.bits(n_pulses)
    bases = rng.bases(n_pulses)
    pulses = Pulses.build(
        n=np.ones(n_pulses, dtype=np.int64),
        bit=bits,
        basis=bases,
        theta=polarization_angle(bits, bases),
    )
    return AliceRecord(bits=bits, bases=bases, intensity=pulses.intensity), pulses


def bob_measure_ideal(pulses: Pulses, rng: RandomSource, nature: np.random.Generator) -> BobRecord:
    """Bob losuje bazę i mierzy pojedynczy foton idealnym detektorem."""
    if np.any(pulses.n != 1):
        raise ValueError("pomiar idealny obsługuje tylko impulsy jednofotonowe (etap 1)")
    bases = rng.bases(len(pulses))
    p0 = prob_detector0(pulses.theta, analyzer_angle(bases))
    bits = (nature.random(len(pulses)) >= p0).astype(np.uint8)
    return BobRecord(
        bases=bases,
        bits=bits,
        detected=np.ones(len(pulses), dtype=np.bool_),
        double=np.zeros(len(pulses), dtype=np.bool_),
    )


def bob_measure(
    pulses: Pulses,
    detectors: DetectorPair,
    rng: RandomSource,
    nature: np.random.Generator,
    random_assignment: bool = False,
) -> BobRecord:
    """Bob losuje bazę i mierzy impulsy parą detektorów progowych (etap 2).

    Pojedyncze kliknięcie daje bit równy numerowi detektora. Podwójne kliknięcie dostaje
    losowy bit z generatora Boba i nie jest odrzucane (squashing, docs/model.md).

    `random_assignment` (obrona przed time-shift, F6.4): Bob w każdej bramce losuje, czy
    zamienić przypisanie bit ↔ detektor, i odwraca odczyt tam, gdzie zamienił.
    """
    bases = rng.bases(len(pulses))
    swap = rng.bits(len(pulses)) if random_assignment else None
    click0, click1 = detectors.clicks(pulses, bases, nature, swap)
    double = click0 & click1
    bits = click1.astype(np.uint8)
    if swap is not None:
        bits ^= swap
    n_double = int(np.count_nonzero(double))
    if n_double:
        bits[double] = rng.bits(n_double)
    return BobRecord(bases=bases, bits=bits, detected=click0 | click1, double=double)


def sift(alice: AliceRecord, bob: BobRecord) -> SiftedKey:
    """Zostają impulsy wykryte przez Boba i zmierzone w bazie Alice."""
    index = np.flatnonzero(bob.detected & (alice.bases == bob.bases))
    return SiftedKey(index=index, alice=alice.bits[index], bob=bob.bits[index])


def run_ideal(n_pulses: int, streams: RandomStreams) -> tuple[AliceRecord, BobRecord, SiftedKey]:
    """Etap 1 w całości: przygotowanie, pomiar, sifting."""
    alice, pulses = alice_prepare_single_photons(n_pulses, streams.alice)
    bob = bob_measure_ideal(pulses, streams.bob, streams.nature)
    return alice, bob, sift(alice, bob)
