"""Alice, Bob i sifting.

Etap 1 (ROADMAP, F1.3): idealny BB84, jeden foton na impuls, kanał bez strat. Pomiar Boba
już teraz korzysta z reguły Borna na kątach polaryzacji, więc etap 2 dokłada tylko straty,
liczbę fotonów i detektory, bez zmiany konwencji.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qkdsim.pulses import Pulses
from qkdsim.rng import Bits, RandomSource, RandomStreams

BASIS_Z = 0
BASIS_X = 1

Angles = npt.NDArray[np.float64]


def polarization_angle(bits: Bits, bases: Bits) -> Angles:
    """Kąt stanu Alice: H = 0, V = π/2, D = π/4, A = 3π/4 (docs/model.md, rozdz. 2)."""
    return bases.astype(np.float64) * (np.pi / 4) + bits.astype(np.float64) * (np.pi / 2)


def analyzer_angle(bases: Bits) -> Angles:
    """Kąt PBS Boba: 0 dla bazy Z, π/4 dla bazy X."""
    return bases.astype(np.float64) * (np.pi / 4)


def prob_detector0(theta: Angles, phi: Angles) -> Angles:
    """Reguła Borna cos²(θ − φ): p-stwo, że foton trafi do detektora 0 (bit 0).

    Liczone jako (1 + cos 2(θ − φ)) / 2, co daje dokładnie 0 i 1 dla stanów z bazy analizatora.
    Wprost liczone cos² zostawiłoby resztę rzędu 10⁻³³ zamiast zera.
    """
    return 0.5 * (1.0 + np.cos(2.0 * (theta - phi)))


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
    return BobRecord(bases=bases, bits=bits, detected=np.ones(len(pulses), dtype=np.bool_))


def sift(alice: AliceRecord, bob: BobRecord) -> SiftedKey:
    """Zostają impulsy wykryte przez Boba i zmierzone w bazie Alice."""
    index = np.flatnonzero(bob.detected & (alice.bases == bob.bases))
    return SiftedKey(index=index, alice=alice.bits[index], bob=bob.bits[index])


def run_ideal(n_pulses: int, streams: RandomStreams) -> tuple[AliceRecord, BobRecord, SiftedKey]:
    """Etap 1 w całości: przygotowanie, pomiar, sifting."""
    alice, pulses = alice_prepare_single_photons(n_pulses, streams.alice)
    bob = bob_measure_ideal(pulses, streams.bob, streams.nature)
    return alice, bob, sift(alice, bob)
