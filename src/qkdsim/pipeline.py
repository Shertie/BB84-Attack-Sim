"""Tor impulsów Alice → kanał → Bob w dwóch warstwach danych (ADR 0002, F1.10).

- `run_bits`: pełne rekordy Alice i Boba oraz klucz po siftingu; do około 10⁷ impulsów;
- `run_counts`: pętla po paczkach zredukowanych do liczników; 10⁹–10¹⁰ impulsów bez trzymania
  tablic w pamięci;
- `run_counts_parallel`: to samo w wielu procesach, z ziarnem każdej części wyprowadzonym
  z jednej SeedSequence, więc wynik nie zależy od kolejności pracy procesów.

Wszystkie warstwy wywołują tę samą funkcję `simulate`, więc fizyka jest jedna.
"""

import dataclasses
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from qkdsim.channel import FiberChannel
from qkdsim.defenses.random_attenuation import RandomAttenuator
from qkdsim.detector import DetectorPair
from qkdsim.eve import ChannelAttack, EveLog
from qkdsim.protocol import AliceRecord, BobRecord, SiftedKey, bob_measure, sift
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

DEFAULT_BATCH_SIZE = 2**20

Table = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Link:
    """Fizyczna konfiguracja łącza, opcjonalnie z Ewą w światłowodzie."""

    source: WCPSource
    channel: FiberChannel
    detectors: DetectorPair
    eve: ChannelAttack | None = None
    attenuator: RandomAttenuator | None = None  # obrona Boba przed oślepianiem (F3.9)


def simulate(
    link: Link, n_pulses: int, streams: RandomStreams
) -> tuple[AliceRecord, BobRecord, EveLog]:
    """Jedna paczka impulsów przez cały tor. Bez Ewy jej log jest pusty."""
    alice, pulses = link.source.emit(n_pulses, streams.alice, streams.nature)
    if link.eve is None:
        pulses = link.channel.transmit(pulses, streams.nature)
        eve = EveLog.empty(n_pulses)
    else:
        before, after = link.channel.split(link.eve.position, link.eve.lossless_after)
        pulses = before.transmit(pulses, streams.nature)
        pulses, eve = link.eve.intercept(pulses, streams.eve)
        pulses = after.transmit(pulses, streams.nature)
    if link.attenuator is None:
        bob = bob_measure(pulses, link.detectors, streams.bob, streams.nature)
    else:
        pulses, attenuated = link.attenuator.apply(pulses, streams.bob, streams.nature)
        bob = bob_measure(pulses, link.detectors, streams.bob, streams.nature)
        bob = dataclasses.replace(bob, attenuated=attenuated)
    return alice, bob, eve


@dataclass(frozen=True)
class BitsRun:
    alice: AliceRecord
    bob: BobRecord
    eve: EveLog
    key: SiftedKey


def run_bits(link: Link, n_pulses: int, streams: RandomStreams) -> BitsRun:
    alice, bob, eve = simulate(link, n_pulses, streams)
    return BitsRun(alice=alice, bob=bob, eve=eve, key=sift(alice, bob))


@dataclass(frozen=True, eq=False)
class Counts:
    """Liczniki o kształcie (intensywność, baza Alice, baza Boba).

    `errors` liczy wykryte impulsy, w których bit Boba różni się od bitu Alice, także
    w niezgodnych bazach; QBER bierze się tylko z przekątnej baz.

    Liczniki Ewy dotyczą wykrytych impulsów:
    - `eve_measured`: Ewa zmierzyła impuls;
    - `eve_correct`: jej wynik równa się bitowi Alice;
    - `eve_known`: mierzyła w bazie Alice, więc po siftingu zna bit na pewno.

    `sent_attenuated` i `detected_attenuated` liczą bramki z tłumikiem Boba (obrona F3.9).
    """

    sent: Table
    detected: Table
    errors: Table
    double: Table
    eve_measured: Table
    eve_correct: Table
    eve_known: Table
    sent_attenuated: Table
    detected_attenuated: Table

    @classmethod
    def zeros(cls, n_intensities: int) -> Self:
        shape = (n_intensities, 2, 2)
        return cls(*(np.zeros(shape, dtype=np.int64) for _ in dataclasses.fields(cls)))

    @classmethod
    def from_records(
        cls, alice: AliceRecord, bob: BobRecord, n_intensities: int, eve: EveLog | None = None
    ) -> Self:
        cell = (alice.intensity.astype(np.intp) * 2 + alice.bases) * 2 + bob.bases
        size = n_intensities * 4

        def table(mask: npt.NDArray[np.bool_] | None) -> Table:
            selected = cell if mask is None else cell[mask]
            counts = np.bincount(selected, minlength=size).astype(np.int64)
            return counts.reshape(n_intensities, 2, 2)

        eve = EveLog.empty(len(cell)) if eve is None else eve
        measured = bob.detected & eve.measured
        attenuated = (
            np.zeros(len(cell), dtype=np.bool_) if bob.attenuated is None else bob.attenuated
        )
        return cls(
            sent=table(None),
            detected=table(bob.detected),
            errors=table(bob.detected & (alice.bits != bob.bits)),
            double=table(bob.double),
            eve_measured=table(measured),
            eve_correct=table(measured & (eve.bit == alice.bits)),
            eve_known=table(measured & (eve.basis == alice.bases)),
            sent_attenuated=table(attenuated),
            detected_attenuated=table(bob.detected & attenuated),
        )

    def __add__(self, other: Self) -> Self:
        return type(self)(
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in dataclasses.fields(self)
            }
        )

    @property
    def n_pulses(self) -> int:
        return int(self.sent.sum())


def run_counts(
    link: Link,
    n_pulses: int,
    streams: RandomStreams,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Counts:
    if batch_size <= 0:
        raise ValueError("batch_size musi być dodatnie")
    n_intensities = len(link.source.intensities)
    total = Counts.zeros(n_intensities)
    done = 0
    while done < n_pulses:
        size = min(batch_size, n_pulses - done)
        alice, bob, eve = simulate(link, size, streams)
        total = total + Counts.from_records(alice, bob, n_intensities, eve)
        done += size
    return total


DEFAULT_CHUNK_SIZE = 50_000_000


def _run_chunk(link: Link, n_pulses: int, seed: np.random.SeedSequence, batch_size: int) -> Counts:
    return run_counts(link, n_pulses, RandomStreams.from_seed(seed), batch_size)


def run_counts_parallel(
    link: Link,
    n_pulses: int,
    seed: int | np.random.SeedSequence,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_workers: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Counts:
    """`run_counts` podzielone na części po `chunk_size` impulsów, liczone w osobnych procesach.

    Każda część dostaje własną gałąź SeedSequence, więc wynik zależy tylko od ziarna
    i `chunk_size`, a nie od liczby procesów.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size musi być dodatnie")
    sizes = [chunk_size] * (n_pulses // chunk_size)
    if n_pulses % chunk_size:
        sizes.append(n_pulses % chunk_size)
    sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    seeds = sequence.spawn(len(sizes))
    total = Counts.zeros(len(link.source.intensities))
    if max_workers == 1 or len(sizes) <= 1:
        parts = [_run_chunk(link, n, s, batch_size) for n, s in zip(sizes, seeds, strict=True)]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            parts = list(
                pool.map(
                    _run_chunk,
                    [link] * len(sizes),
                    sizes,
                    seeds,
                    [batch_size] * len(sizes),
                )
            )
    for part in parts:
        total = total + part
    return total


def run_sifted(
    link: Link, n_pulses: int, streams: RandomStreams, batch_size: int = DEFAULT_BATCH_SIZE
) -> SiftedKey:
    """Klucz po siftingu z `n_pulses` impulsów, liczony w paczkach (pamięć: tylko klucz).

    Indeksy w zwróconym kluczu są globalne, liczone od pierwszego impulsu przebiegu.
    """
    if batch_size <= 0:
        raise ValueError("batch_size musi być dodatnie")
    parts: list[SiftedKey] = []
    done = 0
    while done < n_pulses:
        size = min(batch_size, n_pulses - done)
        alice, bob, _eve = simulate(link, size, streams)
        key = sift(alice, bob)
        parts.append(SiftedKey(index=key.index + done, alice=key.alice, bob=key.bob))
        done += size
    if not parts:
        empty = np.zeros(0, dtype=np.uint8)
        return SiftedKey(index=np.zeros(0, dtype=np.intp), alice=empty, bob=empty)
    return SiftedKey(
        index=np.concatenate([k.index for k in parts]),
        alice=np.concatenate([k.alice for k in parts]),
        bob=np.concatenate([k.bob for k in parts]),
    )
