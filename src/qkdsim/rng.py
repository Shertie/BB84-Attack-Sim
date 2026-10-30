"""Źródła losowości (ADR 0003).

Alice i Bob dostają wstrzykiwany RandomSource, żeby atak 8 (słaby generator) mógł podmienić
tylko ich wybory. Przyroda (liczba fotonów, wynik pomiaru) i Ewa mają osobne generatory, więc
podmiana generatora jednej strony nie zmienia pozostałych losowań.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import numpy.typing as npt

Bits = npt.NDArray[np.uint8]


class RandomSource(Protocol):
    """Losowe wybory jednej strony protokołu."""

    def bits(self, n: int) -> Bits:
        """n bitów klucza."""
        ...

    def bases(self, n: int) -> Bits:
        """n wyborów bazy: 0 = Z, 1 = X."""
        ...

    def choice(self, n: int, p: Sequence[float]) -> Bits:
        """n indeksów z rozkładu p, np. wybór intensywności decoy."""
        ...


class GeneratorSource:
    """Uczciwy RandomSource oparty na numpy.random.Generator."""

    def __init__(self, generator: np.random.Generator) -> None:
        self._generator = generator

    def bits(self, n: int) -> Bits:
        return self._generator.integers(0, 2, size=n, dtype=np.uint8)

    def bases(self, n: int) -> Bits:
        return self._generator.integers(0, 2, size=n, dtype=np.uint8)

    def choice(self, n: int, p: Sequence[float]) -> Bits:
        return self._generator.choice(len(p), size=n, p=p).astype(np.uint8)


@dataclass(frozen=True)
class RandomStreams:
    """Niezależne strumienie losowości jednego przebiegu, wyprowadzone z jednego ziarna."""

    alice: RandomSource
    bob: RandomSource
    nature: np.random.Generator
    eve: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int | np.random.SeedSequence) -> Self:
        """Z ziarna albo z gałęzi SeedSequence (np. osobnej dla każdego punktu L)."""
        sequence = (
            seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        )
        alice, bob, nature, eve = sequence.spawn(4)
        return cls(
            alice=GeneratorSource(np.random.default_rng(alice)),
            bob=GeneratorSource(np.random.default_rng(bob)),
            nature=np.random.default_rng(nature),
            eve=np.random.default_rng(eve),
        )
