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


class BiasedSource:
    """Bity równe 1 z p-stwem `p_one`, baza X z p-stwem `p_x`.

    Uczciwy protokół asymetryczny (Lim et al. 2014) wybiera bazy z p-stwem różnym od ½;
    atak 8 używa tej samej klasy jako obciążonego generatora.
    """

    def __init__(self, generator: np.random.Generator, p_one: float = 0.5, p_x: float = 0.5):
        if not (0.0 <= p_one <= 1.0 and 0.0 <= p_x <= 1.0):
            raise ValueError("p_one i p_x muszą być w [0, 1]")
        self._generator, self.p_one, self.p_x = generator, p_one, p_x

    def bits(self, n: int) -> Bits:
        return (self._generator.random(n) < self.p_one).astype(np.uint8)

    def bases(self, n: int) -> Bits:
        return (self._generator.random(n) < self.p_x).astype(np.uint8)

    def choice(self, n: int, p: Sequence[float]) -> Bits:
        return self._generator.choice(len(p), size=n, p=p).astype(np.uint8)


@dataclass(frozen=True)
class RandomStreams:
    """Niezależne strumienie losowości jednego przebiegu, wyprowadzone z jednego ziarna."""

    alice: RandomSource
    bob: RandomSource
    nature: np.random.Generator
    eve: np.random.Generator
    # Losowość jawna, ogłaszana kanałem klasycznym: próbka do estymacji, permutacje Cascade,
    # ziarna haszy. Osobny strumień, żeby postprocessing nie przesuwał losowań stron.
    public: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int | np.random.SeedSequence) -> Self:
        """Z ziarna albo z gałęzi SeedSequence (np. osobnej dla każdego punktu L).

        Gałęzie SeedSequence zależą tylko od swojego numeru, więc dodanie strumienia `public`
        jako piątego nie zmieniło czterech pierwszych (wyniki F1 są odtwarzalne).
        """
        sequence = (
            seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        )
        alice, bob, nature, eve, public = sequence.spawn(5)
        return cls(
            alice=GeneratorSource(np.random.default_rng(alice)),
            bob=GeneratorSource(np.random.default_rng(bob)),
            nature=np.random.default_rng(nature),
            eve=np.random.default_rng(eve),
            public=np.random.default_rng(public),
        )
