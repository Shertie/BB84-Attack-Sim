"""Atak 8: słaby generator losowy Alice.

Generator Alice to wstrzykiwany `RandomSource`, więc podmiana go zmienia tylko jej
wybory. Warianty:
- `known_seed_source`: MT19937 o ziarnie znanym Ewie; statystycznie dobry, ale przewidywalny;
- `BiasedSource` (z rng.py): obciążone bity i bazy (np. 0,55/0,45);
- `MarkovSource`: kolejne bity (bazy) powtarzają poprzedni z p-stwem `stay`.

Ewa korzysta z tego na dwa sposoby:
- `ReplayPredictor` odtwarza losowania Alice (te same wywołania w tej samej kolejności co
  `WCPSource.emit`: bity, bazy, ewentualnie intensywności) i daje bazy do intercept-resend
  albo cały surowy klucz do ataku pasywnego `PassivePrediction`;
- `MajorityPredictor` zawsze stawia na częstszą bazę (i częstszy bit).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qkdsim.eve import NO_BASIS, EveLog
from qkdsim.pulses import Pulses
from qkdsim.rng import BiasedSource, Bits, GeneratorSource, RandomSource

__all__ = [
    "BiasedSource",
    "MajorityPredictor",
    "MarkovSource",
    "PassivePrediction",
    "ReplayPredictor",
    "known_seed_source",
]


def known_seed_source(seed: int) -> GeneratorSource:
    """Generator Alice z ziarnem, które zna Ewa (MT19937, jak `random` w Pythonie)."""
    return GeneratorSource(np.random.Generator(np.random.MT19937(seed)))


class MarkovSource:
    """Symetryczne łańcuchy Markowa: następny bit (baza) równy poprzedniemu z p-stwem `stay`.

    Stan łańcucha przechodzi między wywołaniami, więc paczki tworzą jeden ciąg.
    """

    def __init__(
        self, generator: np.random.Generator, stay_bits: float = 0.5, stay_bases: float = 0.5
    ):
        if not (0.0 <= stay_bits <= 1.0 and 0.0 <= stay_bases <= 1.0):
            raise ValueError("stay_bits i stay_bases muszą być w [0, 1]")
        self._generator = generator
        self.stay = {"bits": stay_bits, "bases": stay_bases}
        self._last = {"bits": int(generator.integers(0, 2)), "bases": int(generator.integers(0, 2))}

    def _chain(self, n: int, name: str) -> Bits:
        flips = (self._generator.random(n) >= self.stay[name]).astype(np.uint8)
        flips[0] ^= self._last[name]  # pierwszy element liczony od ostatniego z poprzedniej paczki
        chain = np.bitwise_xor.accumulate(flips)
        if n:
            self._last[name] = int(chain[-1])
        return chain

    def bits(self, n: int) -> Bits:
        return self._chain(n, "bits")

    def bases(self, n: int) -> Bits:
        return self._chain(n, "bases")

    def choice(self, n: int, p: Sequence[float]) -> Bits:
        return self._generator.choice(len(p), size=n, p=p).astype(np.uint8)


class ReplayPredictor:
    """Kopia generatora Alice u Ewy, wywoływana w tej samej kolejności co `WCPSource.emit`."""

    def __init__(self, replica: RandomSource, probabilities: Sequence[float] = (1.0,)):
        self._replica, self._probabilities = replica, tuple(probabilities)

    def predict(self, n: int) -> tuple[Bits, Bits]:
        bits = self._replica.bits(n)
        bases = self._replica.bases(n)
        if len(self._probabilities) > 1:
            self._replica.choice(n, self._probabilities)
        return bits, bases

    def bases(self, n: int) -> Bits:
        return self.predict(n)[1]


@dataclass(frozen=True)
class MajorityPredictor:
    """Ewa zawsze stawia na częstszą bazę i częstszy bit; pewności nie ma."""

    basis: int = 0
    bit: int = 0

    def bases(self, n: int) -> Bits:
        return np.full(n, self.basis, dtype=np.uint8)

    def predict(self, n: int) -> tuple[Bits, Bits]:
        return np.full(n, self.bit, dtype=np.uint8), np.full(n, NO_BASIS, dtype=np.uint8)


@dataclass(eq=False)
class PassivePrediction:
    """Atak pasywny: Ewa nie dotyka impulsów, tylko przewiduje bity Alice.

    W `EveLog` baza = przewidziana baza Alice, gdy Ewa ją zna (`ReplayPredictor`), więc
    `eve_known` liczy bity znane na pewno; NO_BASIS, gdy tylko zgaduje (`MajorityPredictor`).
    """

    predictor: ReplayPredictor | MajorityPredictor
    position: float = 0.0
    lossless_after: bool = False

    def intercept(self, pulses: Pulses, rng: np.random.Generator) -> tuple[Pulses, EveLog]:
        bits, bases = self.predictor.predict(len(pulses))
        return pulses, EveLog(measured=np.ones(len(pulses), dtype=np.bool_), basis=bases, bit=bits)
