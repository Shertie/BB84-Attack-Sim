"""Estymacja parametrów: ujawniona próbka bitów, QBER, odrzucenie próbki (F2.2).

Dwa warianty:

- `estimate_random_sample`: losowa próbka k bitów bez zwracania (symetryczny BB84, plan);
- `estimate_by_basis`: cała baza testowa idzie na estymację, klucz powstaje z drugiej bazy
  (asymetryczny protokół Tomamichela et al. 2012, potrzebny w F4).
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qkdsim.classical import ClassicalChannel, Message
from qkdsim.metrics import Estimate
from qkdsim.protocol import SiftedKey
from qkdsim.rng import Bits


@dataclass(frozen=True)
class SampleResult:
    """QBER z ujawnionej próbki i klucz bez tej próbki."""

    qber: Estimate  # QBER z ujawnionej próbki
    key: SiftedKey  # klucz bez próbki


def _subset(key: SiftedKey, mask: npt.NDArray[np.bool_]) -> SiftedKey:
    return SiftedKey(index=key.index[mask], alice=key.alice[mask], bob=key.bob[mask])


def estimate_random_sample(
    key: SiftedKey, sample_size: int, public: np.random.Generator, channel: ClassicalChannel
) -> SampleResult:
    """Ujawnia losową próbkę `sample_size` bitów i estymuje z niej QBER."""
    n = len(key)
    if not 0 < sample_size < n:
        raise ValueError(f"rozmiar próbki musi być w (0, {n}), jest {sample_size}")
    chosen = np.sort(public.choice(n, size=sample_size, replace=False))
    in_sample = np.zeros(n, dtype=np.bool_)
    in_sample[chosen] = True
    errors = int(np.count_nonzero(key.alice[chosen] != key.bob[chosen]))
    position_bits = sample_size * max(1, math.ceil(math.log2(n)))
    channel.send(Message("alice", "sample_positions", position_bits, payload=chosen))
    channel.send(Message("alice", "sample_bits", sample_size, payload=key.alice[chosen]))
    channel.send(Message("bob", "sample_bits", sample_size, payload=key.bob[chosen]))
    return SampleResult(Estimate(errors, sample_size), _subset(key, ~in_sample))


def estimate_by_basis(
    key: SiftedKey, bases: Bits, test_basis: int, channel: ClassicalChannel
) -> SampleResult:
    """`bases` to bazy pozycji klucza (po siftingu bazy Alice i Boba są równe)."""
    if bases.shape != key.alice.shape:
        raise ValueError("bases musi mieć długość klucza")
    test = bases == test_basis
    if not np.any(test) or np.all(test):
        raise ValueError("obie bazy muszą być reprezentowane w kluczu")
    errors = int(np.count_nonzero(key.alice[test] != key.bob[test]))
    k = int(np.count_nonzero(test))
    channel.send(Message("alice", "sample_bits", k, payload=key.alice[test]))
    channel.send(Message("bob", "sample_bits", k, payload=key.bob[test]))
    return SampleResult(Estimate(errors, k), _subset(key, ~test))
