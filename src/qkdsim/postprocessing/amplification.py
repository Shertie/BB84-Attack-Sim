"""Weryfikacja kluczy, długość klucza i wzmacnianie prywatności (F2.5–F2.7)."""

import math
from dataclasses import dataclass

import numpy as np

from qkdsim.analysis.entropy import h2
from qkdsim.classical import ClassicalChannel, Message
from qkdsim.postprocessing.hashing import random_seed, toeplitz_hash
from qkdsim.rng import Bits


@dataclass(frozen=True)
class VerificationResult:
    """Wynik weryfikacji kluczy haszem: zgodność, długość tagu i ziarno."""

    passed: bool
    tag_bits: int
    seed: Bits


def verify(
    alice: Bits,
    bob: Bits,
    public: np.random.Generator,
    channel: ClassicalChannel,
    tag_bits: int = 64,
) -> VerificationResult:
    """Alice ogłasza ziarno i swój hasz; różne klucze przechodzą z p-stwem 2^(−tag_bits).

    Hasz ujawnia `tag_bits` bitów informacji o kluczu, więc wchodzi do wycieku.
    """
    seed = random_seed(public, len(alice), tag_bits)
    channel.send(Message("alice", "verify_seed", len(seed), payload=seed))
    tag = toeplitz_hash(alice, seed, tag_bits)
    channel.send(Message("alice", "verify_hash", tag_bits, leaked_bits=tag_bits, payload=tag))
    passed = bool(np.array_equal(tag, toeplitz_hash(bob, seed, tag_bits)))
    return VerificationResult(passed, tag_bits, seed)


def key_length(
    n: int,
    phase_error: float,
    leaked_bits: int,
    single_photon_fraction: float = 1.0,
) -> int:
    """ℓ = ⌊n·s·[1 − h(e_ph)] − wyciek⌋, nie mniej niż 0 (wersja asymptotyczna, F2.7).

    Dla źródła jednofotonowego s = 1 i e_ph to błąd fazowy. Dla WCP s = Q₁/Q_μ to udział
    detekcji z impulsów jednofotonowych, a e_ph = e₁ (GLLP); w F2 oba pochodzą z modelu,
    w F3 z estymaty decoy. Poprawki skończonego klucza dochodzą w F4.
    """
    if not 0.0 <= single_photon_fraction <= 1.0:
        raise ValueError("single_photon_fraction musi być w [0, 1]")
    secret = n * single_photon_fraction * (1.0 - h2(min(phase_error, 0.5)))
    return max(0, math.floor(secret - leaked_bits))


@dataclass(frozen=True)
class AmplificationResult:
    """Klucze po wzmacnianiu prywatności i ziarno macierzy Toeplitza."""

    alice: Bits
    bob: Bits
    seed: Bits


def privacy_amplification(
    alice: Bits, bob: Bits, length: int, public: np.random.Generator, channel: ClassicalChannel
) -> AmplificationResult:
    """Oba klucze mnożone przez tę samą losową macierz Toeplitza `length` × n."""
    if not 0 <= length <= len(alice):
        raise ValueError(f"długość klucza końcowego musi być w [0, {len(alice)}]")
    seed = random_seed(public, len(alice), length)
    channel.send(Message("alice", "pa_seed", len(seed), payload=seed))
    return AmplificationResult(
        toeplitz_hash(alice, seed, length), toeplitz_hash(bob, seed, length), seed
    )
