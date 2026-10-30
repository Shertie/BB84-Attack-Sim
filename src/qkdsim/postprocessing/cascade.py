"""Korekcja błędów Cascade (Brassard, Salvail 1993), F2.3.

- przebieg 1: bloki rozmiaru k₁ ≈ 0,73/Q, kolejne przebiegi: rozmiar 2×, losowa permutacja;
- w bloku z niezgodną parzystością BINARY (bisekcja) znajduje i poprawia jeden błąd;
- poprawka zmienia parzystość bloków zawierających ten bit we wcześniejszych przebiegach,
  więc te bloki też idą do BINARY (efekt kaskady), najpierw najmniejsze.

Wyciek: jeden bit na każdą parzystość ogłoszoną przez Alice (parzystości bloków i połówek
w BINARY). Parzystości Boba ujawniają tylko, gdzie są błędy, a nie dodatkową informację
o kluczu Alice. Parzystości liczone są wektorowo; pętla w Pythonie idzie tylko po błędach.
"""

import heapq
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qkdsim.classical import ClassicalChannel, Message
from qkdsim.rng import Bits

Positions = npt.NDArray[np.intp]

SEED_BITS = 64  # permutacja ogłaszana jako ziarno wspólnego generatora


@dataclass(frozen=True)
class CascadeResult:
    """Poprawiony klucz Boba, wyciek, liczba poprawek i rozmiary bloków."""

    corrected: Bits  # klucz Boba po korekcji
    leaked_bits: int
    corrections: int  # liczba wywołań BINARY, czyli poprawionych bitów
    block_sizes: tuple[int, ...]


def initial_block_size(qber_estimate: float, n: int) -> int:
    """k₁ = ⌈0,73/Q⌉, przycięte do [2, n]; przy Q = 0 jeden blok na cały klucz."""
    if n <= 0:
        raise ValueError("klucz musi być niepusty")
    if qber_estimate <= 0.0:
        return n
    return int(min(n, max(2, math.ceil(0.73 / qber_estimate))))


def _binary(
    members: Positions, diff: Bits, record: bool
) -> tuple[int, int, list[Positions] | None]:
    """Bisekcja w bloku o nieparzystej liczbie błędów; zwraca pozycję błędu i liczbę kroków."""
    rows: list[Positions] | None = [] if record else None
    steps = 0
    while len(members) > 1:
        half = members[: len(members) // 2]
        steps += 1
        if rows is not None:
            rows.append(half)
        odd = int(np.count_nonzero(diff[half])) & 1
        members = half if odd else members[len(members) // 2 :]
    return int(members[0]), steps, rows


def cascade(
    alice: Bits,
    bob: Bits,
    qber_estimate: float,
    public: np.random.Generator,
    channel: ClassicalChannel,
    passes: int = 4,
    record_rows: bool = False,
) -> CascadeResult:
    """Poprawia klucz Boba do klucza Alice.

    `record_rows=True` zapisuje w transkrypcie pozycje każdej ogłoszonej parzystości,
    potrzebne do dokładnej analizy wiedzy liniowej Ewy (F2.8). Kosztuje pamięć.
    """
    n = len(alice)
    if len(bob) != n:
        raise ValueError("klucze Alice i Boba muszą mieć tę samą długość")
    corrected = bob.copy()
    diff = alice ^ corrected  # 1 tam, gdzie bit Boba jest błędny
    k1 = initial_block_size(qber_estimate, n)
    perms: list[Positions] = []
    block_of: list[Positions] = []
    block_diff: list[Bits] = []
    sizes: list[int] = []
    leaked = corrections = 0
    for p in range(passes):
        size = min(n, k1 * 2**p)
        perm = np.arange(n) if p == 0 else public.permutation(n)
        if p > 0:
            channel.send(Message("alice", "cascade_permutation", SEED_BITS))
        owner = np.empty(n, dtype=np.intp)
        owner[perm] = np.arange(n) // size
        starts = np.arange(0, n, size)
        parities = np.bitwise_xor.reduceat(diff[perm], starts).astype(np.uint8)
        perms.append(perm)
        block_of.append(owner)
        block_diff.append(parities)
        sizes.append(size)
        rows = [perm[s : s + size] for s in starts] if record_rows else None
        channel.send(
            Message("alice", "cascade_parities", len(starts), leaked_bits=len(starts), payload=rows)
        )
        leaked += len(starts)

        heap = [(p, int(b)) for b in np.flatnonzero(parities)]
        heapq.heapify(heap)
        while heap:
            q, b = heapq.heappop(heap)
            if not block_diff[q][b]:
                continue  # naprawiony w międzyczasie przez efekt kaskady
            members = perms[q][b * sizes[q] : (b + 1) * sizes[q]]
            j, steps, binary_rows = _binary(members, diff, record_rows)
            channel.send(
                Message("alice", "cascade_binary", steps, leaked_bits=steps, payload=binary_rows)
            )
            leaked += steps
            corrections += 1
            corrected[j] ^= 1
            diff[j] ^= 1
            for r in range(p + 1):
                blk = int(block_of[r][j])
                block_diff[r][blk] ^= 1
                if block_diff[r][blk]:
                    heapq.heappush(heap, (r, blk))
    return CascadeResult(corrected, leaked, corrections, tuple(sizes))
