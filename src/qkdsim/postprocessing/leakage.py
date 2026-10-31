"""Dokładna wiedza pasywnej, liniowej Ewy o kluczu końcowym.

Ewa słyszy cały kanał klasyczny. Każda parzystość Cascade, każdy bit syndromu LDPC i każdy bit
hasza weryfikacyjnego to znana jej liniowa funkcja klucza Alice x, czyli wiersz macierzy P
nad GF(2). Klucz końcowy to Tx. Ewa zna dokładnie dim(span P ∩ span T) bitów klucza końcowego
(liniowych kombinacji).

- Gdy ℓ uwzględnia wyciek (ℓ ≤ n − rank P), dla losowej T ta wiedza wynosi 0 z dużym p-stwem.
- Gdy ktoś zapomni odjąć wycieku, wymiar przekroju wynosi co najmniej ℓ − (n − rank P) > 0:
  Ewa zna te bity z pewnością, bez żadnego ataku na kanał kwantowy.

Wymaga transkryptu z `record_rows=True` w Cascade albo LDPC. Algebra na liczbach całkowitych,
więc rozsądne dla n do kilku tysięcy bitów.
"""

from dataclasses import dataclass

from qkdsim.classical import ClassicalChannel
from qkdsim.gf2 import intersection_dimension, rank, row_from_positions, rows_from_matrix
from qkdsim.postprocessing.hashing import toeplitz_matrix
from qkdsim.rng import Bits


@dataclass(frozen=True)
class LinearEveKnowledge:
    """Wiedza liniowej Ewy: rząd ujawnionych funkcji i znane bity klucza."""

    disclosed_rank: int  # rank P: niezależne liniowe funkcje klucza ujawnione w kanale
    known_final_bits: int  # dim(span P ∩ span T)


def disclosed_rows(channel: ClassicalChannel, n: int) -> list[int]:
    """Wiersze P: parzystości Cascade, syndromy LDPC i bity hasza weryfikacyjnego."""
    rows: list[int] = []
    for message in channel.messages("cascade_") + channel.messages("ldpc_"):
        if message.leaked_bits == 0:
            continue
        if message.payload is None:
            raise ValueError("uzgadnianie bez record_rows=True: brak pozycji parzystości")
        rows.extend(row_from_positions(positions, n) for positions in message.payload)
    for seed_message in channel.messages("verify_seed"):
        seed = seed_message.payload
        tag_bits = len(seed) - n + 1
        rows.extend(rows_from_matrix(toeplitz_matrix(seed, n, tag_bits)))
    return rows


def linear_eve_knowledge(channel: ClassicalChannel, n: int, pa_seed: Bits) -> LinearEveKnowledge:
    """Wiedza Ewy o kluczu końcowym Tx, gdzie T to macierz Toeplitza z ziarna `pa_seed`."""
    p_rows = disclosed_rows(channel, n)
    length = len(pa_seed) - n + 1
    t_rows = rows_from_matrix(toeplitz_matrix(pa_seed, n, length)) if length > 0 else []
    return LinearEveKnowledge(rank(p_rows), intersection_dimension(p_rows, t_rows))
