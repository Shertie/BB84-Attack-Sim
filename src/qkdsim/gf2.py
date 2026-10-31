"""Algebra liniowa nad GF(2) na wierszach zapisanych jako liczby całkowite (bit j = kolumna j).

Do dokładnej analizy wiedzy Ewy, która zna liniowe funkcje klucza.
"""

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from qkdsim.rng import Bits


def row_from_positions(positions: npt.NDArray[np.intp], n: int) -> int:
    """Wiersz z jedynkami na podanych pozycjach."""
    dense = np.zeros(n, dtype=np.uint8)
    dense[positions] = 1
    return row_from_dense(dense)


def row_from_dense(dense: Bits) -> int:
    """Wektor bitów jako liczba całkowita: bit i to współczynnik 2^i."""
    return int.from_bytes(np.packbits(dense, bitorder="little").tobytes(), "little")


def rows_from_matrix(matrix: Bits) -> list[int]:
    """Wiersze macierzy bitów jako liczby całkowite (`row_from_dense`)."""
    return [row_from_dense(row) for row in matrix]


def rank(rows: Iterable[int]) -> int:
    """Rząd zbioru wierszy (baza XOR indeksowana najwyższym bitem)."""
    basis: dict[int, int] = {}
    for row in rows:
        while row:
            top = row.bit_length() - 1
            if top not in basis:
                basis[top] = row
                break
            row ^= basis[top]
    return len(basis)


def intersection_dimension(a: list[int], b: list[int]) -> int:
    """dim(span A ∩ span B) = rank A + rank B − rank [A; B]."""
    return rank(a) + rank(b) - rank(a + b)
