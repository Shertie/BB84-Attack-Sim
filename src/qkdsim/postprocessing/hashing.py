"""Haszowanie macierzą Toeplitza: weryfikacja kluczy i wzmacnianie prywatności.

Macierz T (m × n) ma stałe przekątne: T[i, j] = s[i − j + n − 1] dla ziarna s długości n + m − 1.
Rodzina {T} jest 2-uniwersalna: dla x ≠ y P[Tx = Ty] = 2⁻ᵐ.

Tx (mod 2) to fragment splotu s * x:
- dla małego m: m iloczynów skalarnych na bitach (`direct`);
- dla dużego m: splot przez FFT (`fft`) na float64. Wartości splotu są liczbami całkowitymi
  ≤ n, a błąd zaokrąglenia FFT rzędu 10⁻¹⁶·log₂(N)·n jest dla n ≤ 10⁸ o wiele rzędów
  mniejszy niż 0,5, więc zaokrąglenie do liczby całkowitej jest dokładne.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import fft

from qkdsim.rng import Bits

DIRECT_MAX_ROWS = 256


def random_seed(public: np.random.Generator, n: int, m: int) -> Bits:
    """Ziarno macierzy Toeplitza m × n, losowane jawnie."""
    return public.integers(0, 2, size=n + m - 1, dtype=np.uint8)


def toeplitz_matrix(seed: Bits, n: int, m: int) -> Bits:
    """Jawna macierz m × n (tylko do testów i analizy GF(2) na małych n)."""
    _check(seed, n, m)
    windows = np.lib.stride_tricks.sliding_window_view(seed, n)[:m]
    return np.ascontiguousarray(windows[:, ::-1])


def toeplitz_hash(
    bits: Bits, seed: Bits, m: int, method: Literal["auto", "direct", "fft"] = "auto"
) -> Bits:
    """y = T x (mod 2), y ma m bitów."""
    n = len(bits)
    _check(seed, n, m)
    if m == 0:
        return np.zeros(0, dtype=np.uint8)
    if method == "direct" or (method == "auto" and m <= DIRECT_MAX_ROWS):
        reversed_bits = bits[::-1]
        return np.fromiter(
            (int(np.count_nonzero(seed[i : i + n] & reversed_bits)) & 1 for i in range(m)),
            dtype=np.uint8,
            count=m,
        )
    size = fft.next_fast_len(len(seed) + n - 1, real=True)
    spectrum = fft.rfft(seed.astype(np.float64), size) * fft.rfft(bits.astype(np.float64), size)
    convolution: npt.NDArray[np.float64] = fft.irfft(spectrum, size)
    window = convolution[n - 1 : n - 1 + m]
    return (np.rint(window).astype(np.int64) & 1).astype(np.uint8)


def _check(seed: Bits, n: int, m: int) -> None:
    if n <= 0 or m < 0:
        raise ValueError("wymagane n > 0 i m ≥ 0")
    if len(seed) != n + m - 1:
        raise ValueError(f"ziarno musi mieć n + m − 1 = {n + m - 1} bitów, ma {len(seed)}")
