"""Kwantowy generator losowy na dzielniku wiązki (przedłużenie ataku 8).

Model (Jennewein i in. 2000; przegląd: Herrero-Collantes, Garcia-Escartin 2017): w każdym takcie
laser daje stan koherentny o średniej μ, dzielnik o transmisji t kieruje go na detektory D0 i D1
(sprawności η₀, η₁, ciemne zliczenia p_d, opcjonalnie czas martwy i afterpulsy z
`detector.with_memory`). Bit surowy to 0, gdy klika tylko D0, i 1, gdy tylko D1; takty bez
kliknięcia i z podwójnym kliknięciem odpadają.

Stan koherentny rozdziela się na dwa niezależne stany koherentne, a pamięć każdego detektora
zależy tylko od jego własnych kliknięć, więc procesy kliknięć D0 i D1 są niezależne. Stąd
dokładne obciążenie także z pamięcią: P(1) = c₁(1 − c₀)/(c₀(1 − c₁) + c₁(1 − c₀)), gdzie c_i to
stacjonarne p-stwo kliknięcia (`detector.stationary_click_rate`). Korelacje kolejnych bitów:
afterpuls powtarza poprzedni bit, czas martwy go odwraca.

Postprocessing:
- `min_entropy`: oszacowanie min-entropii na bit metodą najczęstszej wartości z NIST SP 800-90B
  (rozdz. 6.3.1, górna granica ufności 99%), warunkowo na k poprzednich bitach, w najgorszym
  kontekście (bezpieczne dla łańcucha Markowa rzędu k);
- `extract`: macierz Toeplitza skraca n surowych bitów do m = ⌊nH − 2 log₂(1/ε)⌋ (lemat
  o resztkowym haszowaniu). Ziarno jest jedno dla wszystkich bloków (silny ekstraktor; ε
  sumuje się po blokach).

`QrngSource` to RandomSource Alice z wyjścia generatora, po ekstrakcji albo surowego.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from qkdsim.detector import stationary_click_rate, with_memory
from qkdsim.postprocessing.hashing import random_seed, toeplitz_hash
from qkdsim.rng import Bits


@dataclass(frozen=True)
class BeamSplitterQrng:
    """Generator na dzielniku wiązki; `efficiency` = (η₀, η₁), η = 0 to detektor oślepiony."""

    mean_photons: float
    transmission: float = 0.5
    efficiency: tuple[float, float] = (0.1, 0.1)
    dark_count: float = 1e-6
    dead_time_gates: int = 0
    afterpulse: float = 0.0

    def __post_init__(self) -> None:
        if self.mean_photons <= 0.0 or not 0.0 <= self.transmission <= 1.0:
            raise ValueError("wymagane μ > 0 i t w [0, 1]")
        if not all(0.0 <= eta <= 1.0 for eta in self.efficiency):
            raise ValueError("sprawności muszą być w [0, 1]")

    def fresh_click_rates(self) -> tuple[float, float]:
        """P-stwa kliknięcia D0 i D1 bez pamięci: 1 − (1 − p_d) e^(−μ t_i η_i)."""
        shares = (self.transmission, 1.0 - self.transmission)
        return (
            1.0
            - (1.0 - self.dark_count)
            * math.exp(-self.mean_photons * shares[0] * self.efficiency[0]),
            1.0
            - (1.0 - self.dark_count)
            * math.exp(-self.mean_photons * shares[1] * self.efficiency[1]),
        )

    def click_rates(self) -> tuple[float, float]:
        """Stacjonarne p-stwa kliknięcia z czasem martwym i afterpulsami."""
        f0, f1 = self.fresh_click_rates()
        return (
            stationary_click_rate(f0, self.dead_time_gates, self.afterpulse) if f0 > 0 else 0.0,
            stationary_click_rate(f1, self.dead_time_gates, self.afterpulse) if f1 > 0 else 0.0,
        )

    def bit_rate(self) -> float:
        """Surowe bity na takt: P(tylko D0) + P(tylko D1)."""
        c0, c1 = self.click_rates()
        return c0 * (1.0 - c1) + c1 * (1.0 - c0)

    def bias(self) -> float:
        """P(bit = 1) w stanie ustalonym."""
        c0, c1 = self.click_rates()
        return c1 * (1.0 - c0) / self.bit_rate()

    def raw_bits(self, slots: int, rng: np.random.Generator, batch_size: int = 2**22) -> Bits:
        """Surowe bity z `slots` taktów; pamięć detektorów działa w obrębie paczki."""
        f0, f1 = self.fresh_click_rates()
        memory = self.dead_time_gates > 0 or self.afterpulse > 0.0
        parts, done = [], 0
        while done < slots:
            size = min(batch_size, slots - done)
            click0, click1 = rng.random(size) < f0, rng.random(size) < f1
            if memory:
                click0 = with_memory(click0, self.dead_time_gates, self.afterpulse, rng)
                click1 = with_memory(click1, self.dead_time_gates, self.afterpulse, rng)
            parts.append(click1[click0 ^ click1].astype(np.uint8))
            done += size
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)


def min_entropy(bits: Bits, context: int = 0, confidence: float = 0.99) -> float:
    """−log₂ p_max: najczęstsza wartość z górną granicą ufności, najgorszy z 2ᵏ kontekstów.

    p_u = min(1, p̂ + z√(p̂(1 − p̂)/(n − 1))), z = Φ⁻¹(`confidence`) (NIST SP 800-90B, 6.3.1).
    """
    z = float(norm.ppf(confidence))
    if context == 0:
        labels = np.zeros(len(bits), dtype=np.int64)
        values = bits.astype(np.int64)
    else:
        windows = np.lib.stride_tricks.sliding_window_view(bits.astype(np.int64), context + 1)
        labels = windows[:, :-1] @ (1 << np.arange(context, dtype=np.int64))
        values = windows[:, -1]
    totals = np.bincount(labels, minlength=2**context)
    ones = np.bincount(labels, weights=values, minlength=2**context)
    worst = 0.0
    for n, k in zip(totals, ones, strict=True):
        if n < 2:
            return 0.0  # kontekst bez danych: nic nie wiadomo
        p_hat = max(k, n - k) / n
        worst = max(worst, min(1.0, p_hat + z * math.sqrt(p_hat * (1.0 - p_hat) / (n - 1))))
    return -math.log2(worst)


def output_length(raw_length: int, entropy_per_bit: float, epsilon: float) -> int:
    """m = ⌊nH − 2 log₂(1/ε)⌋ (lemat o resztkowym haszowaniu), nie mniej niż 0."""
    return max(0, math.floor(raw_length * entropy_per_bit - 2.0 * math.log2(1.0 / epsilon)))


def extract(raw: Bits, seed: Bits, block: int, out_per_block: int) -> Bits:
    """Ekstrakcja Toeplitzem pełnych bloków `block` bitów; reszta odpada."""
    blocks = len(raw) // block
    parts = [
        toeplitz_hash(raw[i * block : (i + 1) * block], seed, out_per_block, method="fft")
        for i in range(blocks)
    ]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)


class QrngSource:
    """RandomSource z generatora `device`: bity po ekstrakcji (`entropy_per_bit` < 1) albo surowe.

    `entropy_per_bit` pochodzi z kalibracji (`min_entropy` na osobnej próbce); 1.0 wyłącza
    ekstrakcję. Wybory intensywności biorą 32 bity na impuls.
    """

    def __init__(
        self,
        device: BeamSplitterQrng,
        nature: np.random.Generator,
        public: np.random.Generator,
        entropy_per_bit: float,
        block: int = 2**16,
        epsilon: float = 1e-10,
    ):
        self._device, self._nature = device, nature
        self.raw_used = 0
        self._block = block
        self._extracting = entropy_per_bit < 1.0
        self.out_per_block = (
            output_length(block, entropy_per_bit, epsilon) if self._extracting else block
        )
        if self.out_per_block == 0:
            raise ValueError("za mało min-entropii na blok; zwiększ blok")
        self._seed = random_seed(public, block, self.out_per_block)
        self._raw = np.zeros(0, dtype=np.uint8)
        self._ready = np.zeros(0, dtype=np.uint8)

    def _take(self, n: int) -> Bits:
        while len(self._ready) < n:
            missing = n - len(self._ready)
            blocks = math.ceil(missing / self.out_per_block)
            need = blocks * self._block
            while len(self._raw) < need:
                slots = math.ceil(1.1 * (need - len(self._raw)) / self._device.bit_rate()) + 1024
                self._raw = np.concatenate([self._raw, self._device.raw_bits(slots, self._nature)])
            chunk, self._raw = self._raw[:need], self._raw[need:]
            self.raw_used += need
            fresh = (
                extract(chunk, self._seed, self._block, self.out_per_block)
                if self._extracting
                else chunk
            )
            self._ready = np.concatenate([self._ready, fresh])
        out, self._ready = self._ready[:n], self._ready[n:]
        return out

    def bits(self, n: int) -> Bits:
        return self._take(n)

    def bases(self, n: int) -> Bits:
        return self._take(n)

    def choice(self, n: int, p: Sequence[float]) -> Bits:
        words = self._take(32 * n).reshape(n, 32).astype(np.uint64)
        uniform = (words << np.arange(32, dtype=np.uint64)).sum(axis=1) / 2.0**32
        edges = np.cumsum(p)[:-1]
        picks: Bits = np.searchsorted(edges, uniform, side="right").astype(np.uint8)
        return picks
