"""Uzgadnianie kluczy kodami LDPC z syndromem (alternatywa dla Cascade).

Alice wysyła syndrom s = Hx swojego klucza x; Bob dekoduje swój klucz y algorytmem BP
(sum-product w dziedzinie LLR) do x̂ z Hx̂ = s. Kanał to BSC(Q): LLR = ±ln((1 − Q)/Q). Jedna
wiadomość w jedną stronę zamiast setek rund Cascade; wyciek to długość syndromu m na ramkę.

- Kod: stały stopień zmiennych d_v (domyślnie 3) i stopnie kontroli skupione wokół
  d_v/(1 − R) (dwa sąsiednie stopnie), krawędzie z losowej permutacji gniazd bez krawędzi
  podwójnych (Gallager 1962; Richardson, Urbanke 2001). Cykli długości 4 nie usuwamy.
- Wiadomość z kontroli: L = (−1)^(s_j) ∏ sgn · φ(Σ φ(|L|)), φ(x) = −ln tanh(x/2) (φ = φ⁻¹).
- Klucz dzielimy na ⌈N/n_max⌉ równych ramek; m = ⌈f·n·h(Q̂)⌉ (f: zakładana sprawność).
  Ostatnią ramkę skracamy o resztę z dzielenia: brakujące bity to zera znane obu stronom.
  Ramka, której Bob nie zdekoduje, odpada (Bob ogłasza tylko jej numer); jej syndrom liczymy
  do wycieku, co jest oszacowaniem z góry.
- `density_evolution` liczy rozkład wiadomości dla n → ∞ na siatce LLR (Richardson, Urbanke
  2001), a `bp_threshold` największe Q, przy którym BP zbiega. Dla kodu (3,6) wychodzi 0,084,
  jak w literaturze.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qkdsim.analysis.entropy import h2
from qkdsim.classical import ClassicalChannel, Message
from qkdsim.rng import Bits

Positions = npt.NDArray[np.intp]
Floats = npt.NDArray[np.float64]
MAX_LLR = 50.0  # LLR bitów skróconych i obcięcie φ


@dataclass(frozen=True)
class LdpcCode:
    """Macierz H w postaci krawędzi uporządkowanych według kontroli."""

    n: int
    edge_var: Positions  # zmienna każdej krawędzi
    edge_check: Positions  # kontrola każdej krawędzi (niemalejąco)
    check_start: Positions  # indeks pierwszej krawędzi każdej kontroli

    @property
    def m(self) -> int:
        return len(self.check_start)

    @property
    def rate(self) -> float:
        return 1.0 - self.m / self.n

    def rows(self) -> list[Positions]:
        """Pozycje bitów w każdej kontroli (wiersze H)."""
        return np.split(self.edge_var, self.check_start[1:])

    def syndrome(self, bits: Bits) -> Bits:
        parities: Bits = np.bitwise_xor.reduceat(bits[self.edge_var], self.check_start)
        return parities.astype(np.uint8)


def check_degrees(rate: float, var_degree: int = 3) -> dict[int, float]:
    """Rozkład stopni kontroli (ułamki krawędzi) przy średnim stopniu d_v/(1 − R)."""
    mean = round(var_degree / (1.0 - rate), 9)  # 3/(1 − 0,7) = 9,999… a nie 10
    low = math.floor(mean)
    upper_share = mean - low  # ułamek kontroli o stopniu low + 1
    if upper_share == 0.0:
        return {low: 1.0}
    return {low: (1.0 - upper_share) * low / mean, low + 1: upper_share * (low + 1) / mean}


def random_code(n: int, m: int, rng: np.random.Generator, var_degree: int = 3) -> LdpcCode:
    """Kod o stałym stopniu zmiennych i stopniach kontroli ⌊E/m⌋ albo ⌈E/m⌉, E = n·d_v."""
    if not 0 < m < n:
        raise ValueError("wymagane 0 < m < n")
    edges = n * var_degree
    base, extra = divmod(edges, m)
    if base < 2:
        raise ValueError("za mało krawędzi na kontrolę")
    degrees = np.full(m, base, dtype=np.intp)
    degrees[:extra] += 1
    edge_check = np.repeat(np.arange(m, dtype=np.intp), degrees)
    edge_var = rng.permutation(np.repeat(np.arange(n, dtype=np.intp), var_degree))
    while True:  # usuwanie krawędzi podwójnych przez losowe zamiany
        key = edge_check * n + edge_var
        order = np.argsort(key, kind="stable")
        repeated = order[1:][key[order][1:] == key[order][:-1]]
        if len(repeated) == 0:
            break
        partners = rng.integers(0, edges, len(repeated))
        edge_var[repeated], edge_var[partners] = edge_var[partners], edge_var[repeated].copy()
    starts = np.concatenate([[0], np.cumsum(degrees)[:-1]]).astype(np.intp)
    return LdpcCode(n, edge_var, edge_check, starts)


def _phi(x: Floats) -> Floats:
    """φ(x) = −ln tanh(x/2), dla x > 0; φ(φ(x)) = x."""
    clipped = np.clip(x, 1e-12, MAX_LLR)
    result: Floats = -np.log(np.tanh(clipped / 2.0))
    return result


@dataclass(frozen=True)
class DecodeResult:
    bits: Bits
    success: bool
    iterations: int


def decode(
    code: LdpcCode,
    received: Bits,
    syndrome: Bits,
    qber: float,
    max_iterations: int = 60,
    known: npt.NDArray[np.bool_] | None = None,
) -> DecodeResult:
    """BP (sum-product) z syndromem; `known`: bity pewne (skrócone), LLR = ±MAX_LLR."""
    q = min(max(qber, 1e-6), 0.5 - 1e-9)
    channel = np.where(received == 0, 1.0, -1.0) * math.log((1.0 - q) / q)
    if known is not None:
        channel[known] = np.where(received[known] == 0, MAX_LLR, -MAX_LLR)
    flip = syndrome[code.edge_check].astype(bool)
    to_check = channel[code.edge_var]
    hard = (channel < 0).astype(np.uint8)
    if np.array_equal(code.syndrome(hard), syndrome):
        return DecodeResult(hard, True, 0)
    for iteration in range(1, max_iterations + 1):
        magnitude = _phi(np.abs(to_check))
        negative = (to_check < 0).astype(np.intp)
        total = np.add.reduceat(magnitude, code.check_start)[code.edge_check]
        parity = np.add.reduceat(negative, code.check_start)[code.edge_check] & 1
        others_negative = (parity ^ negative).astype(bool) ^ flip  # znak bez własnej krawędzi
        to_var = np.where(others_negative, -1.0, 1.0) * _phi(total - magnitude)
        posterior = channel + np.bincount(code.edge_var, weights=to_var, minlength=code.n)
        hard = (posterior < 0).astype(np.uint8)
        if np.array_equal(code.syndrome(hard), syndrome):
            return DecodeResult(hard, True, iteration)
        to_check = posterior[code.edge_var] - to_var
    return DecodeResult(hard, False, max_iterations)


@dataclass(frozen=True)
class LdpcResult:
    """Poprawiony klucz Boba, maska bitów z udanych ramek, wyciek i statystyki ramek."""

    corrected: Bits
    kept: npt.NDArray[np.bool_]
    leaked_bits: int
    frames: int
    failed_frames: int
    syndrome_bits: int  # m na pełną ramkę
    iterations: int  # suma iteracji BP


def syndrome_length(frame: int, qber_estimate: float, efficiency: float) -> int:
    """m = ⌈f·n·h(Q̂)⌉, co najmniej 1 i mniej niż n."""
    return min(frame - 1, max(1, math.ceil(efficiency * frame * h2(min(qber_estimate, 0.5)))))


def reconcile(
    alice: Bits,
    bob: Bits,
    qber_estimate: float,
    public: np.random.Generator,
    channel: ClassicalChannel,
    frame: int = 16384,
    efficiency: float = 1.6,
    max_iterations: int = 100,
    record_rows: bool = False,
) -> LdpcResult:
    """Uzgadnianie ramkami do `frame` bitów; macierz H losowana jawnie (ziarno ogłasza Alice)."""
    n = len(alice)
    if len(bob) != n or n == 0:
        raise ValueError("klucze Alice i Boba muszą być niepuste i równej długości")
    frame = math.ceil(n / math.ceil(n / frame))  # równe ramki zamiast jednej mocno skróconej
    m = syndrome_length(frame, qber_estimate, efficiency)
    code = random_code(frame, m, public)
    channel.send(Message("alice", "ldpc_code_seed", 64))
    rows = code.rows() if record_rows else None
    corrected, kept = bob.copy(), np.ones(n, dtype=np.bool_)
    frames = failed = iterations = 0
    for start in range(0, n, frame):
        stop = min(n, start + frame)
        x, y = np.zeros(frame, dtype=np.uint8), np.zeros(frame, dtype=np.uint8)
        x[: stop - start], y[: stop - start] = alice[start:stop], bob[start:stop]
        known = np.arange(frame) >= stop - start
        syndrome = code.syndrome(x)
        payload = [r[r < stop - start] + start for r in rows] if rows is not None else None
        channel.send(Message("alice", "ldpc_syndrome", m, leaked_bits=m, payload=payload))
        result = decode(code, y, syndrome, qber_estimate, max_iterations, known)
        frames += 1
        iterations += result.iterations
        if result.success:
            corrected[start:stop] = result.bits[: stop - start]
        else:
            failed += 1
            kept[start:stop] = False
            channel.send(Message("bob", "ldpc_failed_frame", 32))
    return LdpcResult(corrected, kept, channel.leaked_bits("ldpc_"), frames, failed, m, iterations)


# --- ewolucja gęstości ---------------------------------------------------------------------


def _check_table(grid: Floats) -> Positions:
    """Indeks siatki dla 2 atanh(tanh(a/2) tanh(b/2)) dla każdej pary (a, b) z siatki."""
    t = np.tanh(grid / 2.0)
    combined = 2.0 * np.arctanh(np.clip(np.outer(t, t), -1 + 1e-15, 1 - 1e-15))
    step = grid[1] - grid[0]
    index: Positions = np.rint((combined - grid[0]) / step).astype(np.intp)
    return np.clip(index, 0, len(grid) - 1).ravel()


def _box(p: Floats, q: Floats, table: Positions) -> Floats:
    out: Floats = np.bincount(table, weights=np.outer(p, q).ravel(), minlength=len(p))
    return out


def _box_power(p: Floats, k: int, table: Positions) -> Floats:
    """Gęstość wyjścia kontroli z k wejściami o gęstości p (potęgowanie przez kwadrat)."""
    result: Floats | None = None
    base = p
    while k:
        if k & 1:
            result = base if result is None else _box(result, base, table)
        k >>= 1
        if k:
            base = _box(base, base, table)
    assert result is not None
    return result


def _add(p: Floats, q: Floats) -> Floats:
    """Gęstość sumy LLR na siatce symetrycznej; masa poza siatką na krańce."""
    full = np.convolve(p, q)
    half = len(p) // 2
    out = full[half : half + len(p)].copy()
    out[0] += full[:half].sum()
    out[-1] += full[half + len(p) :].sum()
    return out


def channel_density(qber: float, grid: Floats) -> Floats:
    """LLR kanału BSC(Q) na siatce: masy ±ln((1 − Q)/Q) rozdzielone liniowo między sąsiadów."""
    density = np.zeros(len(grid))
    llr, step = math.log((1.0 - qber) / qber), grid[1] - grid[0]
    for value, weight in ((llr, 1.0 - qber), (-llr, qber)):
        position = (value - grid[0]) / step
        low = min(max(math.floor(position), 0), len(grid) - 2)
        upper = min(max(position - low, 0.0), 1.0)
        density[low] += weight * (1.0 - upper)
        density[low + 1] += weight * upper
    return density


def error_probability(density: Floats) -> float:
    """P(LLR < 0) + ½P(LLR = 0) na siatce symetrycznej."""
    half = len(density) // 2
    return float(density[:half].sum() + 0.5 * density[half])


def density_evolution(
    qber: float,
    var_degree: int,
    checks: dict[int, float],
    step: float = 0.05,
    max_llr: float = 25.0,
    max_iterations: int = 3000,
    target: float = 1e-9,
) -> tuple[bool, int, float]:
    """(zbieżność, liczba iteracji, końcowe p-stwo błędu wiadomości zmienna → kontrola)."""
    grid = (np.arange(-round(max_llr / step), round(max_llr / step) + 1) * step).astype(np.float64)
    table = _check_table(grid)
    channel = channel_density(qber, grid)
    to_check = channel
    previous = error_probability(to_check)
    for iteration in range(1, max_iterations + 1):
        from_check = np.zeros(len(grid))
        for degree, weight in checks.items():
            from_check += weight * _box_power(to_check, degree - 1, table)
        incoming = from_check
        for _ in range(var_degree - 2):
            incoming = _add(incoming, from_check)
        to_check = _add(channel, incoming)
        # Błędy zaokrągleń w masie rosną geometrycznie z iteracjami: normalizacja.
        to_check = np.clip(to_check, 0.0, None)
        to_check /= to_check.sum()
        error = error_probability(to_check)
        if error < target:
            return True, iteration, error
        if abs(previous - error) < 1e-10 * max(error, 1e-30) and iteration > 20:
            return False, iteration, error  # punkt stały z błędem > 0
        previous = error
    return False, max_iterations, previous


def bp_threshold(
    var_degree: int, checks: dict[int, float], tolerance: float = 2e-4, **kwargs: float
) -> float:
    """Największe Q, przy którym ewolucja gęstości zbiega (bisekcja)."""
    low, high = 1e-3, 0.5 - 1e-3
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if density_evolution(middle, var_degree, checks, **kwargs)[0]:  # type: ignore[arg-type]
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)
