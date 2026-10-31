"""QKD ze splątanymi parami: BBM92 ze źródłem SPDC i test CHSH (Ekert 1991).

- `simulate_pairs`: źródło SPDC pośrodku, dwie niezależne pary trybów o termicznej liczbie par
  (analysis/entangled.py); każdy foton dociera z p-stwem η po swojej stronie, u Boba trafia do
  złego detektora z p-stwem e_d; detektory progowe z ciemnymi zliczeniami, podwójne kliknięcie
  daje losowy bit z generatora strony. Liczy koincydencje w zgodnych bazach i błędy po odwróceniu
  bitu Boba (stan antyskorelowany).
- `simulate_chsh`: urządzenia jak czarne skrzynki. Alice wybiera a₀ = 0, a₁ = π/4 albo ustawienie
  klucza a₂ = π/8, Bob b₀ = π/8 albo b₁ = −π/8. Para daje E(a, b) = V cos 2(a − b), detektor
  klika z p-stwem η, a brak kliknięcia czytamy jako +1. Liczniki pozwalają policzyć S na
  wszystkich rundach i na samych koincydencjach.
- `DetectorControl` (Gerhardt i in. 2011): Ewa steruje detektorami tak, że klikają tylko przy
  wybranym przez nią ustawieniu, z wybranym wynikiem. Na koincydencjach daje dowolne S ≤ 4,
  na wszystkich rundach S ≤ 2, bo jest lokalnym modelem.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qkdsim.rng import RandomStreams

ALICE_ANGLES = (0.0, math.pi / 4, math.pi / 8)  # a₀, a₁ (CHSH), a₂ (klucz)
BOB_ANGLES = (math.pi / 8, -math.pi / 8)  # b₀ (CHSH i klucz), b₁
KEY_SETTINGS = (2, 0)
CHSH_SIGNS = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): -1}


@dataclass(frozen=True)
class PairCounts:
    pulses: int
    coincidences: int
    errors: int


def _thermal(lam: float, size: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """Liczba par w jednej parze trybów: P(n) = λⁿ/(1 + λ)ⁿ⁺¹."""
    return rng.geometric(1.0 / (1.0 + lam), size).astype(np.int64) - 1


def simulate_pairs(
    lam: float,
    eta_a: float,
    eta_b: float,
    dark: float,
    misalignment: float,
    n_pulses: int,
    streams: RandomStreams,
    batch_size: int = 2**20,
) -> PairCounts:
    """BBM92 w zgodnych bazach: koincydencje i błędy na `n_pulses` impulsach pompy."""
    nature = streams.nature
    coincidences = errors = done = 0
    while done < n_pulses:
        size = min(batch_size, n_pulses - done)
        n1, n2 = _thermal(lam, size, nature), _thermal(lam, size, nature)
        alice0 = (nature.binomial(n1, eta_a) > 0) | (nature.random(size) < dark)
        alice1 = (nature.binomial(n2, eta_a) > 0) | (nature.random(size) < dark)
        # Bob: n₁ fotonów należy do jego detektora 1, n₂ do detektora 0; e_d przenosi je na drugi.
        arrived1, arrived0 = nature.binomial(n1, eta_b), nature.binomial(n2, eta_b)
        stray1, stray0 = (
            nature.binomial(arrived1, misalignment),
            nature.binomial(arrived0, misalignment),
        )
        bob0 = ((arrived0 - stray0 + stray1) > 0) | (nature.random(size) < dark)
        bob1 = ((arrived1 - stray1 + stray0) > 0) | (nature.random(size) < dark)
        both = (alice0 | alice1) & (bob0 | bob1)
        alice_bit = np.where(alice0 & alice1, streams.alice.bits(size), alice1.astype(np.uint8))
        bob_raw = np.where(bob0 & bob1, streams.bob.bits(size), bob1.astype(np.uint8))
        coincidences += int(np.count_nonzero(both))
        errors += int(np.count_nonzero(both & (bob_raw == alice_bit)))  # Bob odwraca bit
        done += size
    return PairCounts(n_pulses, coincidences, errors)


@dataclass(frozen=True)
class ChshCounts:
    """Liczniki dla par ustawień (x, y): rundy, Σ AB, koincydencje, Σ AB na koincydencjach."""

    rounds: npt.NDArray[np.int64]
    product_sum: npt.NDArray[np.int64]
    coincidences: npt.NDArray[np.int64]
    coincidence_product_sum: npt.NDArray[np.int64]

    def correlation(self, x: int, y: int, postselected: bool = False) -> float:
        if postselected:
            return float(self.coincidence_product_sum[x, y] / self.coincidences[x, y])
        return float(self.product_sum[x, y] / self.rounds[x, y])

    def chsh(self, postselected: bool = False) -> float:
        return sum(s * self.correlation(x, y, postselected) for (x, y), s in CHSH_SIGNS.items())

    def key_qber(self) -> float:
        """Odsetek rund klucza (a₂, b₀) z różnymi wynikami, braki kliknięć czytane jako +1."""
        x, y = KEY_SETTINGS
        return 0.5 * (1.0 - self.correlation(x, y))


@dataclass(frozen=True)
class DetectorControl:
    """Ewa wybiera parę ustawień CHSH i wyniki; detektor klika tylko przy jej ustawieniu.

    Z p-stwem `fidelity` wyniki dają E = ±1 zgodnie ze znakiem w S, inaczej są losowe, więc na
    koincydencjach S = 4·fidelity (2√2 przy fidelity = 1/√2).
    """

    fidelity: float = 1.0 / math.sqrt(2.0)


def simulate_chsh(
    efficiency: float,
    visibility: float,
    n_rounds: int,
    streams: RandomStreams,
    eve: DetectorControl | None = None,
) -> ChshCounts:
    """Rundy CHSH i klucza; Alice i Bob losują ustawienia ze swoich generatorów."""
    alice_rng, bob_rng, nature = streams.alice, streams.bob, streams.nature
    x = alice_rng.choice(n_rounds, (1 / 3, 1 / 3, 1 / 3)).astype(np.intp)
    y = bob_rng.choice(n_rounds, (0.5, 0.5)).astype(np.intp)
    signs_a = 1 - 2 * nature.integers(0, 2, n_rounds)
    if eve is None:
        click_a = nature.random(n_rounds) < efficiency
        click_b = nature.random(n_rounds) < efficiency
        angle = np.asarray(ALICE_ANGLES)[x] - np.asarray(BOB_ANGLES)[y]
        same = nature.random(n_rounds) < 0.5 * (1.0 + visibility * np.cos(2.0 * angle))
        signs_b = np.where(
            click_a, np.where(same, signs_a, -signs_a), 1 - 2 * nature.integers(0, 2, n_rounds)
        )
    else:
        eve_rng = streams.eve
        target_x, target_y = eve_rng.integers(0, 2, n_rounds), eve_rng.integers(0, 2, n_rounds)
        click_a, click_b = x == target_x, y == target_y
        sign = np.where((target_x == 1) & (target_y == 1), -1, 1)
        faithful = eve_rng.random(n_rounds) < eve.fidelity
        signs_b = np.where(faithful, signs_a * sign, 1 - 2 * eve_rng.integers(0, 2, n_rounds))
    a = np.where(click_a, signs_a, 1)  # brak kliknięcia czytany jako +1
    b = np.where(click_b, signs_b, 1)
    both = click_a & click_b
    cell = x * 2 + y

    def table(
        weights: npt.NDArray[np.int64] | None, mask: npt.NDArray[np.bool_] | None = None
    ) -> npt.NDArray[np.int64]:
        selected = cell if mask is None else cell[mask]
        w = None if weights is None else (weights if mask is None else weights[mask])
        return np.rint(np.bincount(selected, weights=w, minlength=6)).astype(np.int64).reshape(3, 2)

    product = (a * b).astype(np.int64)
    return ChshCounts(table(None), table(product), table(None, both), table(product, both))
