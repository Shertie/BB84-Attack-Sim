"""Atak 4: time-shift (Qi et al. 2007, eksperyment Zhao et al. 2008).

Ewa niczego nie mierzy i nie przygotowuje. Przesuwa każdy impuls w czasie: z p-stwem p_A do
chwili A, w której sprawniejszy jest detektor 0, inaczej do chwili B, w której sprawniejszy jest
detektor 1. Bob częściej wykrywa wtedy bit zgodny z przesunięciem, więc po siftingu Ewa zgaduje
bit z własnego wyboru. p_A dobiera tak, by zera i jedynki Boba były zrównoważone, a spadek
sprawności przy przesunięciu kompensuje linią bezstratną i tłumieniem `transmission`.

W `EveLog` atak zapisuje: `measured` = True (Ewa przesunęła każdy impuls), `basis` = NO_BASIS
(nie mierzyła), `bit` = indeks przesunięcia, czyli jej zgadywany bit (A → 0, B → 1).
Model analityczny: analysis/time_shift.py.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qkdsim.eve import NO_BASIS, EveLog
from qkdsim.pipeline import DEFAULT_BATCH_SIZE, Link, simulate
from qkdsim.pulses import Pulses
from qkdsim.rng import RandomStreams


@dataclass(frozen=True)
class TimeShift:
    """Atak 4: Ewa przesuwa impuls do chwili A (p-stwo `p_a`) albo B; niczego nie mierzy."""

    shift_a_ns: float  # przesunięcie, przy którym przeważa detektor 0
    shift_b_ns: float  # przesunięcie, przy którym przeważa detektor 1
    p_a: float
    transmission: float = 1.0  # tłumienie Ewy, którym wyrównuje zysk Boba
    position: float = 0.0  # Ewa tuż za Alice...
    lossless_after: bool = True  # ...i bezstratna linia, która kompensuje spadek sprawności

    def __post_init__(self) -> None:
        for name in ("p_a", "transmission"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} musi być w [0, 1]")

    def intercept(self, pulses: Pulses, rng: np.random.Generator) -> tuple[Pulses, EveLog]:
        size = len(pulses)
        early = rng.random(size) < self.p_a
        n = pulses.n if self.transmission == 1.0 else rng.binomial(pulses.n, self.transmission)
        shifted = pulses.replace(
            n=n, t=pulses.t + np.where(early, self.shift_a_ns, self.shift_b_ns)
        )
        log = EveLog(
            measured=np.ones(size, dtype=np.bool_),
            basis=np.full(size, NO_BASIS, dtype=np.uint8),
            bit=(~early).astype(np.uint8),
        )
        return shifted, log


def detection_table(
    link: Link, n_pulses: int, streams: RandomStreams, batch_size: int = DEFAULT_BATCH_SIZE
) -> tuple[npt.NDArray[np.int64], int]:
    """Detekcje w zgodnej bazie [przesunięcie Ewy, baza, bit Alice, bit Boba] i liczba Ñ impulsów
    w zgodnej bazie, liczone w paczkach. Bez Ewy wszystkie detekcje trafiają do przesunięcia 0.
    """
    table = np.zeros(16, dtype=np.int64)
    matched = done = 0
    while done < n_pulses:
        size = min(batch_size, n_pulses - done)
        alice, bob, eve = simulate(link, size, streams)
        same = alice.bases == bob.bases
        matched += int(np.count_nonzero(same))
        keep = same & bob.detected
        cell = (
            (eve.bit[keep].astype(np.intp) * 2 + alice.bases[keep]) * 2 + alice.bits[keep]
        ) * 2 + bob.bits[keep]
        table += np.bincount(cell, minlength=16).astype(np.int64)
        done += size
    return table.reshape(2, 2, 2, 2), matched
