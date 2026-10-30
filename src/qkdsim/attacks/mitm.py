"""Atak 7: man-in-the-middle na nieuwierzytelniony kanał klasyczny (F4.11, F4.13).

Ewa przecina światłowód i prowadzi dwa niezależne BB84: z Alice (udając Boba) i z Bobem
(udając Alice). Fizyka nie ma tu nic do wykrycia: oba odcinki są uczciwymi łączami, więc
QBER na każdym jest taki jak bez ataku. Na końcu Alice ma klucz K_A, Bob klucz K_B ≠ K_A,
a Ewa zna oba i może przeszyfrowywać wiadomości.

Obrona: uwierzytelnianie Wegmana–Cartera kluczem, który Alice i Bob współdzielą, a Ewa nie.
W sesji z Alice Ewa musi podrobić tagi Boba (i odwrotnie), co udaje się z p-stwem ≤ L/p
na wiadomość. Atak wychodzi przy pierwszej uwierzytelnionej wiadomości Ewy.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from qkdsim.classical import ClassicalChannel
from qkdsim.pipeline import Link, run_sifted
from qkdsim.postprocessing.auth import (
    MERSENNE_61,
    AuthenticatedChannel,
    AuthenticationFailure,
    Authenticator,
    KeyPool,
    WegmanCarter,
)
from qkdsim.postprocessing.run import PostprocessingResult, postprocess
from qkdsim.rng import Bits, RandomStreams


@dataclass(frozen=True)
class SessionOutcome:
    """Jedna z dwóch sesji BB84 prowadzonych przez Ewę i jej koszt uwierzytelniania."""

    result: PostprocessingResult | None  # None, gdy uwierzytelnianie przerwało protokół
    failure: AuthenticationFailure | None
    key_bits_used: int  # klucz uwierzytelniania zużyty w tej sesji
    tags: int
    bytes_sent: int = 0  # tagi i podpisy na łączu klasycznym


@dataclass(frozen=True)
class MITMOutcome:
    """Obie sesje ataku MITM; `aborted`, gdy któraś strona odrzuciła tag Ewy."""

    alice_side: SessionOutcome  # Alice ↔ Ewa udająca Boba
    bob_side: SessionOutcome  # Ewa udająca Alice ↔ Bob

    @property
    def aborted(self) -> bool:
        return self.alice_side.failure is not None or self.bob_side.failure is not None

    def keys(self) -> tuple[Bits, Bits] | None:
        """(klucz Alice, klucz Boba), gdy obie sesje się udały."""
        a, b = self.alice_side.result, self.bob_side.result
        return None if a is None or b is None else (a.alice_key, b.bob_key)


def _session(
    link: Link,
    n_pulses: int,
    seed: np.random.SeedSequence,
    factory: Callable[[], Authenticator] | None,
    impostor: str,
    mode: Literal["message", "transcript"],
) -> SessionOutcome:
    streams = RandomStreams.from_seed(seed)
    key = run_sifted(link, n_pulses, streams)
    auth = None if factory is None else factory()
    channel: ClassicalChannel = (
        ClassicalChannel()
        if auth is None
        else AuthenticatedChannel(auth, mode=mode, impostor=impostor, forger=streams.eve)
    )
    try:
        result = postprocess(key, streams.public, channel=channel)
        failure = None
    except AuthenticationFailure as error:
        result, failure = None, error
    if auth is None:
        return SessionOutcome(result, failure, 0, 0)
    return SessionOutcome(result, failure, auth.key_bits_used, auth.tags_used, auth.bytes_sent)


def run_mitm(
    link_alice_eve: Link,
    link_eve_bob: Link,
    n_pulses: int,
    seed: int,
    pre_shared_bits: int | None = None,
    mode: Literal["message", "transcript"] = "transcript",
    prime: int = MERSENNE_61,
    authenticator: Callable[[], Authenticator] | None = None,
) -> MITMOutcome:
    """Bez uwierzytelniania, gdy `pre_shared_bits` i `authenticator` są None.

    `pre_shared_bits`: Wegman–Carter z pulą tej wielkości (każda sesja ma kopię puli).
    `authenticator`: dowolna metoda (np. ML-DSA, hybryda), tworzona osobno dla każdej sesji.
    """
    factory = authenticator
    if pre_shared_bits is not None:
        pool = np.random.default_rng([seed, 7]).integers(0, 2, pre_shared_bits, dtype=np.uint8)

        def factory() -> Authenticator:
            return WegmanCarter(KeyPool(pool.copy()), prime)

    alice_side = _session(
        link_alice_eve, n_pulses, np.random.SeedSequence([seed, 1]), factory, "bob", mode
    )
    bob_side = _session(
        link_eve_bob, n_pulses, np.random.SeedSequence([seed, 2]), factory, "alice", mode
    )
    return MITMOutcome(alice_side, bob_side)
