"""Cały postprocessing z bilansem klucza.

klucz po siftingu → próbka i QBER → Cascade albo LDPC → weryfikacja → wzmacnianie prywatności → ℓ

Przy LDPC ramki, których Bob nie zdekodował, odpadają z klucza obu stron przed weryfikacją.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from qkdsim.analysis.entropy import h2
from qkdsim.classical import ClassicalChannel
from qkdsim.postprocessing.amplification import key_length, privacy_amplification, verify
from qkdsim.postprocessing.cascade import cascade
from qkdsim.postprocessing.estimation import estimate_random_sample
from qkdsim.postprocessing.ldpc import reconcile
from qkdsim.protocol import SiftedKey
from qkdsim.rng import Bits


@dataclass(frozen=True)
class PostprocessingConfig:
    """Parametry postprocessingu: próbka, Cascade, weryfikacja, model ℓ."""

    sample_fraction: float = 0.1
    reconciliation: Literal["cascade", "ldpc"] = "cascade"
    cascade_passes: int = 4
    ldpc_frame: int = 16384
    ldpc_efficiency: float = 1.6  # m = ⌈f·n·h(Q̂)⌉ na ramkę
    ldpc_iterations: int = 100
    tag_bits: int = 64
    # Model długości klucza (amplification.key_length). None → błąd fazowy = estymowany QBER.
    single_photon_fraction: float = 1.0
    phase_error: float | None = None
    # False tylko do demonstracji błędu „zapomniany leak_EC”.
    subtract_leakage: bool = True
    record_rows: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.sample_fraction < 1.0:
            raise ValueError("sample_fraction musi być w (0, 1)")
        if self.reconciliation not in ("cascade", "ldpc"):
            raise ValueError(f"nieznane uzgadnianie {self.reconciliation!r}")


@dataclass(frozen=True)
class KeyBalance:
    """Ile bitów zostaje na każdym kroku i dokąd odpływa reszta."""

    sifted: int
    sample: int
    qber_sample: float
    reconciled: int  # długość klucza po usunięciu próbki
    errors_before_ec: int
    leak_ec: int
    corrections: int
    residual_errors: int  # błędy po korekcji, przed weryfikacją
    verified: bool
    leak_verification: int
    final_length: int
    discarded: int = 0  # bity z ramek LDPC, których Bob nie zdekodował

    @property
    def efficiency(self) -> float:
        """f = leak_EC / (n·h(Q)), Q = prawdziwy odsetek błędów przed korekcją."""
        q = self.errors_before_ec / self.reconciled
        return self.leak_ec / (self.reconciled * h2(q)) if q > 0 else float("inf")


@dataclass(frozen=True)
class PostprocessingResult:
    """Klucze końcowe, bilans bitów, ziarno PA i kanał z transkryptem."""

    balance: KeyBalance
    alice_key: Bits
    bob_key: Bits
    pa_seed: Bits
    reconciled_alice: Bits  # klucz Alice po usunięciu próbki (do analizy wiedzy Ewy)
    channel: ClassicalChannel = field(repr=False)


def postprocess(
    key: SiftedKey,
    public: np.random.Generator,
    config: PostprocessingConfig = PostprocessingConfig(),  # noqa: B008
    channel: ClassicalChannel | None = None,
) -> PostprocessingResult:
    """Estymacja QBER, Cascade albo LDPC, weryfikacja i wzmacnianie prywatności."""
    channel = ClassicalChannel() if channel is None else channel
    sample_size = max(1, round(config.sample_fraction * len(key)))
    estimate = estimate_random_sample(key, sample_size, public, channel)
    rest = estimate.key
    alice, bob = rest.alice, rest.bob
    errors_before = int(np.count_nonzero(alice != bob))

    if config.reconciliation == "cascade":
        ec = cascade(
            alice,
            bob,
            estimate.qber.value,
            public,
            channel,
            passes=config.cascade_passes,
            record_rows=config.record_rows,
        )
        corrected, leak_ec, corrections, discarded = ec.corrected, ec.leaked_bits, ec.corrections, 0
    else:
        ldpc = reconcile(
            alice,
            bob,
            estimate.qber.value,
            public,
            channel,
            frame=config.ldpc_frame,
            efficiency=config.ldpc_efficiency,
            max_iterations=config.ldpc_iterations,
            record_rows=config.record_rows,
        )
        if config.record_rows and ldpc.failed_frames:
            raise ValueError("analiza liniowej Ewy wymaga zdekodowania wszystkich ramek LDPC")
        corrections = int(np.count_nonzero((ldpc.corrected != bob) & ldpc.kept))
        alice, corrected = alice[ldpc.kept], ldpc.corrected[ldpc.kept]
        leak_ec, discarded = ldpc.leaked_bits, int(np.count_nonzero(~ldpc.kept))
    residual = int(np.count_nonzero(alice != corrected))
    check = verify(alice, corrected, public, channel, config.tag_bits)

    phase_error = estimate.qber.value if config.phase_error is None else config.phase_error
    leaked = channel.leaked_bits() if config.subtract_leakage else 0
    length = key_length(len(alice), phase_error, leaked, config.single_photon_fraction)
    if not check.passed:
        length = 0  # weryfikacja nieudana: klucz odrzucony
    pa = privacy_amplification(alice, corrected, length, public, channel)
    channel.close_round()  # uwierzytelnienie transkryptu przed użyciem klucza

    balance = KeyBalance(
        sifted=len(key),
        sample=sample_size,
        qber_sample=estimate.qber.value,
        reconciled=len(alice),
        errors_before_ec=errors_before,
        leak_ec=leak_ec,
        corrections=corrections,
        residual_errors=residual,
        verified=check.passed,
        leak_verification=check.tag_bits,
        final_length=length,
        discarded=discarded,
    )
    return PostprocessingResult(balance, pa.alice, pa.bob, pa.seed, alice, channel)
