"""Cały postprocessing z bilansem klucza (F2.9).

klucz po siftingu → próbka i QBER → Cascade → weryfikacja → wzmacnianie prywatności → ℓ
"""

from dataclasses import dataclass, field

import numpy as np

from qkdsim.analysis.entropy import h2
from qkdsim.classical import ClassicalChannel
from qkdsim.postprocessing.amplification import key_length, privacy_amplification, verify
from qkdsim.postprocessing.cascade import cascade
from qkdsim.postprocessing.estimation import estimate_random_sample
from qkdsim.protocol import SiftedKey
from qkdsim.rng import Bits


@dataclass(frozen=True)
class PostprocessingConfig:
    sample_fraction: float = 0.1
    cascade_passes: int = 4
    tag_bits: int = 64
    # Model długości klucza (amplification.key_length). None → błąd fazowy = estymowany QBER.
    single_photon_fraction: float = 1.0
    phase_error: float | None = None
    # False tylko do demonstracji błędu „zapomniany leak_EC” (F2.8).
    subtract_leakage: bool = True
    record_rows: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.sample_fraction < 1.0:
            raise ValueError("sample_fraction musi być w (0, 1)")


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
    residual_errors: int  # błędy po Cascade, przed weryfikacją
    verified: bool
    leak_verification: int
    final_length: int

    @property
    def efficiency(self) -> float:
        """f = leak_EC / (n·h(Q)), Q = prawdziwy odsetek błędów przed korekcją."""
        q = self.errors_before_ec / self.reconciled
        return self.leak_ec / (self.reconciled * h2(q)) if q > 0 else float("inf")


@dataclass(frozen=True)
class PostprocessingResult:
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
    channel = ClassicalChannel() if channel is None else channel
    sample_size = max(1, round(config.sample_fraction * len(key)))
    estimate = estimate_random_sample(key, sample_size, public, channel)
    rest = estimate.key
    alice, bob = rest.alice, rest.bob

    ec = cascade(
        alice,
        bob,
        estimate.qber.value,
        public,
        channel,
        passes=config.cascade_passes,
        record_rows=config.record_rows,
    )
    residual = int(np.count_nonzero(alice != ec.corrected))
    check = verify(alice, ec.corrected, public, channel, config.tag_bits)

    phase_error = estimate.qber.value if config.phase_error is None else config.phase_error
    leaked = channel.leaked_bits() if config.subtract_leakage else 0
    length = key_length(len(alice), phase_error, leaked, config.single_photon_fraction)
    if not check.passed:
        length = 0  # weryfikacja nieudana: klucz odrzucony
    pa = privacy_amplification(alice, ec.corrected, length, public, channel)
    channel.close_round()  # uwierzytelnienie transkryptu przed użyciem klucza (F4.12)

    balance = KeyBalance(
        sifted=len(key),
        sample=sample_size,
        qber_sample=estimate.qber.value,
        reconciled=len(alice),
        errors_before_ec=int(np.count_nonzero(alice != bob)),
        leak_ec=ec.leaked_bits,
        corrections=ec.corrections,
        residual_errors=residual,
        verified=check.passed,
        leak_verification=check.tag_bits,
        final_length=length,
    )
    return PostprocessingResult(balance, pa.alice, pa.bob, pa.seed, alice, channel)
