"""Bramka G2 (ROADMAP, rozdz. 6): poprawność i sprawność postprocessingu.

Wolne testy (`make test-slow`).
"""

import numpy as np
import pytest

from conftest import SEED
from qkdsim.analysis.entropy import h2
from qkdsim.classical import ClassicalChannel
from qkdsim.postprocessing.cascade import cascade
from qkdsim.postprocessing.run import postprocess
from qkdsim.protocol import SiftedKey


@pytest.mark.slow
def test_no_undetected_mismatch_in_1000_runs() -> None:
    """1000 przebiegów, n = 10⁴, Q = 3%: po weryfikacji nigdy nie zostają różne klucze."""
    rng = np.random.default_rng(SEED)
    n, qber, runs = 10_000, 0.03, 1000
    failed = 0
    for _ in range(runs):
        alice = rng.integers(0, 2, n, dtype=np.uint8)
        bob = alice ^ (rng.random(n) < qber).astype(np.uint8)
        result = postprocess(SiftedKey(np.arange(n), alice, bob), rng)
        if result.balance.verified:
            assert np.array_equal(result.alice_key, result.bob_key)
            assert result.balance.residual_errors == 0
        else:
            failed += 1
    # Odsetek nieudanych weryfikacji (błędy resztkowe po 4 przebiegach) jest mały.
    assert failed / runs < 0.01


@pytest.mark.slow
@pytest.mark.parametrize("qber", [0.01, 0.02, 0.03, 0.04, 0.05])
def test_cascade_efficiency_in_target_range(qber: float) -> None:
    """Średnie f(Q) w [1,05; 1,25] dla Q ∈ [1%, 5%], n = 10⁵."""
    rng = np.random.default_rng([SEED, int(1000 * qber)])
    n, runs = 100_000, 10
    values = []
    for _ in range(runs):
        alice = rng.integers(0, 2, n, dtype=np.uint8)
        bob = alice ^ (rng.random(n) < qber).astype(np.uint8)
        q = float(np.mean(alice != bob))
        result = cascade(alice, bob, q, rng, ClassicalChannel())
        assert np.array_equal(result.corrected, alice)
        values.append(result.leaked_bits / (n * h2(q)))
    assert 1.05 <= float(np.mean(values)) <= 1.25
