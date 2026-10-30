"""Części postprocessingu: kanał, estymacja, Cascade, hasze, GF(2) (F2.1–F2.7)."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from conftest import SEED
from qkdsim.analysis.entropy import h2
from qkdsim.classical import ClassicalChannel, Message
from qkdsim.gf2 import intersection_dimension, rank, row_from_positions, rows_from_matrix
from qkdsim.postprocessing.amplification import key_length, privacy_amplification, verify
from qkdsim.postprocessing.cascade import cascade, initial_block_size
from qkdsim.postprocessing.estimation import estimate_by_basis, estimate_random_sample
from qkdsim.postprocessing.hashing import random_seed, toeplitz_hash, toeplitz_matrix
from qkdsim.protocol import SiftedKey
from qkdsim.rng import Bits


def synthetic_key(n: int, qber: float, seed: int = SEED) -> SiftedKey:
    rng = np.random.default_rng(seed)
    alice = rng.integers(0, 2, n, dtype=np.uint8)
    bob = alice ^ (rng.random(n) < qber).astype(np.uint8)
    return SiftedKey(index=np.arange(n), alice=alice, bob=bob)


# --- kanał klasyczny ------------------------------------------------------------------


def test_channel_sums_leak_by_prefix_and_applies_hooks_in_order() -> None:
    seen: list[str] = []

    def tag(name: str) -> Callable[[Message], Message]:
        def hook(message: Message) -> Message:
            seen.append(name)
            return message

        return hook

    channel = ClassicalChannel([tag("a"), tag("b")])
    channel.send(Message("alice", "cascade_parities", 10, leaked_bits=10))
    channel.send(Message("alice", "cascade_binary", 3, leaked_bits=3))
    channel.send(Message("bob", "sample_bits", 50))
    assert channel.leaked_bits() == 13
    assert channel.leaked_bits("cascade_binary") == 3
    assert channel.size_bits("sample") == 50
    assert seen == ["a", "b"] * 3


def test_message_validation() -> None:
    with pytest.raises(ValueError):
        Message("eve", "x", 1)
    with pytest.raises(ValueError):
        Message("alice", "x", -1)


# --- estymacja ------------------------------------------------------------------------


def test_random_sample_is_removed_and_leaks_nothing() -> None:
    key = synthetic_key(10_000, 0.05)
    channel = ClassicalChannel()
    result = estimate_random_sample(key, 1000, np.random.default_rng(SEED), channel)
    assert len(result.key) == 9000
    assert result.qber.trials == 1000
    sample_positions = channel.messages("sample_positions")[0].payload
    assert not np.intersect1d(sample_positions, result.key.index).size
    assert channel.leaked_bits() == 0
    assert abs(result.qber.z_score(0.05)) < 4


def test_estimate_by_basis_uses_whole_test_basis() -> None:
    key = synthetic_key(1000, 0.02)
    bases = (np.arange(1000) % 4 == 0).astype(np.uint8)  # 25% w bazie X
    result = estimate_by_basis(key, bases, test_basis=1, channel=ClassicalChannel())
    assert result.qber.trials == 250
    assert len(result.key) == 750


@pytest.mark.parametrize("size", [0, 100])
def test_invalid_sample_size(size: int) -> None:
    with pytest.raises(ValueError):
        estimate_random_sample(
            synthetic_key(100, 0.0), size, np.random.default_rng(0), ClassicalChannel()
        )


# --- Cascade --------------------------------------------------------------------------


def test_initial_block_size() -> None:
    assert initial_block_size(0.03, 10_000) == 25
    assert initial_block_size(0.0, 500) == 500
    assert initial_block_size(0.9, 500) == 2


@pytest.mark.parametrize("qber", [0.005, 0.02, 0.05, 0.08])
def test_cascade_corrects_key_and_accounts_leak(qber: float) -> None:
    key = synthetic_key(50_000, qber)
    channel = ClassicalChannel()
    q = float(np.mean(key.alice != key.bob))
    result = cascade(key.alice, key.bob, q, np.random.default_rng(SEED), channel)
    assert np.array_equal(result.corrected, key.alice)
    assert result.corrections == int(np.count_nonzero(key.alice != key.bob))
    assert result.leaked_bits == channel.leaked_bits("cascade")
    assert 1.0 < result.leaked_bits / (len(key) * h2(q)) < 1.3


def test_cascade_without_errors_only_announces_block_parities() -> None:
    key = synthetic_key(1000, 0.0)
    channel = ClassicalChannel()
    result = cascade(key.alice, key.bob, 0.01, np.random.default_rng(0), channel)
    assert result.corrections == 0
    assert channel.messages("cascade_binary") == []


def test_recorded_rows_reproduce_announced_parities() -> None:
    """Każdy zapisany wiersz to zbiór pozycji, których parzystość Alice ogłosiła."""
    key = synthetic_key(2000, 0.03)
    channel = ClassicalChannel()
    cascade(key.alice, key.bob, 0.03, np.random.default_rng(1), channel, record_rows=True)
    rows = [r for m in channel.messages("cascade_") if m.leaked_bits for r in m.payload]
    assert len(rows) == channel.leaked_bits("cascade")
    assert all(len(r) > 0 for r in rows)


# --- hasze Toeplitza, weryfikacja, wzmacnianie ----------------------------------------


@pytest.mark.parametrize(("n", "m"), [(1, 1), (7, 3), (64, 64), (1000, 300), (4096, 257)])
def test_toeplitz_methods_agree_with_explicit_matrix(n: int, m: int) -> None:
    rng = np.random.default_rng(n * 1000 + m)
    x = rng.integers(0, 2, n, dtype=np.uint8)
    seed = random_seed(rng, n, m)
    expected = (toeplitz_matrix(seed, n, m).astype(np.int64) @ x) % 2
    assert np.array_equal(toeplitz_hash(x, seed, m, "direct"), expected)
    assert np.array_equal(toeplitz_hash(x, seed, m, "fft"), expected)


def test_toeplitz_matrix_has_constant_diagonals() -> None:
    t = toeplitz_matrix(random_seed(np.random.default_rng(0), 20, 8), 20, 8)
    assert np.array_equal(t[1:, 1:], t[:-1, :-1])


def test_toeplitz_hash_is_linear() -> None:
    rng = np.random.default_rng(2)
    x, y = (rng.integers(0, 2, 500, dtype=np.uint8) for _ in range(2))
    seed = random_seed(rng, 500, 100)
    assert np.array_equal(
        toeplitz_hash(x ^ y, seed, 100), toeplitz_hash(x, seed, 100) ^ toeplitz_hash(y, seed, 100)
    )


def test_fft_hash_is_exact_for_large_input() -> None:
    rng = np.random.default_rng(4)
    n, m = 1_000_000, 5
    x = rng.integers(0, 2, n, dtype=np.uint8)
    seed = random_seed(rng, n, m)
    assert np.array_equal(toeplitz_hash(x, seed, m, "fft"), toeplitz_hash(x, seed, m, "direct"))


def test_verification_detects_single_bit_difference() -> None:
    key = synthetic_key(5000, 0.0)
    bob: Bits = key.bob.copy()
    bob[1234] ^= 1
    channel = ClassicalChannel()
    assert verify(key.alice, key.alice, np.random.default_rng(0), channel).passed
    assert not verify(key.alice, bob, np.random.default_rng(0), channel).passed
    assert channel.leaked_bits("verify") == 128


def test_short_tag_collision_rate_is_two_to_minus_m() -> None:
    """Dla tagu 6-bitowego różne klucze przechodzą weryfikację z p-stwem 2⁻⁶."""
    rng = np.random.default_rng(SEED)
    runs, passed = 4000, 0
    for _ in range(runs):
        a = rng.integers(0, 2, 200, dtype=np.uint8)
        b = a.copy()
        b[rng.integers(200)] ^= 1
        passed += verify(a, b, rng, ClassicalChannel(), tag_bits=6).passed
    p = 2.0**-6
    assert abs(passed - runs * p) < 4 * math.sqrt(runs * p * (1 - p))


def test_key_length_formula() -> None:
    assert key_length(10_000, 0.03, 2000) == math.floor(10_000 * (1 - h2(0.03)) - 2000)
    assert key_length(1000, 0.2, 5000) == 0
    assert key_length(1000, 0.0, 0, single_photon_fraction=0.5) == 500


def test_privacy_amplification_gives_equal_keys_of_requested_length() -> None:
    key = synthetic_key(3000, 0.0)
    out = privacy_amplification(
        key.alice, key.bob, 1234, np.random.default_rng(0), ClassicalChannel()
    )
    assert len(out.alice) == 1234
    assert np.array_equal(out.alice, out.bob)


# --- GF(2) ----------------------------------------------------------------------------


def test_gf2_rank_and_intersection() -> None:
    identity = rows_from_matrix(np.eye(8, dtype=np.uint8))
    assert rank(identity) == 8
    assert rank([0b011, 0b101, 0b110]) == 2  # trzeci = suma dwóch pierwszych
    first_half, second_half = identity[:4], identity[4:]
    assert intersection_dimension(first_half, second_half) == 0
    assert intersection_dimension(identity[:5], identity[3:]) == 2
    assert row_from_positions(np.array([0, 3]), 8) == 0b1001
