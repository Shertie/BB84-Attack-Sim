"""Atak 7: uwierzytelnianie Wegmana–Cartera i MITM (F4.11–F4.13)."""

import math

import numpy as np
import pytest

from conftest import SEED
from qkdsim.analysis.key_rate import GYS
from qkdsim.attacks.mitm import run_mitm
from qkdsim.channel import FiberChannel
from qkdsim.classical import Message
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link
from qkdsim.postprocessing.auth import (
    AuthenticatedChannel,
    AuthenticationFailure,
    KeyPool,
    WegmanCarter,
    message_bits,
)
from qkdsim.source import WCPSource

LINK = Link(
    WCPSource((0.48,)),
    FiberChannel(12.5, GYS.alpha_db_per_km, GYS.misalignment),
    DetectorPair(GYS.eta_bob, dark_count_per_detector(GYS.background)),
)


def pool(n: int = 10_000, seed: int = SEED) -> KeyPool:
    return KeyPool(np.random.default_rng(seed).integers(0, 2, n, dtype=np.uint8))


def test_key_pool_takes_and_runs_out() -> None:
    p = KeyPool(np.array([1, 0, 1, 1], dtype=np.uint8))
    assert p.take(3) == 0b101
    assert p.remaining == 1
    with pytest.raises(RuntimeError):
        p.take(2)


def test_message_serialization_is_deterministic_and_content_sensitive() -> None:
    a = Message("alice", "sample_bits", 3, payload=np.array([1, 0, 1], dtype=np.uint8))
    b = Message("alice", "sample_bits", 3, payload=np.array([1, 1, 1], dtype=np.uint8))
    assert np.array_equal(message_bits(a), message_bits(a))
    assert not np.array_equal(message_bits(a), message_bits(b))


def test_genuine_tags_verify_with_shared_key() -> None:
    sender, receiver = WegmanCarter(pool()), WegmanCarter(pool())
    for length in (1, 60, 61, 1000):
        bits = np.random.default_rng(length).integers(0, 2, length, dtype=np.uint8)
        assert receiver.verify(bits, sender.tag(bits))
    assert sender.key_bits_used == 61 * 5  # klucz haszu + 4 klucze jednorazowe


def test_hash_collisions_are_bounded_by_degree_over_p() -> None:
    """Dla m ≠ m' p-stwo h_k(m) = h_k(m') po losowym k jest ≤ L/p (L = liczba kawałków + 1)."""
    prime, trials = 127, 20_000
    rng = np.random.default_rng(SEED)
    m1, m2 = (rng.integers(0, 2, 30, dtype=np.uint8) for _ in range(2))
    collisions = 0
    for t in range(trials):
        auth = WegmanCarter(pool(64, seed=t), prime)
        collisions += auth.polynomial_hash(m1) == auth.polynomial_hash(m2)
    degree = auth.chunks(m1) + 1
    bound = degree / prime
    assert collisions / trials < bound + 4 * math.sqrt(bound / trials)


@pytest.mark.parametrize("prime", [31, 127])
def test_impersonation_succeeds_with_probability_one_over_p(prime: int) -> None:
    trials, rng = 40_000, np.random.default_rng(SEED)
    bits = rng.integers(0, 2, 20, dtype=np.uint8)
    successes = 0
    for t in range(trials):
        auth = WegmanCarter(pool(64, seed=t), prime)
        successes += auth.verify(bits, int(rng.integers(0, prime)))
    p = 1 / prime
    assert abs(successes - trials * p) < 4 * math.sqrt(trials * p * (1 - p))


def test_channel_aborts_at_first_impostor_message() -> None:
    channel = AuthenticatedChannel(WegmanCarter(pool()), mode="message", impostor="bob")
    channel.send(Message("alice", "sample_positions", 10))
    with pytest.raises(AuthenticationFailure) as failure:
        channel.send(Message("bob", "sample_bits", 10))
    assert failure.value.message_index == 1


def test_honest_transcript_mode_uses_one_tag_per_direction() -> None:
    auth = WegmanCarter(pool())
    channel = AuthenticatedChannel(auth, mode="transcript")
    for kind in ("a", "b", "c"):
        channel.send(Message("alice", kind, 1))
        channel.send(Message("bob", kind, 1))
    channel.close_round()
    assert auth.tags_used == 2


def test_mitm_without_authentication_gives_eve_both_keys() -> None:
    out = run_mitm(LINK, LINK, 1_000_000, SEED)
    assert not out.aborted
    keys = out.keys()
    assert keys is not None
    alice_key, bob_key = keys
    assert not np.array_equal(alice_key[: len(bob_key)], bob_key[: len(alice_key)])
    # Ewa jest „Bobem” w sesji z Alice i „Alice” w sesji z Bobem: zna oba klucze.
    assert out.alice_side.result is not None and out.bob_side.result is not None
    assert np.array_equal(out.alice_side.result.bob_key, alice_key)
    assert np.array_equal(out.bob_side.result.alice_key, bob_key)
    for side in (out.alice_side, out.bob_side):
        assert side.result is not None and side.result.balance.qber_sample < 0.05


@pytest.mark.parametrize("mode", ["message", "transcript"])
def test_mitm_with_wegman_carter_is_detected(mode: str) -> None:
    out = run_mitm(LINK, LINK, 1_000_000, SEED, pre_shared_bits=1_000_000, mode=mode)  # type: ignore[arg-type]
    assert out.aborted and out.keys() is None
    failure = out.alice_side.failure
    assert failure is not None
    assert failure.kind.startswith("transkrypt bob" if mode == "transcript" else "sample_bits")
