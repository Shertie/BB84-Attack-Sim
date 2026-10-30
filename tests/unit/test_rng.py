"""Źródła losowości: zakresy, determinizm i niezależność strumieni (ADR 0003)."""

import numpy as np
import pytest

from conftest import SEED
from qkdsim.rng import GeneratorSource, RandomStreams


def test_bits_and_bases_are_binary_uint8(rng: np.random.Generator) -> None:
    source = GeneratorSource(rng)
    for draw in (source.bits(10_000), source.bases(10_000)):
        assert draw.dtype == np.uint8
        assert draw.shape == (10_000,)
        assert set(np.unique(draw)) == {0, 1}


def test_choice_follows_distribution(rng: np.random.Generator) -> None:
    p = [0.7, 0.2, 0.1]
    n = 200_000
    draw = GeneratorSource(rng).choice(n, p)
    assert draw.dtype == np.uint8
    counts = np.bincount(draw, minlength=len(p))
    for k, pk in enumerate(p):
        sigma = np.sqrt(n * pk * (1 - pk))
        assert abs(counts[k] - n * pk) < 4 * sigma


def test_streams_are_deterministic() -> None:
    a = RandomStreams.from_seed(SEED)
    b = RandomStreams.from_seed(SEED)
    assert np.array_equal(a.alice.bits(1000), b.alice.bits(1000))
    assert np.array_equal(a.nature.random(1000), b.nature.random(1000))


def test_streams_differ_between_parties() -> None:
    streams = RandomStreams.from_seed(SEED)
    assert not np.array_equal(streams.alice.bases(1000), streams.bob.bases(1000))


def test_consuming_one_stream_does_not_shift_another() -> None:
    reference = RandomStreams.from_seed(SEED).alice.bits(1000)
    streams = RandomStreams.from_seed(SEED)
    streams.bob.bases(12_345)
    streams.nature.random(777)
    assert np.array_equal(streams.alice.bits(1000), reference)


@pytest.mark.parametrize("other_seed", [SEED + 1, 0])
def test_different_seeds_give_different_streams(other_seed: int) -> None:
    a = RandomStreams.from_seed(SEED).alice.bits(1000)
    b = RandomStreams.from_seed(other_seed).alice.bits(1000)
    assert not np.array_equal(a, b)


def test_public_stream_did_not_change_the_first_four() -> None:
    """Dodanie strumienia `public` (F2) nie zmienia losowań z F1: wyniki są odtwarzalne."""
    alice_seq, _, nature_seq, _ = np.random.SeedSequence(SEED).spawn(4)
    streams = RandomStreams.from_seed(SEED)
    expected_bits = np.random.default_rng(alice_seq).integers(0, 2, 1000, dtype=np.uint8)
    assert np.array_equal(streams.alice.bits(1000), expected_bits)
    assert np.array_equal(streams.nature.random(10), np.random.default_rng(nature_seq).random(10))
    assert not np.array_equal(streams.public.random(10), streams.eve.random(10))
