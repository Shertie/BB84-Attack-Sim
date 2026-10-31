"""Etap 1: idealny BB84, jeden foton na impuls, kanał bez strat."""

import numpy as np
import pytest

from conftest import SEED
from qkdsim.metrics import qber, sifted_fraction
from qkdsim.optics import analyzer_angle, polarization_angle, prob_detector0
from qkdsim.protocol import (
    BASIS_X,
    BASIS_Z,
    AliceRecord,
    BobRecord,
    SiftedKey,
    alice_prepare_single_photons,
    bob_measure_ideal,
    run_ideal,
    sift,
)
from qkdsim.pulses import Pulses
from qkdsim.rng import RandomStreams

N_PULSES = 1_000_000

IdealRun = tuple[AliceRecord, BobRecord, SiftedKey]


@pytest.fixture(scope="module")
def ideal_run() -> IdealRun:
    return run_ideal(N_PULSES, RandomStreams.from_seed(SEED))


def test_polarization_angles_follow_convention() -> None:
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)
    bases = np.array([BASIS_Z, BASIS_Z, BASIS_X, BASIS_X], dtype=np.uint8)
    expected = np.array([0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4])  # H, V, D, A
    assert np.allclose(polarization_angle(bits, bases), expected)


def test_born_rule_is_exact_in_matching_basis() -> None:
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)
    bases = np.array([BASIS_Z, BASIS_Z, BASIS_X, BASIS_X], dtype=np.uint8)
    p0 = prob_detector0(polarization_angle(bits, bases), analyzer_angle(bases))
    assert np.array_equal(p0, np.array([1.0, 0.0, 1.0, 0.0]))


def test_born_rule_is_half_in_conjugate_basis() -> None:
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)
    bases = np.array([BASIS_Z, BASIS_Z, BASIS_X, BASIS_X], dtype=np.uint8)
    p0 = prob_detector0(polarization_angle(bits, bases), analyzer_angle(1 - bases))
    assert np.allclose(p0, 0.5, rtol=0.0, atol=1e-15)


def test_qber_is_exactly_zero(ideal_run: IdealRun) -> None:
    _alice, _bob, key = ideal_run
    assert qber(key) == 0.0


def test_sifted_fraction_is_half_within_3_sigma(ideal_run: IdealRun) -> None:
    _alice, _bob, key = ideal_run
    sigma = 0.5 / np.sqrt(N_PULSES)
    assert abs(sifted_fraction(key, N_PULSES) - 0.5) < 3 * sigma


def test_conjugate_basis_outcomes_are_coin_flips() -> None:
    """Przy niezgodnych bazach bit Boba zgadza się z bitem Alice w połowie przypadków."""
    streams = RandomStreams.from_seed(SEED)
    alice, pulses = alice_prepare_single_photons(N_PULSES, streams.alice)
    bob = bob_measure_ideal(pulses, streams.bob, streams.nature)
    mismatch = alice.bases != bob.bases
    agree = np.mean(alice.bits[mismatch] == bob.bits[mismatch])
    sigma = 0.5 / np.sqrt(np.count_nonzero(mismatch))
    assert abs(agree - 0.5) < 3 * sigma


def test_same_seed_gives_identical_run() -> None:
    _, bob_a, key_a = run_ideal(10_000, RandomStreams.from_seed(SEED))
    _, bob_b, key_b = run_ideal(10_000, RandomStreams.from_seed(SEED))
    assert np.array_equal(bob_a.bits, bob_b.bits)
    assert np.array_equal(key_a.index, key_b.index)


def test_different_seed_gives_different_run() -> None:
    _, _, key_a = run_ideal(10_000, RandomStreams.from_seed(SEED))
    _, _, key_b = run_ideal(10_000, RandomStreams.from_seed(SEED + 1))
    assert not np.array_equal(key_a.index, key_b.index)


def test_ideal_measurement_rejects_multiphoton_pulses() -> None:
    streams = RandomStreams.from_seed(SEED)
    pulses = Pulses.build(n=[1, 2], bit=[0, 0], basis=[0, 0], theta=[0.0, 0.0])
    with pytest.raises(ValueError, match="jednofotonowe"):
        bob_measure_ideal(pulses, streams.bob, streams.nature)


def test_sift_keeps_only_matching_bases() -> None:
    streams = RandomStreams.from_seed(SEED)
    alice, pulses = alice_prepare_single_photons(1000, streams.alice)
    bob = bob_measure_ideal(pulses, streams.bob, streams.nature)
    key = sift(alice, bob)
    assert np.all(alice.bases[key.index] == bob.bases[key.index])
    assert len(key) == np.count_nonzero(alice.bases == bob.bases)


def test_qber_of_empty_key_is_an_error() -> None:
    streams = RandomStreams.from_seed(SEED)
    alice, pulses = alice_prepare_single_photons(0, streams.alice)
    key = sift(alice, bob_measure_ideal(pulses, streams.bob, streams.nature))
    with pytest.raises(ValueError, match="pustego"):
        qber(key)
