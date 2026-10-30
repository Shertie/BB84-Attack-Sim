"""Własności idealnego BB84 dla dowolnej liczby impulsów i ziarna (hypothesis)."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from qkdsim.metrics import qber
from qkdsim.protocol import alice_prepare_single_photons, run_ideal
from qkdsim.rng import RandomStreams

seeds = st.integers(min_value=0, max_value=2**32 - 1)


@given(n_pulses=st.integers(min_value=1, max_value=5000), seed=seeds)
def test_ideal_key_has_no_errors(n_pulses: int, seed: int) -> None:
    _alice, _bob, key = run_ideal(n_pulses, RandomStreams.from_seed(seed))
    assert len(key) <= n_pulses
    assert np.all(np.diff(key.index) > 0)
    assert np.array_equal(key.alice, key.bob)
    if len(key) > 0:
        assert qber(key) == 0.0


@given(n_pulses=st.integers(min_value=0, max_value=2000), seed=seeds)
def test_alice_sends_only_bb84_states(n_pulses: int, seed: int) -> None:
    _alice, pulses = alice_prepare_single_photons(n_pulses, RandomStreams.from_seed(seed).alice)
    assert len(pulses) == n_pulses
    assert np.all(pulses.n == 1)
    expected = pulses.basis * (np.pi / 4) + pulses.bit * (np.pi / 2)
    assert np.array_equal(pulses.theta, expected)
