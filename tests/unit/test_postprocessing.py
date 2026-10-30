"""Cały postprocessing, bilans klucza i test liniowej Ewy (F2.8–F2.9)."""

import numpy as np
import pytest

from conftest import SEED
from qkdsim.analysis.key_rate import GYS
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link, run_bits, run_sifted
from qkdsim.postprocessing.leakage import linear_eve_knowledge
from qkdsim.postprocessing.run import PostprocessingConfig, postprocess
from qkdsim.protocol import SiftedKey
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource


def synthetic_key(n: int, qber: float, seed: int = SEED) -> SiftedKey:
    rng = np.random.default_rng(seed)
    alice = rng.integers(0, 2, n, dtype=np.uint8)
    bob = alice ^ (rng.random(n) < qber).astype(np.uint8)
    return SiftedKey(index=np.arange(n), alice=alice, bob=bob)


def gys_link(length_km: float) -> Link:
    return Link(
        WCPSource((0.48,)),
        FiberChannel(length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, dark_count_per_detector(GYS.background)),
    )


def test_balance_is_consistent() -> None:
    key = synthetic_key(20_000, 0.03)
    result = postprocess(key, np.random.default_rng(SEED))
    b = result.balance
    assert b.sample + b.reconciled == b.sifted
    assert b.verified and b.residual_errors == 0
    assert b.corrections == b.errors_before_ec
    assert b.leak_ec + b.leak_verification == result.channel.leaked_bits()
    assert 0 < b.final_length == len(result.alice_key)
    assert np.array_equal(result.alice_key, result.bob_key)


def test_failed_verification_discards_key() -> None:
    key = synthetic_key(5000, 0.03)
    # Jeden przebieg Cascade zostawia błędy resztkowe, które weryfikacja musi złapać.
    result = postprocess(key, np.random.default_rng(SEED), PostprocessingConfig(cascade_passes=1))
    assert result.balance.residual_errors > 0
    assert not result.balance.verified
    assert result.balance.final_length == 0


@pytest.mark.parametrize("subtract", [True, False])
def test_linear_eve_knows_nothing_only_if_leak_is_subtracted(subtract: bool) -> None:
    """F2.8: dokładna wiedza Ewy, która zna parzystości Cascade i hasz weryfikacyjny."""
    key = synthetic_key(2200, 0.03)
    config = PostprocessingConfig(record_rows=True, subtract_leakage=subtract)
    result = postprocess(key, np.random.default_rng(SEED), config)
    n = result.balance.reconciled
    eve = linear_eve_knowledge(result.channel, n, result.pa_seed)
    if subtract:
        assert eve.known_final_bits == 0
    else:
        lower_bound = result.balance.final_length - (n - eve.disclosed_rank)
        assert lower_bound > 0
        assert eve.known_final_bits >= lower_bound


def test_sifted_collection_matches_bits_layer() -> None:
    n = 400_000
    whole = run_bits(gys_link(25.0), n, RandomStreams.from_seed(SEED)).key
    batched = run_sifted(gys_link(25.0), n, RandomStreams.from_seed(SEED), batch_size=n)
    assert np.array_equal(whole.index, batched.index)
    assert np.array_equal(whole.alice, batched.alice)


def test_end_to_end_on_gys_link() -> None:
    key = run_sifted(gys_link(25.0), 3_000_000, RandomStreams.from_seed(SEED))
    streams = RandomStreams.from_seed(SEED)
    result = postprocess(key, streams.public)
    b = result.balance
    assert b.verified
    assert 0.02 < b.qber_sample < 0.05
    assert np.array_equal(result.alice_key, result.bob_key)
