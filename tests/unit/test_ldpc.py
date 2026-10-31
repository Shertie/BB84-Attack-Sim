"""Uzgadnianie LDPC z syndromem i ewolucja gęstości."""

import math

import numpy as np
import pytest

from conftest import SEED
from qkdsim.analysis.entropy import h2
from qkdsim.classical import ClassicalChannel
from qkdsim.postprocessing.ldpc import (
    bp_threshold,
    channel_density,
    check_degrees,
    decode,
    density_evolution,
    error_probability,
    random_code,
    reconcile,
    syndrome_length,
)
from qkdsim.postprocessing.leakage import linear_eve_knowledge
from qkdsim.postprocessing.run import PostprocessingConfig, postprocess
from qkdsim.protocol import SiftedKey


def noisy_pair(n: int, qber: float, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    alice = rng.integers(0, 2, n, dtype=np.uint8)
    return alice, alice ^ (rng.random(n) < qber).astype(np.uint8)


def test_code_has_fixed_variable_degree_and_no_double_edges() -> None:
    code = random_code(1200, 500, np.random.default_rng(SEED))
    assert code.m == 500 and code.rate == pytest.approx(1 - 500 / 1200)
    assert np.all(np.bincount(code.edge_var, minlength=1200) == 3)
    degrees = np.diff(np.r_[code.check_start, len(code.edge_var)])
    assert set(degrees.tolist()) <= {7, 8}  # 3600/500 = 7,2
    assert all(len(np.unique(row)) == len(row) for row in code.rows())


def test_syndrome_equals_dense_product() -> None:
    code = random_code(300, 120, np.random.default_rng(SEED))
    dense = np.zeros((code.m, code.n), dtype=np.uint8)
    dense[code.edge_check, code.edge_var] = 1
    bits = np.random.default_rng(SEED + 1).integers(0, 2, 300, dtype=np.uint8)
    assert np.array_equal(code.syndrome(bits), (dense.astype(int) @ bits) % 2)


def test_check_degrees_keep_mean_degree() -> None:
    for rate in (0.5, 0.7, 0.83):
        dist = check_degrees(rate)
        assert sum(dist.values()) == pytest.approx(1.0)
        mean = 1.0 / sum(w / d for d, w in dist.items())  # średnia po kontrolach
        assert mean == pytest.approx(3 / (1 - rate))
    assert check_degrees(0.7) == {10: 1.0}


def test_decoder_corrects_errors_below_threshold_and_fails_above() -> None:
    code = random_code(4096, 2048, np.random.default_rng(SEED))  # (3,6), próg BP 0,084
    alice, bob = noisy_pair(4096, 0.04)
    result = decode(code, bob, code.syndrome(alice), 0.04)
    assert result.success and np.array_equal(result.bits, alice)
    alice, bob = noisy_pair(4096, 0.15)
    assert not decode(code, bob, code.syndrome(alice), 0.15, max_iterations=30).success


def test_decoder_without_errors_stops_at_once() -> None:
    code = random_code(1000, 400, np.random.default_rng(SEED))
    alice, _ = noisy_pair(1000, 0.0)
    assert decode(code, alice, code.syndrome(alice), 0.02).iterations == 0


def test_reconcile_splits_key_into_equal_frames_and_counts_leak() -> None:
    alice, bob = noisy_pair(5000, 0.03)
    channel = ClassicalChannel()
    result = reconcile(
        alice, bob, 0.03, np.random.default_rng(SEED), channel, frame=2048, efficiency=1.8
    )
    assert result.frames == 3 and result.failed_frames == 0  # 3 ramki po 1667 bitów
    assert result.syndrome_bits == syndrome_length(1667, 0.03, 1.8)
    assert result.leaked_bits == 3 * result.syndrome_bits == channel.leaked_bits()
    assert result.kept.all() and np.array_equal(result.corrected, alice)


def test_reconcile_discards_frames_it_cannot_decode() -> None:
    alice, bob = noisy_pair(4096, 0.05)
    result = reconcile(
        alice,
        bob,
        0.05,
        np.random.default_rng(SEED),
        ClassicalChannel(),
        frame=1024,
        efficiency=0.3,
    )
    assert result.failed_frames == result.frames == 4
    assert not result.kept.any()


def test_syndrome_length_follows_target_efficiency() -> None:
    assert syndrome_length(10_000, 0.02, 1.5) == math.ceil(1.5 * 10_000 * h2(0.02))
    assert syndrome_length(100, 0.5, 3.0) == 99


def test_postprocess_with_ldpc_balances_bits() -> None:
    rng = np.random.default_rng(SEED)
    alice = rng.integers(0, 2, 40_000, dtype=np.uint8)
    bob = alice ^ (rng.random(40_000) < 0.03).astype(np.uint8)
    key = SiftedKey(index=np.arange(40_000), alice=alice, bob=bob)
    config = PostprocessingConfig(reconciliation="ldpc", ldpc_frame=8192)
    result = postprocess(key, np.random.default_rng(SEED), config)
    b = result.balance
    assert b.verified and b.residual_errors == 0 and b.discarded == 0
    assert b.corrections == b.errors_before_ec
    assert b.leak_ec + b.leak_verification == result.channel.leaked_bits()
    assert 0 < b.final_length == len(result.alice_key)
    assert np.array_equal(result.alice_key, result.bob_key)
    assert len(result.channel.messages("ldpc_syndrome")) == 5  # jedna wiadomość na ramkę


@pytest.mark.parametrize("subtract", [True, False])
def test_linear_eve_and_ldpc_syndromes(subtract: bool) -> None:
    rng = np.random.default_rng(SEED)
    alice = rng.integers(0, 2, 2400, dtype=np.uint8)
    bob = alice ^ (rng.random(2400) < 0.03).astype(np.uint8)
    key = SiftedKey(index=np.arange(2400), alice=alice, bob=bob)
    config = PostprocessingConfig(
        reconciliation="ldpc", ldpc_frame=1024, record_rows=True, subtract_leakage=subtract
    )
    result = postprocess(key, np.random.default_rng(SEED), config)
    n = result.balance.reconciled
    eve = linear_eve_knowledge(result.channel, n, result.pa_seed)
    if subtract:
        assert eve.known_final_bits == 0
    else:
        assert eve.known_final_bits >= result.balance.final_length - (n - eve.disclosed_rank) > 0


def test_channel_density_is_normalized_with_exact_mean() -> None:
    grid = (np.arange(-400, 401) * 0.05).astype(np.float64)
    density = channel_density(0.07, grid)
    llr = math.log(0.93 / 0.07)
    assert density.sum() == pytest.approx(1.0)
    assert float(density @ grid) == pytest.approx((1 - 2 * 0.07) * llr)
    assert error_probability(density) == pytest.approx(0.07)


def test_density_evolution_brackets_known_36_threshold() -> None:
    """Próg BP kodu (3,6) na BSC to 0,084 (Richardson, Urbanke 2001)."""
    assert density_evolution(0.080, 3, {6: 1.0})[0]
    assert not density_evolution(0.088, 3, {6: 1.0})[0]


@pytest.mark.slow
def test_bp_threshold_of_36_code() -> None:
    assert bp_threshold(3, {6: 1.0}) == pytest.approx(0.084, abs=5e-4)
