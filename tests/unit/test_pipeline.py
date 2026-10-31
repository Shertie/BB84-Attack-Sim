"""Warstwa bitów i warstwa statystyk oraz estymaty z liczników."""

import numpy as np
import pytest

from conftest import SEED
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.metrics import Estimate, gain, matched_qber, qber
from qkdsim.pipeline import Counts, Link, run_bits, run_counts, run_counts_parallel
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

LINK = Link(
    WCPSource((0.48, 0.05, 0.0), (0.8, 0.15, 0.05)),
    FiberChannel(25.0, 0.21, 0.033),
    DetectorPair(0.045, dark_count_per_detector(1.7e-6)),
)


def test_counts_from_bits_layer_match_sifted_key() -> None:
    run = run_bits(LINK, 200_000, RandomStreams.from_seed(SEED))
    counts = Counts.from_records(run.alice, run.bob, 3)
    assert counts.n_pulses == 200_000
    matched_detected = int(np.trace(counts.detected, axis1=1, axis2=2).sum())
    assert matched_detected == len(run.key)
    matched_errors = int(np.trace(counts.errors, axis1=1, axis2=2).sum())
    assert matched_errors == int(np.count_nonzero(run.key.alice != run.key.bob))
    assert matched_errors / matched_detected == pytest.approx(qber(run.key))


def test_statistics_layer_equals_bits_layer_with_one_batch() -> None:
    n = 300_000
    run = run_bits(LINK, n, RandomStreams.from_seed(SEED))
    counts = run_counts(LINK, n, RandomStreams.from_seed(SEED), batch_size=n)
    expected = Counts.from_records(run.alice, run.bob, 3)
    for name in ("sent", "detected", "errors", "double"):
        assert np.array_equal(getattr(counts, name), getattr(expected, name))


def test_batches_sum_to_requested_pulse_count() -> None:
    counts = run_counts(LINK, 1_000_003, RandomStreams.from_seed(SEED), batch_size=2**18)
    assert counts.n_pulses == 1_000_003


def test_parallel_runner_is_independent_of_worker_count() -> None:
    a = run_counts_parallel(LINK, 400_000, SEED, chunk_size=100_000, max_workers=1)
    b = run_counts_parallel(LINK, 400_000, SEED, chunk_size=100_000, max_workers=2)
    assert a.n_pulses == 400_000
    for name in ("sent", "detected", "errors", "double"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_estimates_from_counts() -> None:
    counts = run_counts(LINK, 500_000, RandomStreams.from_seed(SEED))
    g = gain(counts, intensity=0)
    assert g.trials == int(counts.sent[0].sum())
    z_only = matched_qber(counts, intensity=0, basis=0)
    both = matched_qber(counts, intensity=0)
    assert z_only.trials < both.trials


def test_estimate_z_score_and_validation() -> None:
    est = Estimate(successes=520, trials=1000)
    assert est.value == 0.52
    assert est.z_score(0.5) == pytest.approx(20 / np.sqrt(250))
    with pytest.raises(ValueError):
        Estimate(successes=5, trials=0)
    with pytest.raises(ValueError):
        est.z_score(1.0)
