"""Obrony z progiem dla ataków 1, 4, 5 i 6: test QBER, monitor czasu, monitor wejścia Alice."""

import math

import numpy as np
import pytest
from scipy.stats import binom, norm

from conftest import SEED
from qkdsim.defenses.input_monitor import InputMonitor
from qkdsim.defenses.power_monitor import PowerMonitor
from qkdsim.defenses.qber_test import (
    binomial_auc,
    binomial_roc,
    detection_probability,
    error_limit,
    tolerance,
)
from qkdsim.defenses.roc import auc
from qkdsim.defenses.timing_monitor import TimingMonitor


def test_qber_threshold_keeps_false_alarms_below_alpha() -> None:
    limit = error_limit(1000, 0.02, 0.01)
    assert binom.sf(limit, 1000, 0.02) <= 0.01 < binom.sf(limit - 1, 1000, 0.02)
    assert tolerance(1000, 0.02) == limit / 1000
    assert detection_probability(1000, 0.02, 0.02) <= 0.01
    assert detection_probability(1000, 0.02, 0.05) > 0.99


def test_binomial_roc_and_auc_agree_with_sampling() -> None:
    fpr, tpr = binomial_roc(200, 0.02, 0.04)
    assert (fpr[0], tpr[0]) == pytest.approx((1.0, 1.0))
    assert (fpr[-1], tpr[-1]) == pytest.approx((0.0, 0.0))
    rng = np.random.default_rng(SEED)
    honest = rng.binomial(200, 0.02, 20_000).astype(float)
    attacked = rng.binomial(200, 0.04, 20_000).astype(float)
    assert binomial_auc(200, 0.02, 0.04) == pytest.approx(
        auc(honest[:4000], attacked[:4000]), abs=0.015
    )
    assert binomial_auc(50, 0.1, 0.1) == pytest.approx(0.5)


def test_timing_monitor_window_statistics() -> None:
    monitor = TimingMonitor(jitter_ns=0.1, window_ns=0.2)
    assert monitor.outside_probability(0.0) == pytest.approx(2 * norm.sf(2.0))
    assert monitor.outside_probability(0.4) > 0.97
    rng = np.random.default_rng(SEED)
    counts = [monitor.outside(np.zeros(1000), rng) for _ in range(200)]
    assert np.mean(counts) == pytest.approx(1000 * monitor.outside_probability(0.0), rel=0.05)
    limit = monitor.alarm_limit(30)
    assert binom.sf(limit, 30, monitor.outside_probability(0.0)) <= 0.01


def test_input_monitor_bound_matches_readings() -> None:
    monitor = InputMonitor(PowerMonitor(0.01, 1e-8), repetition_hz=2e6, windows=100)
    assert monitor.detection_probability(0.0) == pytest.approx(0.01)
    hidden = monitor.largest_hidden_probe(missed=0.05)
    assert monitor.detection_probability(hidden) == pytest.approx(0.95)
    rng = np.random.default_rng(SEED)
    power = monitor.probe_power_w(hidden)
    alarms = [
        monitor.monitor.readings(power, monitor.windows, rng).mean() > monitor.alarm_level_w()
        for _ in range(4000)
    ]
    assert np.mean(alarms) == pytest.approx(0.95, abs=4 * math.sqrt(0.05 * 0.95 / 4000))
    assert monitor.auc(0.0) == pytest.approx(0.5)
