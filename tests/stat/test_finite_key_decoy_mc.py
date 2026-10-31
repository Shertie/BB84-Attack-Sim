"""Skończony klucz z decoy: liczby z Lima et al. 2014 i rachunek na danych Monte Carlo."""

import dataclasses

import numpy as np
import pytest

from conftest import SEED
from golden import LIM_MAX_DISTANCE_KM_1E4, LIM_RATIO_100KM, LIM_RATIO_TOLERANCE
from qkdsim.analysis.finite_key_decoy import (
    LimChannel,
    ProtocolChoice,
    SimulatorChannel,
    expected_counts,
    optimal_rate,
)
from qkdsim.analysis.key_rate import GYS
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link, run_counts
from qkdsim.postprocessing.finite_key_decoy import (
    DecoySettings,
    counts_from_simulation,
    key_length_decoy,
)
from qkdsim.rng import BiasedSource, RandomStreams
from qkdsim.source import WCPSource

Z_MAX = 4.0


def test_reproduces_lim_maximal_distance_for_small_blocks() -> None:
    """n_X = 10⁴: klucz jeszcze ~2 km przed 135 km, już nie 5 km za."""
    assert optimal_rate(LimChannel(LIM_MAX_DISTANCE_KM_1E4 - 2.0), 1e4)[0] > 0.0
    assert optimal_rate(LimChannel(LIM_MAX_DISTANCE_KM_1E4 + 5.0), 1e4)[0] <= 0.0


def test_reproduces_lim_block_size_ratio_at_100_km() -> None:
    large, _ = optimal_rate(LimChannel(100.0), 1e9)
    small, _ = optimal_rate(LimChannel(100.0), 1e7)
    assert large / small == pytest.approx(LIM_RATIO_100KM, abs=LIM_RATIO_TOLERANCE)


@pytest.mark.slow
def test_key_length_from_simulated_counts_matches_model() -> None:
    """Asymetryczny BB84 z trzema intensywnościami w torze GYS (25 km), blok ~10⁵ bitów."""
    p_d = dark_count_per_detector(GYS.background)
    choice = ProtocolChoice(0.631, DecoySettings((0.40, 0.13, 2e-4), (0.075, 0.562, 0.363)))
    channel = SimulatorChannel(GYS.eta(25.0), p_d, GYS.misalignment)
    key, test, pulses = expected_counts(choice, channel, 1e5)
    link = Link(
        WCPSource(choice.settings.intensities, choice.settings.probabilities),
        FiberChannel(25.0, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, p_d),
    )
    base = RandomStreams.from_seed(SEED)
    p_x = 1.0 - choice.key_basis  # baza klucza to Z (0)
    streams = dataclasses.replace(
        base,
        alice=BiasedSource(np.random.default_rng([SEED, 1]), p_x=p_x),
        bob=BiasedSource(np.random.default_rng([SEED, 2]), p_x=p_x),
    )
    mc_key, mc_test = counts_from_simulation(run_counts(link, round(pulses), streams))
    for model, mc in ((key, mc_key), (test, mc_test)):
        for expected, observed in zip(
            model.detections + model.errors, mc.detections + mc.errors, strict=True
        ):
            assert abs(observed - expected) < Z_MAX * np.sqrt(expected)
    args = (choice.settings, 1.22, 1e-10, 1e-15)
    model_length = key_length_decoy(key, test, *args).value
    mc_length = key_length_decoy(mc_key, mc_test, *args).value
    assert model_length > 0.0
    assert mc_length == pytest.approx(model_length, rel=0.1)
