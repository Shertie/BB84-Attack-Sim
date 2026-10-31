"""Atak 8 w torze impulsów: Monte Carlo vs model (F6.8–F6.10)."""

import dataclasses
import math

import numpy as np
import pytest

from conftest import SEED
from qkdsim.analysis.intercept_resend import intercept_resend_model
from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.analysis.weak_rng import monobit_power, runs_power, sifted_stay
from qkdsim.attacks.intercept_resend import InterceptResend
from qkdsim.attacks.weak_rng import (
    BiasedSource,
    MajorityPredictor,
    MarkovSource,
    PassivePrediction,
    ReplayPredictor,
    known_seed_source,
)
from qkdsim.channel import FiberChannel
from qkdsim.defenses.randomness_tests import TESTS
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.eve import ChannelAttack
from qkdsim.metrics import eve_known_fraction, gain, matched_qber
from qkdsim.pipeline import Counts, Link, run_counts, run_sifted
from qkdsim.rng import RandomSource, RandomStreams
from qkdsim.source import WCPSource

Z_MAX = 4.0
MU, LENGTH, PULSES = 0.48, 25.0, 4_000_000
P_D = dark_count_per_detector(GYS.background)


def link(eve: ChannelAttack | None) -> Link:
    return Link(
        WCPSource((MU,)),
        FiberChannel(LENGTH, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
        eve=eve,
    )


def streams(alice: RandomSource) -> RandomStreams:
    return dataclasses.replace(RandomStreams.from_seed(SEED), alice=alice)


@pytest.mark.parametrize("p_x", [0.5, 0.7, 1.0])
def test_biased_bases_intercept_resend(p_x: float) -> None:
    alice = BiasedSource(np.random.default_rng(SEED), p_x=p_x)
    eve = InterceptResend(1.0, predictor=MajorityPredictor(basis=1))
    counts = run_counts(link(eve), PULSES, streams(alice))
    model = intercept_resend_model(GYS, LENGTH, MU, 1.0, 1.0, P_D, right_basis=p_x)
    assert abs(matched_qber(counts).z_score(model.clicks.qber)) < Z_MAX
    known = model.known_detected / model.clicks.detected
    assert abs(eve_known_fraction(counts).z_score(known)) < Z_MAX


def test_known_seed_replay_leaves_no_errors() -> None:
    eve = InterceptResend(1.0, predictor=ReplayPredictor(known_seed_source(7)))
    counts = run_counts(link(eve), PULSES, streams(known_seed_source(7)))
    model = intercept_resend_model(GYS, LENGTH, MU, 1.0, 1.0, P_D, right_basis=1.0)
    honest = matched_basis_clicks(MU, GYS.eta(LENGTH), P_D, GYS.misalignment)
    assert abs(matched_qber(counts).z_score(model.clicks.qber)) < Z_MAX
    assert abs(matched_qber(counts).value - honest.qber) < 0.003


def test_known_seed_passive_reads_the_whole_key() -> None:
    eve = PassivePrediction(ReplayPredictor(known_seed_source(7)))
    counts: Counts = run_counts(link(eve), PULSES, streams(known_seed_source(7)))
    honest = matched_basis_clicks(MU, GYS.eta(LENGTH), P_D, GYS.misalignment)
    assert eve_known_fraction(counts).value == 1.0
    assert abs(matched_qber(counts).z_score(honest.qber)) < Z_MAX
    assert abs(gain(counts).z_score(honest.detected)) < Z_MAX


@pytest.mark.parametrize("stay", [0.99, 0.999])
def test_sifting_thins_markov_correlations(stay: float) -> None:
    pulses = 20_000_000
    alice = MarkovSource(np.random.default_rng(SEED), stay_bits=stay)
    key = run_sifted(link(None), pulses, streams(alice))
    equal = int(np.count_nonzero(key.alice[1:] == key.alice[:-1]))
    expected = sifted_stay(stay, len(key) / pulses)
    sigma = math.sqrt(expected * (1 - expected) / (len(key) - 1))
    assert abs(equal / (len(key) - 1) - expected) < Z_MAX * sigma


def test_known_seed_passes_tests_and_power_matches_model() -> None:
    """Znane ziarno odrzucane z p-stwem α jak dobry generator; obciążenie i Markow jak w modelu."""
    n, runs = 10_000, 300
    sources: dict[str, tuple[RandomSource, tuple[float, float]]] = {
        "znane ziarno": (known_seed_source(SEED), (0.01, 0.01)),
        "obciążenie": (
            BiasedSource(np.random.default_rng(1), p_one=0.53),
            (monobit_power(n, p_one=0.53), runs_power(n, p_one=0.53)),
        ),
        "Markow": (
            MarkovSource(np.random.default_rng(2), stay_bits=0.52),
            (monobit_power(n, stay=0.52), runs_power(n, stay=0.52)),
        ),
    }
    for name, (source, expected) in sources.items():
        rejected = np.zeros(2)
        for _ in range(runs):
            bits = source.bits(n)
            rejected += [TESTS["monobit"](bits) < 0.01, TESTS["runs"](bits) < 0.01]
        for observed, model in zip(rejected / runs, expected, strict=True):
            sigma = math.sqrt(max(model * (1 - model), 0.01 * 0.99) / runs)
            assert abs(observed - model) < Z_MAX * sigma, (name, observed, model)
