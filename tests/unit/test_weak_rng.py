"""Atak 8, słaby generator: źródła, przewidywanie Ewy, deficyt entropii, testy losowości."""

import numpy as np
import pytest

from conftest import SEED
from golden import NIST_MONOBIT_EXAMPLE, NIST_PVALUE_DIGITS, NIST_RUNS_EXAMPLE
from qkdsim.analysis.intercept_resend import intercept_resend_model
from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.weak_rng import (
    autocorrelation_power,
    min_entropy_deficit,
    monobit_power,
    runs_power,
    shannon_deficit,
    sifted_stay,
)
from qkdsim.attacks.intercept_resend import InterceptResend
from qkdsim.attacks.weak_rng import (
    BiasedSource,
    MajorityPredictor,
    MarkovSource,
    ReplayPredictor,
    known_seed_source,
)
from qkdsim.defenses.randomness_tests import autocorrelation_pvalue, monobit_pvalue, runs_pvalue
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource


def as_bits(text: str) -> np.ndarray:
    return np.array([int(c) for c in text], dtype=np.uint8)


def test_nist_examples() -> None:
    bits, expected = NIST_MONOBIT_EXAMPLE
    assert monobit_pvalue(as_bits(bits)) == pytest.approx(expected, abs=NIST_PVALUE_DIGITS)
    bits, expected = NIST_RUNS_EXAMPLE
    assert runs_pvalue(as_bits(bits)) == pytest.approx(expected, abs=NIST_PVALUE_DIGITS)


def test_runs_prerequisite_and_autocorrelation() -> None:
    assert runs_pvalue(np.ones(100, dtype=np.uint8)) == 0.0
    alternating = np.tile(np.array([0, 1], dtype=np.uint8), 500)
    assert autocorrelation_pvalue(alternating) < 1e-10
    assert monobit_pvalue(alternating) == 1.0


def test_biased_and_markov_sources() -> None:
    rng = np.random.default_rng(SEED)
    biased = BiasedSource(rng, p_one=0.55, p_x=0.7)
    assert biased.bits(200_000).mean() == pytest.approx(0.55, abs=0.005)
    assert biased.bases(200_000).mean() == pytest.approx(0.7, abs=0.005)
    markov = MarkovSource(rng, stay_bits=0.9)
    chain = np.concatenate([markov.bits(50_000), markov.bits(50_000)])
    assert np.mean(chain[1:] == chain[:-1]) == pytest.approx(0.9, abs=0.005)
    with pytest.raises(ValueError):
        BiasedSource(rng, p_one=1.5)


def test_replay_follows_alice_with_decoy_intensities() -> None:
    source = WCPSource((0.48, 0.05), (0.8, 0.2))
    alice = known_seed_source(SEED)
    eve = ReplayPredictor(known_seed_source(SEED), source.probabilities)
    nature = np.random.default_rng(1)
    for size in (1000, 37, 500):
        record, _ = source.emit(size, alice, nature)
        bits, bases = eve.predict(size)
        assert np.array_equal(bits, record.bits) and np.array_equal(bases, record.bases)


def test_intercept_resend_uses_predicted_bases() -> None:
    streams = RandomStreams.from_seed(SEED)
    record, pulses = WCPSource((0.48,)).emit(5000, known_seed_source(7), streams.nature)
    attack = InterceptResend(1.0, predictor=ReplayPredictor(known_seed_source(7)))
    _, log = attack.intercept(pulses, streams.eve)
    measured = log.measured
    assert np.array_equal(log.basis[measured], record.bases[measured])
    assert np.array_equal(log.bit[measured], record.bits[measured])  # bez błędu optyki przed Ewą
    majority = InterceptResend(1.0, predictor=MajorityPredictor(basis=1))
    _, log = majority.intercept(pulses, streams.eve)
    assert (log.basis[log.measured] == 1).all()


def test_right_basis_controls_model() -> None:
    def model(right: float) -> tuple[float, float]:
        m = intercept_resend_model(GYS, 25.0, 0.48, 1.0, 1.0, 1e-6, right)
        return m.clicks.qber, m.known_detected / m.clicks.detected

    default = intercept_resend_model(GYS, 25.0, 0.48, 1.0, 1.0, 1e-6)
    assert default.clicks.qber == pytest.approx(model(0.5)[0])
    random_qber, random_known = model(0.5)
    known_qber, known_known = model(1.0)
    assert known_qber == pytest.approx(GYS.misalignment, abs=2e-3)  # bez błędów od Ewy
    assert random_qber > 0.25
    assert known_known == pytest.approx(2 * random_known)


def test_entropy_deficits() -> None:
    assert shannon_deficit(0.5) == pytest.approx(0.0)
    assert shannon_deficit(0.55) == pytest.approx(0.00722, abs=1e-5)
    assert min_entropy_deficit(0.55) == pytest.approx(0.13750, abs=1e-5)
    assert sifted_stay(0.8, 1.0) == pytest.approx(0.8)
    assert sifted_stay(0.8, 1e-9) == pytest.approx(0.5, abs=1e-8)


def test_power_limits() -> None:
    for power in (monobit_power, runs_power, autocorrelation_power):
        assert power(10_000) == pytest.approx(0.01, abs=1e-3)  # bez odchylenia: p-stwo α
    assert monobit_power(10_000, p_one=0.55) > 0.99
    assert runs_power(10_000, stay=0.55) > 0.99
    with pytest.raises(ValueError):
        monobit_power(100, p_one=0.6, stay=0.6)
