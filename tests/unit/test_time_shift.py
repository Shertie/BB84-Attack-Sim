"""Atak 4, time-shift: krzywe sprawności, model, dane Zhao et al. 2008 i obrona."""

import numpy as np
import pytest

from conftest import SEED
from golden import (
    ZHAO_BALANCING_P_A,
    ZHAO_K_LOWER_BITS,
    ZHAO_K_UPPER_BITS,
    ZHAO_QBER,
    ZHAO_ROUNDING,
)
from qkdsim.analysis.time_shift import (
    ZHAO_BACKGROUND,
    ZHAO_DELTA_NS,
    ZHAO_DETECTIONS,
    ZHAO_F_EC,
    ZHAO_MATCHED,
    ZHAO_MU,
    ZHAO_P_A,
    ZHAO_PULSES,
    ZHAO_TABLE,
    balancing_probability,
    bit_probabilities,
    eve_information,
    fit_zhao,
    symmetric_mismatch,
    time_shift_model,
    zhao_key_bounds,
)
from qkdsim.attacks.time_shift import TimeShift
from qkdsim.detector import DetectorPair, EfficiencyMismatch, dark_count_per_detector
from qkdsim.protocol import bob_measure
from qkdsim.pulses import Pulses
from qkdsim.rng import RandomStreams

FIT = fit_zhao()


def test_relative_efficiency_is_one_at_nominal_arrival() -> None:
    curves = EfficiencyMismatch((-0.05, 0.05), (0.15, 0.14))
    t = np.array([0.0, -0.05, 0.3])
    assert curves.relative(t, 0)[0] == curves.relative(t, 1)[0] == 1.0
    assert curves.relative(t, 0)[1] > 1.0  # szczyt detektora 0 leży przed nominalnym przyjściem
    assert curves.relative(t, 1)[2] > curves.relative(t, 0)[2]
    with pytest.raises(ValueError):
        EfficiencyMismatch((0.0, 0.1), (0.1, 0.0))


def test_timing_at_nominal_arrival_changes_nothing() -> None:
    """Bez przesunięcia krzywe dają tę samą sprawność, więc te same ziarna, te same kliknięcia."""
    pulses = Pulses.build(
        n=np.full(1000, 3), bit=np.zeros(1000), basis=np.zeros(1000), theta=np.full(1000, 0.3)
    )
    bases = np.zeros(1000, dtype=np.uint8)
    plain = DetectorPair(0.2, 1e-3)
    timed = DetectorPair(0.2, 1e-3, timing=FIT.mismatch)
    a = plain.clicks(pulses, bases, np.random.default_rng(SEED))
    b = timed.clicks(pulses, bases, np.random.default_rng(SEED))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_random_assignment_keeps_bobs_bits_correct() -> None:
    streams = RandomStreams.from_seed(SEED)
    bits = streams.alice.bits(2000)
    bases = streams.alice.bases(2000)
    pulses = Pulses.build(
        n=np.ones(2000), bit=bits, basis=bases, theta=bases * np.pi / 4 + bits * np.pi / 2
    )
    bob = bob_measure(pulses, DetectorPair(1.0, 0.0), streams.bob, streams.nature, True)
    same = bob.bases == bases
    assert np.array_equal(bob.bits[same], bits[same])


def test_fit_reproduces_zhao_table_ia() -> None:
    dark = dark_count_per_detector(ZHAO_BACKGROUND) * ZHAO_PULSES
    c0, c1 = FIT.mismatch.centers_ns
    assert c1 - c0 == pytest.approx(ZHAO_DELTA_NS)
    for shift, (d0, d1) in zip(FIT.shifts_ns, ZHAO_DETECTIONS, strict=True):
        t = np.array([shift])
        ratio = FIT.mismatch.relative(t, 0)[0] / FIT.mismatch.relative(t, 1)[0]
        assert ratio == pytest.approx((d0 - dark) / (d1 - dark), rel=1e-9)


def test_balancing_reproduces_zhao_p_a() -> None:
    at_a, at_b = (np.array(d, dtype=np.float64) for d in ZHAO_DETECTIONS)
    assert balancing_probability(at_a, at_b) == pytest.approx(ZHAO_BALANCING_P_A, abs=5e-4)
    assert balancing_probability(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    assert balancing_probability(np.array([2.0, 1.0]), np.array([3.0, 2.5])) == 0.0


def test_key_bounds_reproduce_zhao_table_ic() -> None:
    """K_U < K_L z danych artykułu: Ewa wie więcej, niż Alice i Bob usuwają."""
    mixed = np.stack([ZHAO_P_A * ZHAO_TABLE[0], (1.0 - ZHAO_P_A) * ZHAO_TABLE[1]])
    bounds = zhao_key_bounds(mixed, ZHAO_MATCHED, ZHAO_MU, ZHAO_BACKGROUND, ZHAO_F_EC)
    assert bounds.qber == pytest.approx(ZHAO_QBER, abs=5e-5)
    assert bounds.upper == pytest.approx(ZHAO_K_UPPER_BITS, abs=ZHAO_ROUNDING)
    assert bounds.lower == pytest.approx(ZHAO_K_LOWER_BITS, abs=ZHAO_ROUNDING)
    assert bounds.attack_succeeds


def test_eve_information_limits() -> None:
    assert eve_information(np.ones((2, 2, 2))) == pytest.approx(0.0, abs=1e-12)
    perfect = np.zeros((2, 2, 2))
    perfect[0, 0, 0] = perfect[1, 1, 1] = 5.0
    assert eve_information(perfect) == pytest.approx(1.0)


@pytest.mark.parametrize("shift", [-0.3, 0.0, 0.4])
def test_misalignment_qber_does_not_depend_on_shift(shift: float) -> None:
    """Bez ciemnych zliczeń i przy małym ημ QBER = e_d przy każdym niedopasowaniu."""
    t = np.array([shift])
    etas = (0.1 * FIT.mismatch.relative(t, 0)[0], 0.1 * FIT.mismatch.relative(t, 1)[0])
    probs = bit_probabilities(1e-4, 1.0, etas, 0.0, 0.033)
    assert (probs[0, 1] + probs[1, 0]) / probs.sum() == pytest.approx(0.033, rel=1e-3)


def test_random_assignment_removes_eves_information() -> None:
    for swap, positive in ((False, True), (True, False)):
        model = time_shift_model(
            0.48, 0.05, 0.045, 1e-6, 0.033, FIT.mismatch, FIT.shifts_ns, swap=swap
        )
        assert (model.eve_information > 0.2) is positive
    defended = time_shift_model(
        0.48, 0.05, 0.045, 1e-6, 0.033, FIT.mismatch, FIT.shifts_ns, 0.3, True
    )
    assert defended.eve_information == pytest.approx(0.0, abs=1e-12)


def test_information_grows_with_mismatch() -> None:
    width = float(np.mean(FIT.mismatch.widths_ns))
    info = [
        time_shift_model(
            0.48, 0.05, 0.045, 1e-6, 0.033, symmetric_mismatch(d, width), FIT.shifts_ns
        ).eve_information
        for d in (0.0, 0.05, 0.1, 0.2)
    ]
    assert info[0] == pytest.approx(0.0, abs=1e-9)
    assert info == sorted(info)


def test_attack_shifts_and_logs_its_guess() -> None:
    pulses = Pulses.build(
        n=np.full(10_000, 5), bit=np.zeros(10_000), basis=np.zeros(10_000), theta=np.zeros(10_000)
    )
    attack = TimeShift(-0.3, 0.4, p_a=0.25, transmission=0.5)
    shifted, log = attack.intercept(pulses, np.random.default_rng(SEED))
    early = shifted.t < 0
    assert set(np.unique(shifted.t)) == {-0.3, 0.4}
    assert np.mean(early) == pytest.approx(0.25, abs=0.02)
    assert np.array_equal(log.bit, (~early).astype(np.uint8))
    assert log.measured.all()
    assert shifted.n.mean() == pytest.approx(2.5, abs=0.05)
    with pytest.raises(ValueError):
        TimeShift(-0.3, 0.4, p_a=1.5)
