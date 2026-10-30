"""Atak 1, intercept-resend: zachowanie na pojedynczych impulsach i integracja z torem."""

import numpy as np
import numpy.typing as npt
import pytest

from conftest import SEED
from qkdsim.analysis.intercept_resend import intercept_resend_model, qber_line
from qkdsim.analysis.key_rate import GYS
from qkdsim.attacks.intercept_resend import InterceptResend
from qkdsim.channel import FiberChannel
from qkdsim.detector import dark_count_per_detector
from qkdsim.eve import NO_BASIS
from qkdsim.optics import polarization_angle
from qkdsim.pulses import Pulses

N = 100_000


def bb84_pulses(n: int, photons: npt.NDArray[np.int64], seed: int = SEED) -> Pulses:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n, dtype=np.uint8)
    bases = rng.integers(0, 2, n, dtype=np.uint8)
    return Pulses.build(n=photons, bit=bits, basis=bases, theta=polarization_angle(bits, bases))


@pytest.mark.parametrize(("fraction", "position"), [(-0.1, 0.5), (1.1, 0.5), (0.5, 1.5)])
def test_rejects_invalid_parameters(fraction: float, position: float) -> None:
    with pytest.raises(ValueError):
        InterceptResend(fraction, position)


def test_zero_fraction_changes_nothing() -> None:
    pulses = bb84_pulses(N, np.random.default_rng(1).poisson(0.48, N))
    out, log = InterceptResend(0.0).intercept(pulses, np.random.default_rng(2))
    assert np.array_equal(out.n, pulses.n)
    assert np.array_equal(out.theta, pulses.theta)
    assert not log.measured.any()
    assert np.all(log.basis == NO_BASIS)


def test_full_attack_resends_single_photons_and_keeps_vacuum() -> None:
    photons = np.random.default_rng(1).poisson(0.48, N)
    pulses = bb84_pulses(N, photons)
    out, log = InterceptResend(1.0).intercept(pulses, np.random.default_rng(2))
    assert np.array_equal(log.measured, photons > 0)
    assert np.array_equal(out.n, (photons > 0).astype(np.int64))
    resent = log.measured
    assert np.allclose(out.theta[resent], polarization_angle(log.bit, log.basis)[resent])


def test_eve_is_exact_in_alice_basis_and_random_otherwise() -> None:
    pulses = bb84_pulses(N, np.full(N, 3))
    _, log = InterceptResend(1.0).intercept(pulses, np.random.default_rng(2))
    same = log.basis == pulses.basis
    assert np.array_equal(log.bit[same], pulses.bit[same])
    agree = np.mean(log.bit[~same] == pulses.bit[~same])
    assert abs(agree - 0.5) < 4 * 0.5 / np.sqrt(np.count_nonzero(~same))


def test_channel_split_preserves_transmittance_and_puts_misalignment_after_eve() -> None:
    channel = FiberChannel(80.0, 0.21, 0.033)
    before, after = channel.split(0.25)
    assert before.transmittance * after.transmittance == pytest.approx(channel.transmittance)
    assert before.misalignment == 0.0
    assert after.misalignment == 0.033
    _, lossless = channel.split(0.0, lossless_after=True)
    assert lossless.transmittance == 1.0
    with pytest.raises(ValueError):
        channel.split(1.5)


def test_plan_formula_is_exact_at_the_ends_and_good_where_mu_t_is_small() -> None:
    """QBER = e_d + p(¼ − e_d/2): dokładne dla p ∈ {0, 1}; pomiędzy tylko gdy μT₁ ≪ 1."""
    p_d = dark_count_per_detector(GYS.background)
    for fraction in (0.0, 1.0):
        model = intercept_resend_model(GYS, 0.0, 0.48, fraction, 1.0, p_d)
        assert model.clicks.qber == pytest.approx(qber_line(fraction, GYS.misalignment), abs=1e-4)
    for fraction in (0.3, 0.7):
        at_50km = intercept_resend_model(GYS, 50.0, 0.48, fraction, 1.0, p_d).clicks.qber
        assert at_50km == pytest.approx(qber_line(fraction, GYS.misalignment), abs=1e-3)
        # Przy 0 km impuls niesie do Ewy średnio 0,48 fotonu: QBER wyraźnie poniżej prostej.
        at_0km = intercept_resend_model(GYS, 0.0, 0.48, fraction, 1.0, p_d).clicks.qber
        assert at_0km < qber_line(fraction, GYS.misalignment) - 5e-3


def test_model_eve_knowledge_at_full_attack() -> None:
    p_d = dark_count_per_detector(GYS.background)
    model = intercept_resend_model(GYS, 25.0, 0.48, 1.0, 1.0, p_d)
    assert model.measured_detected / model.clicks.detected == pytest.approx(1.0, abs=1e-3)
    assert model.known_fraction == pytest.approx(0.5, abs=1e-3)
    assert model.guess_accuracy == pytest.approx(0.75, abs=1e-3)
