"""Źródło WCP, światłowód i para detektorów (F1.5–F1.7)."""

import math

import numpy as np
import pytest

from conftest import SEED
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.protocol import bob_measure
from qkdsim.pulses import Pulses
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

N = 1_000_000


def _within(observed: float, expected: float, sigma: float, k: float = 4.0) -> bool:
    return abs(observed - expected) < k * sigma


# --- źródło ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("intensities", "probabilities"),
    [((0.5, 0.1), (1.0,)), ((-0.1,), (1.0,)), ((0.5, 0.1), (0.6, 0.6)), ((), ())],
)
def test_source_rejects_invalid_parameters(
    intensities: tuple[float, ...], probabilities: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError):
        WCPSource(intensities, probabilities)


def test_source_photon_numbers_are_poisson_per_intensity() -> None:
    source = WCPSource((0.48, 0.05, 0.0), (0.7, 0.2, 0.1))
    streams = RandomStreams.from_seed(SEED)
    alice, pulses = source.emit(N, streams.alice, streams.nature)
    assert np.array_equal(alice.intensity, pulses.intensity)
    for i, (mu, p) in enumerate(zip(source.intensities, source.probabilities, strict=True)):
        chosen = pulses.intensity == i
        k = int(np.count_nonzero(chosen))
        assert _within(k / N, p, math.sqrt(p * (1 - p) / N))
        photons = pulses.n[chosen]
        if mu == 0.0:
            assert np.all(photons == 0)
        else:
            # Średnia i wariancja rozkładu Poissona są równe μ.
            assert _within(float(photons.mean()), mu, math.sqrt(mu / k))
            assert photons.var() == pytest.approx(mu, rel=0.02)


def test_single_intensity_source_does_not_consume_alice_choice() -> None:
    a = RandomStreams.from_seed(SEED)
    b = RandomStreams.from_seed(SEED)
    WCPSource((0.48,)).emit(1000, a.alice, a.nature)
    WCPSource((0.48,)).emit(1000, b.alice, b.nature)
    assert np.array_equal(a.alice.bits(100), b.alice.bits(100))


# --- kanał ----------------------------------------------------------------------------


def test_channel_transmittance_and_rotation() -> None:
    channel = FiberChannel(100.0, 0.21, 0.033)
    assert channel.transmittance == pytest.approx(10 ** (-2.1))
    assert math.sin(channel.rotation) ** 2 == pytest.approx(0.033)


@pytest.mark.parametrize(
    "kwargs", [{"length_km": -1.0}, {"alpha_db_per_km": -0.1}, {"misalignment": 0.6}]
)
def test_channel_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    params = {"length_km": 10.0, "alpha_db_per_km": 0.21, "misalignment": 0.0} | kwargs
    with pytest.raises(ValueError):
        FiberChannel(**params)


def test_channel_thins_photons_binomially() -> None:
    channel = FiberChannel(50.0, 0.21, 0.0)
    nature = np.random.default_rng(SEED)
    pulses = Pulses.build(n=np.full(N, 3), bit=np.zeros(N), basis=np.zeros(N), theta=np.zeros(N))
    out = channel.transmit(pulses, nature)
    t = channel.transmittance
    assert _within(out.n.mean(), 3 * t, math.sqrt(3 * t * (1 - t) / N))
    assert np.all(out.n <= pulses.n)


def test_channel_rotates_polarization() -> None:
    channel = FiberChannel(0.0, 0.21, 0.033)
    pulses = Pulses.build(n=[1, 1], bit=[0, 1], basis=[0, 1], theta=[0.0, 3 * math.pi / 4])
    out = channel.transmit(pulses, np.random.default_rng(SEED))
    assert np.allclose(out.theta - pulses.theta, channel.rotation)


# --- detektory ------------------------------------------------------------------------


@pytest.mark.parametrize(("efficiency", "dark"), [(0.0, 0.0), (1.1, 0.0), (0.5, -0.1), (0.5, 1.0)])
def test_detector_rejects_invalid_parameters(efficiency: float, dark: float) -> None:
    with pytest.raises(ValueError):
        DetectorPair(efficiency, dark)


def test_dark_count_per_detector_inverts_background() -> None:
    y0 = 1.7e-6
    p_d = dark_count_per_detector(y0)
    assert 1 - (1 - p_d) ** 2 == pytest.approx(y0, rel=1e-12)
    assert p_d == pytest.approx(y0 / 2, rel=1e-6)


def test_ideal_detector_is_deterministic_in_matching_basis() -> None:
    bits = np.array([0, 1, 0, 1] * 1000, dtype=np.uint8)
    bases = np.array([0, 0, 1, 1] * 1000, dtype=np.uint8)
    theta = bases * (math.pi / 4) + bits * (math.pi / 2)
    pulses = Pulses.build(n=np.ones(4000), bit=bits, basis=bases, theta=theta)
    click0, click1 = DetectorPair(1.0, 0.0).clicks(pulses, bases, np.random.default_rng(SEED))
    assert np.array_equal(click1, bits.astype(bool))
    assert np.array_equal(click0, ~bits.astype(bool))


def test_vacuum_pulses_click_only_through_dark_counts() -> None:
    p_d = 0.01
    pulses = Pulses.build(n=np.zeros(N), bit=np.zeros(N), basis=np.zeros(N), theta=np.zeros(N))
    bases = np.zeros(N, dtype=np.uint8)
    click0, click1 = DetectorPair(0.5, p_d).clicks(pulses, bases, np.random.default_rng(SEED))
    sigma = math.sqrt(p_d * (1 - p_d) / N)
    assert _within(click0.mean(), p_d, sigma)
    assert _within(click1.mean(), p_d, sigma)
    # Niezależność: podwójne kliknięcia z p-stwem p_d².
    assert _within((click0 & click1).mean(), p_d**2, math.sqrt(p_d**2 / N))


def test_double_clicks_get_random_bits() -> None:
    """Przy samych ciemnych zliczeniach p_d = 1/2 podwójne kliknięcia to 1/4 impulsów."""
    streams = RandomStreams.from_seed(SEED)
    pulses = Pulses.build(n=np.zeros(N), bit=np.zeros(N), basis=np.zeros(N), theta=np.zeros(N))
    bob = bob_measure(pulses, DetectorPair(0.5, 0.5), streams.bob, streams.nature)
    assert np.all(bob.detected[bob.double])
    bits = bob.bits[bob.double]
    assert _within(bits.mean(), 0.5, 0.5 / math.sqrt(len(bits)))
