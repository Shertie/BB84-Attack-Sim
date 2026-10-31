"""Atak 2: tryb liniowy detektora, atak oślepiania i obrony."""

import math
from typing import Any

import numpy as np
import pytest

from conftest import SEED
from qkdsim.attacks.blinding import DetectorBlinding, matching_send_fraction
from qkdsim.defenses.power_monitor import PowerMonitor
from qkdsim.defenses.random_attenuation import AttenuationTally, RandomAttenuator, attenuation_z
from qkdsim.defenses.roc import auc, roc_curve
from qkdsim.detector import CLAVIS2, DetectorPair, LinearMode
from qkdsim.optics import polarization_angle
from qkdsim.pulses import Pulses
from qkdsim.rng import RandomStreams

N = 200_000


def classical_pulses(n: int, power: float, cw: float, bit: int = 0, basis: int = 0) -> Pulses:
    bits, bases = np.full(n, bit, np.uint8), np.full(n, basis, np.uint8)
    return Pulses.build(
        n=np.zeros(n), bit=bits, basis=bases, theta=polarization_angle(bits, bases),
        power=np.full(n, power), cw_power=np.full(n, cw),
    )  # fmt: skip


# --- tryb liniowy ---------------------------------------------------------------------


def test_linear_response_ramp_and_step() -> None:
    mode = LinearMode((1.0, 1.0), never_w=(1.0, 2.0), always_w=(3.0, 2.0))
    power = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    assert np.allclose(mode.response(power, 0), [0.0, 0.0, 0.5, 1.0, 1.0])
    assert np.array_equal(mode.response(power, 1), [0.0, 0.0, 1.0, 1.0, 1.0])  # próg ostry


def test_linear_mode_validation_and_clavis2_window() -> None:
    with pytest.raises(ValueError):
        LinearMode((1.0, 1.0), never_w=(2.0, 2.0), always_w=(1.0, 3.0))
    low, high = CLAVIS2.perfect_window_w
    assert low == pytest.approx(932e-6)
    assert high == pytest.approx(2 * 647e-6)
    assert low < high  # warunek Lydersena: max P_always < 2 min P_never


def test_bright_pulse_saturates_unblinded_detectors() -> None:
    """Bez oślepienia jasny impuls klika w obu detektorach, nawet od małej części mocy."""
    detectors = DetectorPair(0.045, 1e-6, CLAVIS2)
    theta_bits = classical_pulses(1000, 1e-3, cw=0.0).replace(theta=np.full(1000, 0.2))
    c0, c1 = detectors.clicks(theta_bits, np.zeros(1000, np.uint8), np.random.default_rng(0))
    assert c0.all() and c1.all()


def test_blinded_detectors_have_no_dark_counts_and_ignore_photons() -> None:
    detectors = DetectorPair(1.0, 0.5, CLAVIS2)
    pulses = classical_pulses(N, power=0.0, cw=1.08e-3).replace(n=np.full(N, 5, dtype=np.int64))
    c0, c1 = detectors.clicks(pulses, np.zeros(N, np.uint8), np.random.default_rng(0))
    assert not c0.any() and not c1.any()


def test_trigger_pulse_in_window_clicks_only_in_matching_basis() -> None:
    detectors = DetectorPair(0.045, 1e-6, CLAVIS2)
    pulses = classical_pulses(1000, 1.1e-3, cw=1.08e-3, bit=1, basis=0)
    rng = np.random.default_rng(0)
    c0, c1 = detectors.clicks(pulses, np.zeros(1000, np.uint8), rng)
    assert c1.all() and not c0.any()  # zgodna baza: klika detektor bitu Ewy
    c0, c1 = detectors.clicks(pulses, np.ones(1000, np.uint8), rng)
    assert not c0.any() and not c1.any()  # niezgodna baza: po 550 µW, poniżej P_never


def test_partial_blinding_leaves_one_detector_in_geiger_mode() -> None:
    detectors = DetectorPair(0.045, 0.0, CLAVIS2)
    cw = np.full(3, 0.5e-3)  # między 397 µW a 765 µW
    assert detectors.blinded(cw, 0).all()
    assert not detectors.blinded(cw, 1).any()


# --- atak -----------------------------------------------------------------------------


def test_blinding_attack_replaces_pulses_with_classical_light() -> None:
    streams = RandomStreams.from_seed(SEED)
    bits, bases = streams.alice.bits(N), streams.alice.bases(N)
    pulses = Pulses.build(
        n=streams.nature.poisson(0.48, N), bit=bits, basis=bases,
        theta=polarization_angle(bits, bases),
    )  # fmt: skip
    attack = DetectorBlinding(1.1e-3, send_fraction=0.3)
    out, log = attack.intercept(pulses, streams.eve)
    assert np.all(out.n == 0)
    assert np.all(out.cw_power == 1.08e-3)
    assert np.array_equal(log.measured, pulses.n > 0)
    sent = out.power > 0
    n_measured = int(log.measured.sum())
    assert np.all(log.measured[sent])
    assert abs(int(sent.sum()) / n_measured - 0.3) < 4 * math.sqrt(0.21 / n_measured)
    same = log.basis == pulses.basis
    assert np.array_equal(log.bit[same], pulses.bit[same])


def test_matching_send_fraction() -> None:
    target = 6.4e-3
    s = matching_send_fraction(target, 0.48)
    assert s * 0.5 * (1 - math.exp(-0.48)) == pytest.approx(target)
    assert matching_send_fraction(1.0, 0.48) == 1.0


@pytest.mark.parametrize("kwargs", [{"send_fraction": 1.5}, {"mimic_qber": -0.1}])
def test_blinding_attack_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DetectorBlinding(1e-3, **kwargs)


# --- obrony ---------------------------------------------------------------------------


def test_attenuator_masks_and_scales() -> None:
    streams = RandomStreams.from_seed(SEED)
    pulses = classical_pulses(N, 1e-3, 0.0).replace(n=np.full(N, 4, dtype=np.int64))
    out, mask = RandomAttenuator(0.1, 0.5).apply(pulses, streams.bob, streams.nature)
    assert abs(mask.mean() - 0.1) < 4 * math.sqrt(0.09 / N)
    assert np.allclose(out.power[mask], 0.5e-3)
    assert np.allclose(out.power[~mask], 1e-3)
    assert abs(out.n[mask].mean() - 2.0) < 0.05
    assert np.array_equal(out.n[~mask], pulses.n[~mask])


def test_attenuation_z_on_tallies() -> None:
    # Reszta: 9500 kliknięć na 900 000 bramek; w tłumionych oczekiwane 0,5 · 9500/9 ≈ 528.
    honest = AttenuationTally(
        sent=10**6, detected=9500 + 528, sent_attenuated=10**5, detected_attenuated=528
    )
    assert abs(attenuation_z(honest, 0.5)) < 0.1
    attacked = AttenuationTally(10**6, 10_000, 10**5, 0)
    assert attenuation_z(attacked, 0.5) < -10.0


def test_power_monitor_readings() -> None:
    monitor = PowerMonitor(tap_fraction=0.01, noise_w=1e-8)
    rng = np.random.default_rng(0)
    honest = monitor.readings(0.0, 10_000, rng)
    attacked = monitor.readings(1.08e-3, 10_000, rng)
    assert abs(honest.mean()) < 4e-10
    assert attacked.mean() == pytest.approx(1.08e-5, rel=1e-3)


def test_roc_and_auc() -> None:
    honest, attacked = np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0])
    assert auc(honest, attacked) == 1.0
    fpr, tpr = roc_curve(honest, attacked)
    assert fpr[0] == 0.0 and tpr[-1] == 1.0
    same = np.arange(100, dtype=np.float64)
    assert auc(same, same) == pytest.approx(0.5)
