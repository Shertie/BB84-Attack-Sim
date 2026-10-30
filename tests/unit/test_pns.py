"""Atak 3, PNS: zachowanie na impulsach, model analityczny i strategia Ewy (F3.11–F3.12)."""

import math
from typing import Any

import numpy as np
import pytest

from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.pns import critical_transmittance, fock_clicks, pns_model, pns_strategy
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.attacks.pns import PhotonNumberSplitting
from qkdsim.detector import dark_count_per_detector
from qkdsim.optics import polarization_angle
from qkdsim.pulses import Pulses

P_D = dark_count_per_detector(GYS.background)


def pulses_with(photons: list[int]) -> Pulses:
    n = len(photons)
    bits = np.arange(n, dtype=np.uint8) % 2
    bases = (np.arange(n, dtype=np.uint8) // 2) % 2
    return Pulses.build(n=photons, bit=bits, basis=bases, theta=polarization_angle(bits, bases))


def test_full_splitting_blocks_singles_and_keeps_one_photon_from_multi() -> None:
    pulses = pulses_with([0, 1, 2, 3, 5, 1])
    out, log = PhotonNumberSplitting(block=1.0, forward=1.0).intercept(
        pulses, np.random.default_rng(0)
    )
    assert out.n.tolist() == [0, 0, 1, 2, 4, 0]
    assert log.measured.tolist() == [False, False, True, True, True, False]
    assert np.array_equal(log.bit[log.measured], pulses.bit[log.measured])
    assert np.array_equal(log.basis[log.measured], pulses.basis[log.measured])
    assert np.array_equal(out.theta, pulses.theta)  # PNS nie zmienia polaryzacji


def test_no_blocking_passes_singles() -> None:
    out, _ = PhotonNumberSplitting(block=0.0).intercept(
        pulses_with([1, 1, 1]), np.random.default_rng(0)
    )
    assert out.n.tolist() == [1, 1, 1]


@pytest.mark.parametrize("kwargs", [{"block": 1.2}, {"block": 0.5, "forward": -0.1}])
def test_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PhotonNumberSplitting(**kwargs)


@pytest.mark.parametrize("k", [0, 1, 2, 7])
def test_fock_clicks_is_a_distribution(k: int) -> None:
    c = fock_clicks(k, 0.3, 0.01, 0.05)
    assert c.right_only + c.wrong_only + c.double + c.none == pytest.approx(1.0)
    assert min(c.right_only, c.wrong_only, c.double, c.none) >= 0.0


def test_critical_distance_for_gys() -> None:
    t = critical_transmittance(0.48, GYS.eta_bob)
    length = -10 * math.log10(t) / GYS.alpha_db_per_km
    assert 30.0 < length < 35.0


@pytest.mark.parametrize("length_km", [10.0, 32.0, 50.0, 100.0])
def test_strategy_reproduces_honest_signal_gain(length_km: float) -> None:
    honest = matched_basis_clicks(0.48, GYS.eta(length_km), P_D, GYS.misalignment)
    block, forward = pns_strategy(honest.detected, 0.48, GYS.eta_bob, P_D)
    attacked = pns_model(0.48, block, forward, GYS.eta_bob, P_D, GYS.misalignment)
    assert attacked.clicks.detected == pytest.approx(honest.detected, rel=1e-9)
    beyond = length_km > 33.0
    assert (block == 1.0 and forward < 1.0) if beyond else (forward == 1.0 and block < 1.0)


def test_eve_knows_almost_everything_beyond_critical_distance() -> None:
    honest = matched_basis_clicks(0.48, GYS.eta(80.0), P_D, GYS.misalignment)
    block, forward = pns_strategy(honest.detected, 0.48, GYS.eta_bob, P_D)
    model = pns_model(0.48, block, forward, GYS.eta_bob, P_D, GYS.misalignment)
    assert model.known_fraction > 0.95
    assert model.clicks.qber == pytest.approx(honest.qber, abs=5e-3)  # QBER prawie bez zmian
