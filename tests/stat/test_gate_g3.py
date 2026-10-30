"""Bramka G3 (ROADMAP, rozdz. 6) przy 25 km, parametry GYS.

Wolne testy (`make test-slow`). Atak 1: siatka p × położenie Ewy; atak 2: siatka mocy impulsu
wyzwalającego dla Ewy skalibrowanej i naiwnej. Część dla ataku 3 dojdzie razem z nim.
"""

import math

import numpy as np
import pytest

from conftest import SEED
from qkdsim.analysis.blinding import blinding_model
from qkdsim.analysis.intercept_resend import intercept_resend_model
from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.attacks.blinding import DetectorBlinding, matching_send_fraction
from qkdsim.attacks.intercept_resend import InterceptResend
from qkdsim.channel import FiberChannel
from qkdsim.detector import CLAVIS2, DetectorPair, dark_count_per_detector
from qkdsim.metrics import eve_known_fraction, gain, matched_qber
from qkdsim.pipeline import Link, run_counts_parallel
from qkdsim.source import WCPSource

Z_MAX = 4.0
MU, LENGTH = 0.48, 25.0
P_D = dark_count_per_detector(GYS.background)


@pytest.mark.slow
@pytest.mark.parametrize("position", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
def test_intercept_resend_grid(fraction: float, position: float) -> None:
    link = Link(
        WCPSource((MU,)),
        FiberChannel(LENGTH, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
        eve=InterceptResend(fraction, position),
    )
    seed = np.random.SeedSequence([SEED, int(100 * fraction), int(10 * position)])
    counts = run_counts_parallel(link, 20_000_000, seed, chunk_size=5_000_000)
    model = intercept_resend_model(GYS, LENGTH, MU, fraction, position, P_D)
    assert abs(gain(counts).z_score(model.clicks.detected)) < Z_MAX
    assert abs(matched_qber(counts).z_score(model.clicks.qber)) < Z_MAX
    if fraction > 0:
        assert abs(eve_known_fraction(counts).z_score(model.known_fraction)) < Z_MAX


# --- atak 2: oślepianie -----------------------------------------------------------------

BLIND_POWERS = [0.6e-3, 0.85e-3, 1.0e-3, 1.1e-3, 1.25e-3, 1.5e-3, 2.0e-3]


@pytest.mark.slow
@pytest.mark.parametrize("calibrated", [True, False])
@pytest.mark.parametrize("power", BLIND_POWERS)
def test_blinding_grid(power: float, calibrated: bool) -> None:
    detectors = DetectorPair(GYS.eta_bob, P_D, CLAVIS2)
    honest = matched_basis_clicks(MU, GYS.eta(LENGTH), P_D, GYS.misalignment)
    eve = DetectorBlinding(
        power,
        send_fraction=matching_send_fraction(honest.detected, MU),
        mimic_qber=honest.qber,
        compensation_rad=math.asin(math.sqrt(GYS.misalignment)) if calibrated else 0.0,
    )
    link = Link(
        WCPSource((MU,)),
        FiberChannel(LENGTH, GYS.alpha_db_per_km, GYS.misalignment),
        detectors,
        eve=eve,
    )
    seed = np.random.SeedSequence([SEED, int(power * 1e6), int(calibrated)])
    counts = run_counts_parallel(link, 8_000_000, seed, chunk_size=2_000_000)
    model = blinding_model(detectors, eve, MU, GYS.misalignment)
    if model.detected == 0.0:
        # Impuls poniżej P_never obu detektorów: nic nie klika ani w modelu, ani w MC.
        assert counts.detected.sum() == 0
        return
    assert abs(gain(counts).z_score(model.detected)) < Z_MAX
    assert abs(matched_qber(counts).z_score(model.qber)) < Z_MAX
