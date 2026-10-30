"""Bramka G3 (ROADMAP, rozdz. 6), część dla ataku 1: siatka p × położenie Ewy przy 25 km.

Wolne testy (`make test-slow`). Części dla ataków 2 i 3 dojdą razem z nimi.
"""

import numpy as np
import pytest

from conftest import SEED
from qkdsim.analysis.intercept_resend import intercept_resend_model
from qkdsim.analysis.key_rate import GYS
from qkdsim.attacks.intercept_resend import InterceptResend
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair, dark_count_per_detector
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
