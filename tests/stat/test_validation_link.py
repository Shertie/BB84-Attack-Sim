"""Walidacja toru: MC zgodne z modelem na 0–150 km dla parametrów GYS.

Testy wolne (`make test-slow`): łącznie około 8·10⁸ impulsów, liczone równolegle.
"""

import numpy as np
import pytest

from conftest import SEED
from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.wcp import gain as gain_model
from qkdsim.analysis.wcp import matched_basis_clicks, qber
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.metrics import gain, matched_qber
from qkdsim.pipeline import Link, run_counts_parallel
from qkdsim.source import WCPSource

Z_MAX = 4.0
MU = 0.48
P_D = dark_count_per_detector(GYS.background)

# Liczba impulsów rośnie z dystansem, żeby przy 150 km zostało ~3500 detekcji.
PULSES = {0.0: 10**7, 25.0: 10**7, 50.0: 2 * 10**7, 75.0: 4 * 10**7,
          100.0: 10**8, 125.0: 2 * 10**8, 150.0: 4 * 10**8}  # fmt: skip


@pytest.mark.slow
@pytest.mark.parametrize("length_km", sorted(PULSES))
def test_gain_and_qber_agree_with_model(length_km: float) -> None:
    link = Link(
        WCPSource((MU,)),
        FiberChannel(length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
    )
    seed = np.random.SeedSequence([SEED, int(length_km)])
    counts = run_counts_parallel(link, PULSES[length_km], seed)
    eta = GYS.eta(length_km)
    model = matched_basis_clicks(MU, eta, P_D, GYS.misalignment)
    assert abs(gain(counts).z_score(model.detected)) < Z_MAX
    assert abs(matched_qber(counts).z_score(model.qber)) < Z_MAX
    # Model z artykułu (Ma et al.) jest tak bliski, że przechodzi ten sam test.
    assert abs(gain(counts).z_score(gain_model(MU, eta, GYS.background))) < Z_MAX
    assert (
        abs(matched_qber(counts).z_score(qber(MU, eta, GYS.background, GYS.misalignment))) < Z_MAX
    )
    for basis in (0, 1):
        assert abs(matched_qber(counts, basis=basis).z_score(model.qber)) < Z_MAX
