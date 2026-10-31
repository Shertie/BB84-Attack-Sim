"""Atak 4: Monte Carlo vs model time-shift i obrona losowym przypisaniem (F6.2–F6.4).

Krzywe dopasowane do Zhao et al. 2008, GYS przy 75 km, Ewa tuż za Alice z linią bezstratną
i tłumieniem, które wyrównuje zysk Boba.
"""

import numpy as np
import pytest
from scipy.stats import chi2_contingency

from conftest import SEED
from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.time_shift import (
    eve_information,
    fit_zhao,
    gain_matching_transmission,
    time_shift_model,
)
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.attacks.time_shift import TimeShift, detection_table
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

Z_MAX = 4.0
MU, LENGTH, PULSES = 0.48, 75.0, 20_000_000
P_D = dark_count_per_detector(GYS.background)
FIT = fit_zhao()
T = 10.0 ** (-GYS.alpha_db_per_km * LENGTH / 10.0)


def setup(swap: bool) -> tuple[Link, float, float]:
    transmission = gain_matching_transmission(
        MU, T, GYS.eta_bob, P_D, GYS.misalignment, FIT.mismatch, FIT.shifts_ns
    )
    assert transmission is not None
    p_a = time_shift_model(
        MU, transmission, GYS.eta_bob, P_D, GYS.misalignment, FIT.mismatch, FIT.shifts_ns
    ).p_a
    link = Link(
        WCPSource((MU,)),
        FiberChannel(LENGTH, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D, timing=FIT.mismatch),
        eve=TimeShift(*FIT.shifts_ns, p_a=p_a, transmission=transmission),
        random_assignment=swap,
    )
    return link, transmission, p_a


@pytest.mark.parametrize("swap", [False, True])
def test_detections_match_model(swap: bool) -> None:
    link, transmission, p_a = setup(swap)
    table, matched = detection_table(link, PULSES, RandomStreams.from_seed(SEED))
    model = time_shift_model(
        MU, transmission, GYS.eta_bob, P_D, GYS.misalignment, FIT.mismatch, FIT.shifts_ns, p_a, swap
    )
    expected = model.table * matched
    observed = table.sum(axis=1)  # model nie rozróżnia baz
    z = (observed - expected) / np.sqrt(expected)
    assert np.abs(z).max() < Z_MAX, z


def test_qber_and_gain_unchanged_while_eve_learns() -> None:
    link, _, _ = setup(False)
    table, matched = detection_table(link, PULSES, RandomStreams.from_seed(SEED))
    honest = matched_basis_clicks(MU, GYS.eta(LENGTH), P_D, GYS.misalignment)
    detected = int(table.sum())
    errors = int(table[..., 0, 1].sum() + table[..., 1, 0].sum())
    expected = honest.detected * matched
    assert abs(detected - expected) < Z_MAX * np.sqrt(expected)
    sigma_qber = np.sqrt(honest.qber * (1 - honest.qber) / detected)
    assert abs(errors / detected - honest.qber) < Z_MAX * sigma_qber
    assert eve_information(table) > 0.25


def test_random_assignment_leaves_eve_nothing() -> None:
    """Bit Alice jest niezależny od tego, co Ewa wie (przesunięcie i baza)."""
    link, _, _ = setup(True)
    table, _ = detection_table(link, PULSES, RandomStreams.from_seed(SEED))
    by_knowledge = table.sum(axis=-1).reshape(4, 2)  # [przesunięcie·baza, bit Alice]
    assert chi2_contingency(by_knowledge).pvalue > 1e-3
