"""Spójność analitycznych wartości kontrolnych z golden.py (bez qkd)."""

import math

import pytest
from scipy.optimize import brentq

import golden as g


def _h(x: float) -> float:
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def test_shor_preskill_threshold() -> None:
    root = brentq(lambda q: 1.0 - 2.0 * _h(q), 0.01, 0.3)
    assert root == pytest.approx(g.SHOR_PRESKILL_QBER, abs=g.ANALYTIC_TOLERANCE)


def test_fuchs_crossing() -> None:
    # I_AB = 1 - h(D) i I_AE = 1 - h(1/2 - sqrt(D(1 - D))) przecinają się w D*.
    root = brentq(lambda d: _h(0.5 - math.sqrt(d * (1.0 - d))) - _h(d), 0.01, 0.2)
    assert root == pytest.approx(g.FUCHS_CROSSING_D, abs=g.ANALYTIC_TOLERANCE)
    assert (1.0 - 1.0 / math.sqrt(2.0)) / 2.0 == pytest.approx(root, abs=1e-12)


@pytest.mark.parametrize(
    ("f_ec", "expected"), [(g.GYS_F_EC, g.MU_OPT_F_EC), (1.0, g.MU_OPT_SHANNON)]
)
def test_optimal_mu_equation_12(f_ec: float, expected: float) -> None:
    # Ma et al. 2005, wzór (12): (1 - mu) exp(-mu) = f h(e_d) / (1 - h(e_d)).
    want = f_ec * _h(g.GYS_E_DETECTOR) / (1.0 - _h(g.GYS_E_DETECTOR))
    mu = brentq(lambda m: (1.0 - m) * math.exp(-m) - want, 0.0, 1.0)
    assert mu == pytest.approx(expected, abs=g.MU_OPT_TOLERANCE)
