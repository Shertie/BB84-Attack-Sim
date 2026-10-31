"""QKD ze splątanymi parami: BBM92 ze źródłem SPDC, CHSH i device-independent."""

import math

import pytest

from golden import PIRONIO_DI_EFFICIENCY, PIRONIO_ROUNDING
from qkdsim.analysis.entangled import (
    chsh_value,
    device_independent_rate,
    device_independent_threshold,
    holevo_bound,
    key_setting_qber,
    optimal_bbm92,
    pair_statistics,
    violation_threshold,
)


def test_weak_source_errors_come_from_misalignment() -> None:
    stats = pair_statistics(1e-4, 0.5, 0.5, 0.0, 0.03)
    assert stats.qber == pytest.approx(0.03, abs=1e-4)  # wiele par dokłada O(λ)
    assert stats.gain == pytest.approx(2 * 1e-4 * 0.25, rel=1e-3)  # 2λ par, obie strony 0,5


def test_multi_pair_emission_raises_errors() -> None:
    low = pair_statistics(0.01, 0.5, 0.5, 0.0, 0.0).qber
    high = pair_statistics(0.5, 0.5, 0.5, 0.0, 0.0).qber
    assert low < high < 0.5


def test_chsh_limits() -> None:
    assert chsh_value(1.0) == pytest.approx(2 * math.sqrt(2))
    assert chsh_value(0.0) == pytest.approx(2.0)  # same braki kliknięć: lokalny model
    assert chsh_value(1.0, visibility=1 / math.sqrt(2)) == pytest.approx(2.0)
    assert key_setting_qber(1.0) == 0.0
    assert key_setting_qber(0.5) == pytest.approx(0.25)


def test_holevo_bound_limits() -> None:
    assert holevo_bound(2.0) == 1.0
    assert holevo_bound(2 * math.sqrt(2)) == pytest.approx(0.0, abs=1e-12)
    assert device_independent_rate(0.0, 2 * math.sqrt(2)) == pytest.approx(1.0)


def test_detection_thresholds() -> None:
    """S > 2 od η = 2/(1 + √2); dodatnia stopa DI od ~92,4% (Pironio i in. 2009)."""
    assert violation_threshold() == pytest.approx(2 / (1 + math.sqrt(2)), abs=1e-9)
    assert device_independent_threshold() == pytest.approx(
        PIRONIO_DI_EFFICIENCY, abs=PIRONIO_ROUNDING
    )
    assert device_independent_threshold(0.99) > device_independent_threshold(1.0)


def test_optimal_brightness_is_interior() -> None:
    rate, lam = optimal_bbm92(0.02, 0.02, 1e-6, 0.03, 1.22)
    assert rate > 0.0
    assert 1e-4 < lam < 0.5
