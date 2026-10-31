"""Splątane pary: zysk i QBER źródła SPDC (Ma, Fung, Lo 2007), CHSH i granica Acína et al.
kontra `qkd`. Zysk jest dokładny; QBER Ma i in. to przybliżenie (koincydencje nieskorelowane
mylą się z ½), więc tu tolerancja 0,1%."""

import pytest

pytest.importorskip("qkd", reason="testy krzyżowe wymagają qkd: pip install -e '.[crosscheck]'")

from qkd import _core as core  # wzory Ma, Funga i Lo w qkd 0.2.0
from qkd.pairs import DI_ETA_SINGLET

from qkdsim.analysis.entangled import (
    chsh_value,
    device_independent_threshold,
    holevo_bound,
    pair_statistics,
)

CASES = [
    (0.01, 0.1, 0.05, 1e-6, 0.0),
    (0.1, 0.3, 0.2, 1e-5, 0.03),
    (0.5, 0.045, 0.045, 8.5e-7, 0.033),
]


@pytest.mark.parametrize(("lam", "eta_a", "eta_b", "dark", "misalignment"), CASES)
def test_pair_source_matches_ma_fung_lo(
    lam: float, eta_a: float, eta_b: float, dark: float, misalignment: float
) -> None:
    stats = pair_statistics(lam, eta_a, eta_b, dark, misalignment)
    y0 = 1 - (1 - dark) ** 2  # tło odbiornika z dwóch detektorów
    assert stats.gain == pytest.approx(core.pair_gain(lam, eta_a, eta_b, y0, y0), rel=1e-10)
    assert stats.qber == pytest.approx(
        core.pair_error(lam, eta_a, eta_b, y0, y0, misalignment), rel=1e-3
    )


def test_chsh_and_holevo_match() -> None:
    for v in (1.0, 0.9, 0.75):
        assert chsh_value(1.0, v) == pytest.approx(core.pair_chsh(v))
    for s in (2.1, 2.5, 2.8):
        assert holevo_bound(s) == pytest.approx(core.pair_holevo(s))
    assert device_independent_threshold() == pytest.approx(DI_ETA_SINGLET, abs=2e-3)
