"""MDI-QKD: zyski, QBER i para pojedynczych fotonów kontra wzory Ma i Razaviego w `qkd`.

`qkd` traktuje błąd optyki fenomenologicznie (odwrócenie bitu z p-stwem e_d), a symulator jako
obroty polaryzacji. Bez błędu optyki oba modele są tym samym i porównujemy je wprost; z błędem
sprawdzamy, że wzór `qkd` to E = e_d(1 − E₀) + (1 − e_d)E₀ z naszym E₀.
"""

import pytest

pytest.importorskip("qkd", reason="testy krzyżowe wymagają qkd: pip install -e '.[crosscheck]'")

from qkd import _core as core  # wzory Ma i Razaviego w qkd 0.2.0 (ADR 0007)

from qkdsim.analysis.mdi import coherent_gains, single_photon_pair
from qkdsim.mdi import Relay

ETA, DARK = 0.145, 3.01e-6
CASES = [(0.5, 0.4, 0.0), (0.3, 0.3, 50.0), (0.1, 0.4, 150.0)]


@pytest.mark.parametrize(("mu_a", "mu_b", "length"), CASES)
def test_coherent_gains_match_ma_razavi(mu_a: float, mu_b: float, length: float) -> None:
    t = 10 ** (-0.2 * length / 20)
    relay = Relay(ETA, DARK)
    for basis, theirs in ((0, core.mdi_rect), (1, core.mdi_diag)):
        gain, error = coherent_gains(mu_a * t, mu_b * t, relay, basis)
        expected_gain, expected_error = theirs(mu_a, mu_b, t * ETA, t * ETA, DARK, 0.0)
        assert gain == pytest.approx(expected_gain, rel=1e-10)
        assert error == pytest.approx(expected_error, rel=1e-9)
        _, with_misalignment = theirs(mu_a, mu_b, t * ETA, t * ETA, DARK, 0.015)
        assert with_misalignment == pytest.approx(0.015 * (1 - error) + 0.985 * error, rel=1e-9)


@pytest.mark.parametrize("length", [0.0, 80.0, 200.0])
def test_single_photon_pair_matches_ma_razavi(length: float) -> None:
    t = 10 ** (-0.2 * length / 20)
    yield_z, _ = single_photon_pair(t, t, Relay(ETA, DARK), 0)
    _, error_x = single_photon_pair(t, t, Relay(ETA, DARK), 1)
    expected_yield, expected_error = core.mdi_yield(t * ETA, t * ETA, DARK, 0.0)
    assert yield_z == pytest.approx(expected_yield, rel=1e-10)
    assert error_x == pytest.approx(expected_error, rel=1e-9)
