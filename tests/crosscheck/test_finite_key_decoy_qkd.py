"""Skończony klucz z decoy: wzory (1)–(5) Lima et al. 2014 kontra niezależna implementacja `qkd`."""

import pytest

pytest.importorskip("qkd", reason="testy krzyżowe wymagają qkd: pip install -e '.[crosscheck]'")

from qkd import _core as core  # wzory Lima w qkd 0.2.0

from qkdsim.analysis.finite_key_decoy import LimChannel, ProtocolChoice, expected_counts
from qkdsim.postprocessing.finite_key_decoy import (
    DecoySettings,
    hoeffding,
    key_length_decoy,
    single_photon_errors_upper,
    single_photon_lower,
    vacuum_lower,
)

CASES = [
    (ProtocolChoice(0.8, DecoySettings((0.5, 0.1, 2e-4), (0.6, 0.3, 0.1))), 25.0, 1e6),
    (ProtocolChoice(0.4, DecoySettings((0.49, 0.11, 2e-4), (0.05, 0.55, 0.4))), 120.0, 1e4),
    (ProtocolChoice(0.93, DecoySettings((0.44, 0.17, 2e-4), (0.76, 0.15, 0.09))), 100.0, 1e9),
]
EPS_SEC, EPS_COR, F_EC = 1e-10, 1e-15, 1.16


@pytest.mark.parametrize(("choice", "length_km", "n_key"), CASES)
def test_every_step_matches_qkd(choice: ProtocolChoice, length_km: float, n_key: float) -> None:
    key, test, _ = expected_counts(choice, LimChannel(length_km), n_key)
    mu1, mu2, mu3 = choice.settings.intensities
    probs = choice.settings.probabilities
    eps = EPS_SEC / 21
    delta = hoeffding(key.total, EPS_SEC)
    s0 = vacuum_lower(key.detections, choice.settings, delta)
    s1 = single_photon_lower(key.detections, choice.settings, delta, s0)
    assert (s0, s1) == pytest.approx(
        core.decoy_counts(mu1, mu2, mu3, probs, key.detections, eps), rel=1e-12, abs=1e-9
    )
    delta_test = hoeffding(test.total, EPS_SEC)
    s1_test = single_photon_lower(
        test.detections,
        choice.settings,
        delta_test,
        vacuum_lower(test.detections, choice.settings, delta_test),
    )
    assert s1_test == pytest.approx(
        core.decoy_counts(mu1, mu2, mu3, probs, test.detections, eps)[1], rel=1e-12
    )
    v1 = single_photon_errors_upper(
        test.errors, choice.settings, hoeffding(sum(test.errors), EPS_SEC)
    )
    assert v1 == pytest.approx(core.decoy_errors(mu1, mu2, mu3, probs, test.errors, eps), rel=1e-12)
    ours = key_length_decoy(key, test, choice.settings, F_EC, EPS_SEC, EPS_COR)
    phi = core.bb84_phase(v1, s1_test, s1, EPS_SEC)
    assert ours.phase_error == pytest.approx(phi, rel=1e-12)
    theirs = core.bb84_length(s0, s1, phi, key.total, key.error_rate, F_EC, EPS_SEC, EPS_COR)
    assert max(ours.value, 0.0) == pytest.approx(theirs, rel=1e-9, abs=1.0)
