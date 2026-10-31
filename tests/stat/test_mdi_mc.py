"""MDI-QKD: Monte Carlo przekaźnika kontra model i Ewa jako przekaźnik (F7)."""

import math

import pytest

from conftest import SEED
from qkdsim.analysis.mdi import coherent_gains
from qkdsim.mdi import MDILink, MeasureAndAnnounce, Relay, run_mdi
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

Z_MAX = 4.0
RELAY = Relay.with_misalignment(0.145, 3.01e-6, 0.015)
INTENSITIES = (0.4, 0.1)
LENGTH = 10.0


def test_gains_and_errors_match_model() -> None:
    source = WCPSource(INTENSITIES, (0.5, 0.5))
    link = MDILink(source, source, LENGTH / 2, LENGTH / 2, RELAY)
    counts = run_mdi(link, 20_000_000, RandomStreams.from_seed(SEED))
    t = link.transmittance(LENGTH / 2)
    for i, mu_a in enumerate(INTENSITIES):
        for j, mu_b in enumerate(INTENSITIES):
            for basis in (0, 1):
                gain, error = coherent_gains(mu_a * t, mu_b * t, RELAY, basis)
                sent, hits = counts.sent[i, j, basis], counts.detected[i, j, basis]
                wrong = counts.errors[i, j, basis]
                assert abs(hits - sent * gain) < Z_MAX * math.sqrt(sent * gain)
                assert abs(wrong - hits * error) < Z_MAX * math.sqrt(hits * error * (1 - error))


def test_relay_attack_is_seen_in_the_test_basis() -> None:
    source = WCPSource((0.4,))
    link = MDILink(source, source, LENGTH / 2, LENGTH / 2, RELAY, attack=MeasureAndAnnounce())
    counts = run_mdi(link, 2_000_000, RandomStreams.from_seed(SEED))
    key_hits, test_hits = counts.detected[0, 0, 0], counts.detected[0, 0, 1]
    assert counts.errors[0, 0, 0] == 0  # baza klucza: bez błędów
    assert counts.eve_correct[0, 0, 0] == key_hits  # Ewa zna każdy bit klucza
    sigma = math.sqrt(0.25 / test_hits)
    assert counts.errors[0, 0, 1] / test_hits == pytest.approx(0.5, abs=Z_MAX * sigma)
