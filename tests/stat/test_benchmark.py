"""Benchmark warstwy statystyk.

Punkt odniesienia z 30.10.2026 (12 rdzeni, Python 3.14): ~92 ns na impuls w jednym procesie.
Próg testu jest celowo luźny (5×), bo CI ma wolniejsze maszyny; łapie tylko duże regresje,
np. przypadkową pętlę po impulsach w Pythonie.
"""

import time

import pytest

from conftest import SEED
from qkdsim.analysis.key_rate import GYS
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link, run_counts
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

BASELINE_NS_PER_PULSE = 92.0
TOLERANCE = 5.0


@pytest.mark.slow
def test_statistics_layer_throughput() -> None:
    link = Link(
        WCPSource((0.48,)),
        FiberChannel(50.0, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, dark_count_per_detector(GYS.background)),
    )
    n = 10**7
    start = time.perf_counter()
    run_counts(link, n, RandomStreams.from_seed(SEED))
    ns_per_pulse = (time.perf_counter() - start) / n * 1e9
    assert ns_per_pulse < TOLERANCE * BASELINE_NS_PER_PULSE, f"{ns_per_pulse:.0f} ns/impuls"
