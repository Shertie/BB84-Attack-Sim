"""Wspólne fiksury testów.

Stałe ziarna sprawiają, że testy statystyczne nie są losowo czerwone (ROADMAP, rozdz. 7.2).
"""

import os

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

SEED = 20261102

settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile(
    "ci", max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)
