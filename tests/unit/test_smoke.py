"""Test dymny: pakiet się importuje, a fiksura RNG jest deterministyczna."""

import importlib

import numpy as np
import pytest

import qkdsim


def test_version_is_set() -> None:
    assert qkdsim.__version__


@pytest.mark.parametrize("name", ["postprocessing", "attacks", "defenses", "analysis"])
def test_subpackages_import(name: str) -> None:
    importlib.import_module(f"qkdsim.{name}")


def test_rng_fixture_is_deterministic(rng: np.random.Generator) -> None:
    from conftest import SEED

    expected = np.random.default_rng(SEED).integers(0, 2, 16)
    assert np.array_equal(rng.integers(0, 2, 16), expected)
