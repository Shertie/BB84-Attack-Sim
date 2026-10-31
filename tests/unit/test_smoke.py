"""Test dymny: pakiet się importuje, a fiksura RNG jest deterministyczna."""

import importlib
import tomllib
from pathlib import Path

import numpy as np
import pytest

import qkdsim


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text("utf-8"))
    assert qkdsim.__version__ == pyproject["project"]["version"]


@pytest.mark.parametrize("name", ["postprocessing", "attacks", "defenses", "analysis"])
def test_subpackages_import(name: str) -> None:
    module = importlib.import_module(f"qkdsim.{name}")
    assert module.__name__ == f"qkdsim.{name}" and module.__doc__


def test_rng_fixture_is_deterministic(rng: np.random.Generator) -> None:
    from conftest import SEED

    expected = np.random.default_rng(SEED).integers(0, 2, 16)
    assert np.array_equal(rng.integers(0, 2, 16), expected)
