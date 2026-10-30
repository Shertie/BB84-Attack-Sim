"""Struktura Pulses: walidacja, wartości domyślne, niezmienność (ADR 0001)."""

import numpy as np
import pytest

from qkdsim.pulses import Pulses


def _pulses(size: int = 4) -> Pulses:
    return Pulses.build(
        n=np.ones(size, dtype=np.int64),
        bit=np.zeros(size, dtype=np.uint8),
        basis=np.zeros(size, dtype=np.uint8),
        theta=np.zeros(size),
    )


def test_build_fills_defaults() -> None:
    pulses = _pulses(4)
    assert len(pulses) == 4
    assert np.array_equal(pulses.intensity, np.zeros(4, dtype=np.uint8))
    assert np.array_equal(pulses.t, np.zeros(4))
    assert np.all(np.isnan(pulses.power))


def test_build_converts_python_lists() -> None:
    pulses = Pulses.build(n=[1, 2], bit=[0, 1], basis=[1, 0], theta=[0.0, 1.0])
    assert pulses.n.dtype == np.int64
    assert pulses.bit.dtype == np.uint8
    assert pulses.theta.dtype == np.float64


def test_build_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="nie mieszczą się"):
        Pulses.build(n=[1], bit=[256], basis=[0], theta=[0.0])


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="długość"):
        Pulses.build(n=[1, 1], bit=[0], basis=[0, 0], theta=[0.0, 0.0])


def test_wrong_dtype_is_rejected() -> None:
    pulses = _pulses(2)
    with pytest.raises(TypeError, match="dtype"):
        pulses.replace(bit=np.zeros(2, dtype=np.int64))


def test_two_dimensional_array_is_rejected() -> None:
    pulses = _pulses(2)
    with pytest.raises(ValueError, match="jednowymiarowa"):
        pulses.replace(theta=np.zeros((2, 1)))


def test_arrays_are_read_only() -> None:
    pulses = _pulses(3)
    with pytest.raises(ValueError, match="read-only"):
        pulses.theta[0] = 1.0


def test_replace_leaves_original_unchanged() -> None:
    pulses = _pulses(3)
    shifted = pulses.replace(t=np.full(3, 0.5))
    assert np.array_equal(pulses.t, np.zeros(3))
    assert np.array_equal(shifted.t, np.full(3, 0.5))
