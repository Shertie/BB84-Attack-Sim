"""Atak 6: kloner, granica Fuchsa i Monte Carlo."""

import math

import numpy as np
import pytest

import golden as g
from conftest import SEED
from qkdsim.analysis.individual import (
    FUCHS_CROSSING,
    bob_information,
    eve_error_individual,
    eve_information_individual,
    eve_information_intercept_resend,
)
from qkdsim.attacks.optimal_individual import (
    FRAME,
    OptimalIndividualAttack,
    attack_unitary,
    eta_for_error,
    mutual_information,
)

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])


def test_frame_rotation_puts_z_and_x_on_the_equator() -> None:
    assert np.allclose(FRAME @ Z @ FRAME.conj().T, X)
    assert np.allclose(FRAME @ X @ FRAME.conj().T, Y)


@pytest.mark.parametrize("error", [0.0, 0.07, 0.25, 0.5])
def test_attack_is_unitary(error: float) -> None:
    u = attack_unitary(error)
    assert np.allclose(u.conj().T @ u, np.eye(4))


def test_eta_mapping_and_validation() -> None:
    assert math.sin(eta_for_error(0.1) / 2) ** 2 == pytest.approx(0.1)
    with pytest.raises(ValueError):
        eta_for_error(0.6)


@pytest.mark.parametrize("error", [0.0, 0.05, 0.11, FUCHS_CROSSING, 0.3, 0.5])
def test_attack_reaches_fuchs_bound_for_all_four_states(error: float) -> None:
    probs = OptimalIndividualAttack(error).joint_probabilities()
    assert probs.sum(axis=(2, 3)) == pytest.approx(np.ones((2, 2)))
    for basis in (0, 1):
        for bit in (0, 1):
            assert probs[basis, bit, 1 - bit, :].sum() == pytest.approx(error, abs=1e-12)
            assert probs[basis, bit, :, 1 - bit].sum() == pytest.approx(
                eve_error_individual(error), abs=1e-12
            )


def test_crossing_is_the_golden_value() -> None:
    assert pytest.approx(g.FUCHS_CROSSING_D, abs=g.ANALYTIC_TOLERANCE) == FUCHS_CROSSING
    assert bob_information(FUCHS_CROSSING) == pytest.approx(
        eve_information_individual(FUCHS_CROSSING), abs=1e-12
    )


def test_intercept_resend_line() -> None:
    assert eve_information_intercept_resend(0.25) == 0.5
    with pytest.raises(ValueError):
        eve_information_intercept_resend(0.3)


@pytest.mark.parametrize("error", [0.05, 0.1464, 0.25])
def test_monte_carlo_matches_formula(error: float) -> None:
    n = 1_000_000
    alice, _bases, bob, eve = OptimalIndividualAttack(error).sample(n, np.random.default_rng(SEED))
    for observed, expected in (
        (np.mean(alice != bob), error),
        (np.mean(alice != eve), eve_error_individual(error)),
    ):
        assert abs(observed - expected) < 4 * math.sqrt(expected * (1 - expected) / n)
    assert mutual_information(alice, bob) == pytest.approx(bob_information(error), abs=5e-3)
    assert mutual_information(alice, eve) == pytest.approx(
        eve_information_individual(error), abs=5e-3
    )
