"""Atak 6: weryfikacja w QuTiP (F4.8, ADR 0004).

Unitarna Ewy jest tu zbudowana niezależnie, z operatorów QuTiP: kloner jako exp(ηA),
A = |01⟩⟨10| − |10⟩⟨01|, obrót układu jako exp(−i(π/3)(X + Y + Z)/√3). Wierności Boba i Ewy
liczone są z śladów częściowych, a nie z prawdopodobieństw wyników jak w numpy.
"""

import math
from typing import Any

import numpy as np
import pytest

qutip = pytest.importorskip("qutip", reason="weryfikacja wymaga QuTiP: pip install -e '.[verify]'")

from qkdsim.analysis.individual import eve_error_individual  # noqa: E402
from qkdsim.attacks.optimal_individual import OptimalIndividualAttack, attack_unitary  # noqa: E402

ERRORS = [0.0, 0.03, 0.11, 0.1464, 0.25, 0.4]


def qutip_unitary(error: float) -> Any:
    eta = 2 * math.asin(math.sqrt(error))
    generator = qutip.tensor(qutip.sigmap(), qutip.sigmam()) - qutip.tensor(
        qutip.sigmam(), qutip.sigmap()
    )
    cloner = (eta * generator).expm()
    axis = (qutip.sigmax() + qutip.sigmay() + qutip.sigmaz()) / math.sqrt(3)
    frame = (-1j * (math.pi / 3) * axis).expm()
    identity = qutip.qeye(2)
    return qutip.tensor(frame.dag(), frame.dag()) * cloner * qutip.tensor(frame, identity)


def bb84_states() -> list[tuple[int, int, Any]]:
    zero, one = qutip.basis(2, 0), qutip.basis(2, 1)
    plus, minus = (zero + one).unit(), (zero - one).unit()
    return [(0, 0, zero), (0, 1, one), (1, 0, plus), (1, 1, minus)]


@pytest.mark.parametrize("error", ERRORS)
def test_numpy_and_qutip_unitaries_agree(error: float) -> None:
    assert np.allclose(qutip_unitary(error).full(), attack_unitary(error), atol=1e-12)


@pytest.mark.parametrize("error", ERRORS)
def test_reduced_states_give_fuchs_fidelities(error: float) -> None:
    """Bob: wierność 1 − D; Ewa (sonda zmierzona w bazie Alice): 1 − (½ − √(D(1 − D)))."""
    u = qutip_unitary(error)
    probe = qutip.basis(2, 0)
    for _basis, _bit, state in bb84_states():
        out = u * qutip.tensor(state, probe)
        bob, eve = out.ptrace(0), out.ptrace(1)
        bob_fidelity = qutip.expect(bob, state)
        eve_fidelity = qutip.expect(eve, state)
        assert bob_fidelity == pytest.approx(1 - error, abs=1e-10)
        assert eve_fidelity == pytest.approx(1 - eve_error_individual(error), abs=1e-10)


@pytest.mark.parametrize("error", ERRORS)
def test_joint_probabilities_match_qutip_projectors(error: float) -> None:
    probs = OptimalIndividualAttack(error).joint_probabilities()
    u = qutip_unitary(error)
    states = {(b, x): s for b, x, s in bb84_states()}
    for (basis, bit), state in states.items():
        out = u * qutip.tensor(state, qutip.basis(2, 0))
        for bob in (0, 1):
            for eve in (0, 1):
                projector = qutip.tensor(states[basis, bob].proj(), states[basis, eve].proj())
                expected = qutip.expect(projector, out)
                assert probs[basis, bit, bob, eve] == pytest.approx(expected, abs=1e-10)
