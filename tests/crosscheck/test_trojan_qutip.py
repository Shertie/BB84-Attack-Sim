"""Atak 5: Helstrom dla baz zweryfikowany w QuTiP (F6.6, ADR 0004).

Stany sondy są tu zbudowane niezależnie od numpy: stan koherentny jako iloczyn stanów
`coherent` w modach H i V, stan Focka |n_θ⟩ z operatora kreacji (cos θ a_H† + sin θ a_V†)ⁿ.
Przestrzeń jest obcięta inaczej (każdy mod osobno, a nie łączna liczba fotonów). Odległość
śladową liczymy z wartości własnych różnicy stanów: `tracedist` bierze pierwiastki wartości
własnych ρ†ρ, więc szum 10⁻¹⁶ w setkach wymiarów daje mu błąd rzędu 10⁻⁷.
"""

import math
from typing import Any

import pytest

qutip = pytest.importorskip("qutip", reason="weryfikacja wymaga QuTiP: pip install -e '.[verify]'")

from qkdsim.analysis.trojan_horse import (  # noqa: E402
    BASIS_ANGLES,
    basis_guess_probability,
    bit_guess_probability,
)

CUTOFF = 22  # stanów na mod


def probe(mu: float, theta: float, random_phase: bool) -> Any:
    alpha = math.sqrt(mu)
    if not random_phase:
        h = qutip.coherent(CUTOFF, alpha * math.cos(theta), method="analytic")
        v = qutip.coherent(CUTOFF, alpha * math.sin(theta), method="analytic")
        state = qutip.tensor(h, v)
        return state * state.dag()
    a_h = qutip.tensor(qutip.destroy(CUTOFF), qutip.qeye(CUTOFF))
    a_v = qutip.tensor(qutip.qeye(CUTOFF), qutip.destroy(CUTOFF))
    creation = (math.cos(theta) * a_h + math.sin(theta) * a_v).dag()
    vacuum = qutip.tensor(qutip.basis(CUTOFF, 0), qutip.basis(CUTOFF, 0))
    rho = 0 * vacuum * vacuum.dag()
    state = vacuum
    for n in range(CUTOFF):
        rho += math.exp(-mu) * mu**n / math.factorial(n) * state * state.dag()
        state = creation * state / math.sqrt(n + 1)
    return rho


def trace_distance(rho: Any, sigma: Any) -> float:
    return 0.5 * float(sum(abs(value) for value in (rho - sigma).eigenenergies()))


def basis_state(mu: float, name: str, random_phase: bool) -> Any:
    first, second = BASIS_ANGLES[name]
    return 0.5 * (probe(mu, first, random_phase) + probe(mu, second, random_phase))


@pytest.mark.parametrize("mu", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("random_phase", [False, True])
def test_basis_helstrom_matches_qutip(mu: float, random_phase: bool) -> None:
    distance = trace_distance(
        basis_state(mu, "Z", random_phase), basis_state(mu, "X", random_phase)
    )
    assert basis_guess_probability(mu, random_phase) == pytest.approx(
        0.5 + 0.5 * distance, abs=1e-9
    )


@pytest.mark.parametrize("mu", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("random_phase", [False, True])
def test_bit_helstrom_matches_qutip(mu: float, random_phase: bool) -> None:
    distance = trace_distance(probe(mu, 0.0, random_phase), probe(mu, math.pi / 2, random_phase))
    assert bit_guess_probability(mu, random_phase) == pytest.approx(0.5 + 0.5 * distance, abs=1e-9)
