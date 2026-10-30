"""Atak 6: optymalny atak indywidualny (Fuchs et al. 1997), F4.7 i F4.9.

Ewa splata każdy foton z własnym kubitem (sondą, początkowo |0⟩) fazowo-kowariantnym
klonerem: częściową zamianą w podprzestrzeni {|01⟩, |10⟩} o kąt η,
|00⟩ → |00⟩, |10⟩ → cos η|10⟩ + sin η|01⟩, |01⟩ → cos η|01⟩ − sin η|10⟩, |11⟩ → |11⟩.
Kloner jest optymalny dla stanów z równika sfery Blocha, więc Ewa najpierw obraca foton
unitarną V (obrót o 120° wokół osi (1, 1, 1): z → x, x → y), a po klonowaniu odwraca obrót
na obu kubitach. Bazy Z i X protokołu trafiają wtedy na równik.

Bob ma błąd D = sin²(η/2). Ewa po ujawnieniu baz mierzy sondę w bazie Alice z błędem
½(1 − sin η) = ½ − √(D(1 − D)), czyli dokładnie na granicy Fuchsa.

Atak działa na pojedynczych fotonach i idealnych detektorach (granica teoretyczna), dlatego
ma własny sampler zamiast toru impulsów: prawdopodobieństwa wyników liczy się raz dla czterech
stanów wejściowych, a potem losuje wektorowo (F4.9).
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qkdsim.rng import Bits

Complex = npt.NDArray[np.complex128]

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Obrót o 120° wokół (1, 1, 1)/√3: V Z V† = X, V X V† = Y.
FRAME = 0.5 * (_I - 1j * _X - 1j * _Y - 1j * _Z)

# Stany BB84 i bazy pomiarowe: [baza][bit] → wektor.
_KETS = np.array(
    [
        [[1, 0], [0, 1]],
        [[1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]],
    ],
    dtype=np.complex128,
)


def eta_for_error(error: float) -> float:
    """η z D = sin²(η/2), D ∈ [0, ½]."""
    if not 0.0 <= error <= 0.5:
        raise ValueError("błąd Boba musi być w [0, ½]")
    return 2.0 * math.asin(math.sqrt(error))


def partial_swap(eta: float) -> Complex:
    """Kloner fazowo-kowariantny w bazie |sygnał, sonda⟩ = |00⟩, |01⟩, |10⟩, |11⟩."""
    c, s = math.cos(eta), math.sin(eta)
    u = np.eye(4, dtype=np.complex128)
    u[1, 1], u[2, 1] = c, -s  # obraz |01⟩
    u[1, 2], u[2, 2] = s, c  # obraz |10⟩
    return u


def attack_unitary(error: float) -> Complex:
    """Całkowita operacja Ewy na (foton, sonda |0⟩) dla zadanego błędu Boba."""
    back = FRAME.conj().T
    return np.kron(back, back) @ partial_swap(eta_for_error(error)) @ np.kron(FRAME, _I)


@dataclass(frozen=True)
class OptimalIndividualAttack:
    """Atak 6: kloner fazowo-kowariantny z błędem D u Boba (granica Fuchsa)."""

    error: float  # D: błąd, który Ewa wprowadza u Boba

    def joint_probabilities(self) -> npt.NDArray[np.float64]:
        """P[baza, bit Alice, bit Boba, bit Ewy] w zgodnej bazie Boba; Ewa mierzy w bazie Alice."""
        u = attack_unitary(self.error)
        probs = np.zeros((2, 2, 2, 2))
        for basis in (0, 1):
            for bit in (0, 1):
                state = u @ np.kron(_KETS[basis, bit], np.array([1, 0], dtype=np.complex128))
                for bob in (0, 1):
                    for eve in (0, 1):
                        outcome = np.kron(_KETS[basis, bob], _KETS[basis, eve])
                        probs[basis, bit, bob, eve] = abs(np.vdot(outcome, state)) ** 2
        return probs

    def sample(self, n: int, rng: np.random.Generator) -> tuple[Bits, Bits, Bits, Bits]:
        """(bit Alice, baza, bit Boba, bit Ewy) dla n fotonów po siftingu."""
        bits = rng.integers(0, 2, n, dtype=np.uint8)
        bases = rng.integers(0, 2, n, dtype=np.uint8)
        table = self.joint_probabilities().reshape(2, 2, 4)  # wynik = 2·bob + ewa
        cumulative = np.cumsum(table, axis=2)[bases, bits]
        outcome = (rng.random(n)[:, None] >= cumulative[:, :3]).sum(axis=1).astype(np.uint8)
        return bits, bases, outcome >> 1, outcome & 1


def mutual_information(x: Bits, y: Bits) -> float:
    """Estymator „plug-in” I(X; Y) dla dwóch ciągów bitów."""
    joint = np.bincount(2 * x.astype(np.intp) + y, minlength=4).reshape(2, 2) / len(x)
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    info = 0.0
    for i in (0, 1):
        for j in (0, 1):
            if joint[i, j] > 0:
                info += joint[i, j] * math.log2(joint[i, j] / (px[i] * py[j]))
    return info
