"""Model analityczny QKD ze splątanymi parami: BBM92 i CHSH (Ekert 1991, Bennett i in. 1992).

BBM92 ze źródłem SPDC (Ma, Fung, Lo 2007): stan to dwie niezależne pary trybów (A_H, B_V)
i (A_V, B_H), każda o termicznej liczbie par z średnią λ, P(n) = λⁿ/(1 + λ)ⁿ⁺¹. Wykładnik
stanu jest singletem SU(2), więc w każdej wspólnej bazie Alice i Boba liczby fotonów mają ten
sam rozkład: Alice ma n₁ fotonów w trybie 0 i n₂ w trybie 1, Bob odwrotnie. Każdy foton
dociera z p-stwem η po swojej stronie, u Boba trafia do złego detektora z p-stwem e_d, detektory
są progowe z ciemnymi zliczeniami, podwójne kliknięcie daje losowy bit. `pair_statistics` sumuje
to dokładnie po (n₁, n₂). Zysk to wzór (9) Ma i in.; ich QBER (wzór (10)) jest przybliżeniem:
skorelowane koincydencje mylą się z p-stwem e_d, wszystkie inne z ½.

CHSH (urządzenia jak czarne skrzynki): para w stanie o widzialności V, korelacja
E(a, b) = V cos 2(a − b), detektory o sprawności η, brak kliknięcia czytany jako wynik +1.
Wtedy S = 2√2Vη² + 2(1 − η)², a QBER ustawień klucza Q = η²(1 − V)/2 + η(1 − η). Stopa
device-independent przeciw atakom kolektywnym (Acín i in. 2007):
r = 1 − h(Q) − h((1 + √(S²/4 − 1))/2).
"""

import math
from dataclasses import dataclass

from scipy.optimize import brentq, minimize_scalar

from qkdsim.analysis.entropy import h2


def _thermal(lam: float, n: int) -> float:
    return lam**n / (1.0 + lam) ** (n + 1)


def _cutoff(lam: float) -> int:
    """Liczba par, powyżej której ogon rozkładu termicznego jest pomijalny (< 10⁻¹⁵)."""
    if lam <= 0.0:
        return 0
    return max(5, math.ceil(math.log(1e-15) / math.log(lam / (1.0 + lam))))


def _clicks(
    none0: float, none1: float, none_both: float, dark: float
) -> tuple[float, float, float]:
    """P(tylko detektor 0), P(tylko 1), P(oba) z p-stw braku fotonów i ciemnych zliczeń."""
    silent0, silent1 = (1.0 - dark) * none0, (1.0 - dark) * none1
    silent_both = (1.0 - dark) ** 2 * none_both
    return silent1 - silent_both, silent0 - silent_both, 1.0 - silent0 - silent1 + silent_both


@dataclass(frozen=True)
class PairStatistics:
    """Na impuls pompy: p-stwo koincydencji i koincydencji z błędem (po odwróceniu bitu Boba)."""

    gain: float
    errors: float

    @property
    def qber(self) -> float:
        return self.errors / self.gain


def pair_statistics(
    lam: float, eta_a: float, eta_b: float, dark: float, misalignment: float
) -> PairStatistics:
    """Dokładna wartość oczekiwana dla `entangled.simulate_pairs` w zgodnych bazach.

    `dark`: ciemne zliczenia na detektor (dwa detektory po każdej stronie).
    """
    right, wrong = eta_b * (1.0 - misalignment), eta_b * misalignment
    gain = errors = 0.0
    top = _cutoff(lam)
    for n1 in range(top + 1):
        p1 = _thermal(lam, n1)
        for n2 in range(top + 1):
            weight = p1 * _thermal(lam, n2)
            if weight < 1e-18:
                continue
            alice = _clicks((1 - eta_a) ** n1, (1 - eta_a) ** n2, (1 - eta_a) ** (n1 + n2), dark)
            # Bob: n₁ fotonów należy do jego detektora 1, n₂ do detektora 0.
            bob = _clicks(
                (1 - right) ** n2 * (1 - wrong) ** n1,
                (1 - right) ** n1 * (1 - wrong) ** n2,
                (1 - eta_b) ** (n1 + n2),
                dark,
            )
            a0, a1, a_both = alice
            b0, b1, b_both = bob
            detected = (a0 + a1 + a_both) * (b0 + b1 + b_both)
            # Bob odwraca bit: błąd, gdy jego surowy bit równa się bitowi Alice.
            wrong_pairs = (
                a0 * b0 + a1 * b1 + 0.5 * (a_both * (b0 + b1 + b_both) + (a0 + a1) * b_both)
            )
            gain += weight * detected
            errors += weight * wrong_pairs
    return PairStatistics(gain, errors)


def bbm92_rate(stats: PairStatistics, f_ec: float, sift: float = 0.5) -> float:
    """R = q·Q·[1 − h(E) − f h(E)] na impuls pompy (Ma, Fung, Lo 2007, wzór (5))."""
    e = min(stats.qber, 0.5)
    return sift * stats.gain * (1.0 - h2(e) - f_ec * h2(e))


def optimal_bbm92(
    eta_a: float, eta_b: float, dark: float, misalignment: float, f_ec: float, sift: float = 0.5
) -> tuple[float, float]:
    """(R, λ) przy najlepszej jasności źródła (optymalizacja po ln λ)."""
    result = minimize_scalar(
        lambda log_lam: (
            -bbm92_rate(
                pair_statistics(math.exp(log_lam), eta_a, eta_b, dark, misalignment), f_ec, sift
            )
        ),
        bounds=(math.log(1e-6), math.log(1.0)),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return -float(result.fun), math.exp(float(result.x))


def chsh_value(efficiency: float, visibility: float = 1.0) -> float:
    """S przy braku kliknięcia czytanym jako +1: S = 2√2Vη² + 2(1 − η)²."""
    return 2.0 * math.sqrt(2.0) * visibility * efficiency**2 + 2.0 * (1.0 - efficiency) ** 2


def key_setting_qber(efficiency: float, visibility: float = 1.0) -> float:
    """QBER ustawień klucza: oba kliknięcia (1 − V)/2, jedno ½, żadne 0."""
    return efficiency**2 * (1.0 - visibility) / 2.0 + efficiency * (1.0 - efficiency)


def holevo_bound(s: float) -> float:
    """h((1 + √(S²/4 − 1))/2): wiedza Ewy przy ataku kolektywnym (Acín i in. 2007); 1 dla S ≤ 2."""
    if s <= 2.0:
        return 1.0
    return h2((1.0 + math.sqrt(min(s, 2.0 * math.sqrt(2.0)) ** 2 / 4.0 - 1.0)) / 2.0)


def device_independent_rate(qber: float, s: float) -> float:
    """r = 1 − h(Q) − h((1 + √(S²/4 − 1))/2) na rundę klucza."""
    return 1.0 - h2(min(qber, 0.5)) - holevo_bound(s)


def violation_threshold(visibility: float = 1.0) -> float:
    """Najmniejsze η, przy którym S > 2 (brak kliknięcia jako +1)."""
    return float(brentq(lambda eta: chsh_value(eta, visibility) - 2.0, 0.5, 1.0))


def device_independent_threshold(visibility: float = 1.0) -> float:
    """Najmniejsze η z dodatnią stopą device-independent."""

    def rate(eta: float) -> float:
        return device_independent_rate(
            key_setting_qber(eta, visibility), chsh_value(eta, visibility)
        )

    return float(brentq(rate, 0.85, 1.0))
