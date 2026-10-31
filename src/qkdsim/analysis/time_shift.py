"""Model analityczny ataku time-shift (Qi et al. 2007, Zhao et al. 2008).

Detektory Boba mają różne krzywe sprawności w czasie (`detector.EfficiencyMismatch`). Ewa nie
mierzy impulsów, tylko przesuwa każdy w czasie: z p-stwem p_A do chwili A, w której przeważa
detektor 0, inaczej do chwili B, w której przeważa detektor 1. Po siftingu zgaduje bit Boba
z własnego przesunięcia. Błąd optyki daje przy każdym przesunięciu ten sam QBER e_d (w granicy
małego ημ błędne i poprawne detekcje skalują się tą samą sprawnością), więc atak widać tylko
w zysku, który Ewa wyrównuje, zastępując światłowód linią bezstratną.

Dane Zhao et al. 2008 (Tabela I, system ID-500, μ = 0,1):
- (a) liczby detekcji d₀, d₁ przy przesunięciach A = −250 ps i B = +500 ps;
- (b) detekcje w zgodnej bazie według bazy, bitu Alice i bitu Boba;
- (c) p_A = 23,0%, E = 5,68%, K_U = 1131 bitów < K_L = 1297 bitów.

Krzywe Gaussa dopasowujemy do (a): rozsunięcie Δ = 100 ps z tekstu artykułu, równe szczyty
sprawności (założenie; artykuł mówi o „nieco różnych”), a środki i szerokości wynikają wprost
z czterech liczb detekcji po odjęciu ciemnych zliczeń.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from qkdsim.analysis.entropy import h2
from qkdsim.analysis.key_rate import infinite_decoy_singles
from qkdsim.analysis.wcp import ClickProbabilities
from qkdsim.detector import EfficiencyMismatch, dark_count_per_detector

Floats = npt.NDArray[np.float64]

ZHAO_SHIFTS_NS = (-0.250, 0.500)  # przesunięcia A i B w osi czasu artykułu
ZHAO_DETECTIONS = ((10992, 1541), (1231, 4059))  # (d₀, d₁) przy A i przy B, Tabela I(a)
ZHAO_PULSES = 20_966_400  # N przy każdym przesunięciu
ZHAO_MATCHED = 10_481_280  # Ñ: impulsy w zgodnej bazie
ZHAO_MU = 0.1
ZHAO_BACKGROUND = 2.26e-5  # Y₀
ZHAO_F_EC = 1.22
ZHAO_P_A = 0.230
ZHAO_DELTA_NS = 0.100  # największe rozsunięcie aktywacji detektorów (tekst artykułu)
# Tabela I(b): [przesunięcie A/B][baza Z₂][bit Alice X][bit Boba Y].
ZHAO_TABLE = np.array(
    [
        [[[2557, 65], [139, 336]], [[2634, 59], [120, 333]]],
        [[[260, 41], [31, 979]], [[279, 35], [37, 1022]]],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ZhaoFit:
    """Krzywe i przesunięcia Ewy w czasie względem nominalnego przyjścia (przecięcia krzywych)."""

    mismatch: EfficiencyMismatch
    shifts_ns: tuple[float, float]  # A i B
    crossing_ns: float  # nominalne przyjście w osi czasu artykułu


def fit_zhao(delta_ns: float = ZHAO_DELTA_NS) -> ZhaoFit:
    """Gaussy o równych szczytach przez cztery liczby z Tabeli I(a), c₁ − c₀ = `delta_ns`."""
    dark = dark_count_per_detector(ZHAO_BACKGROUND) * ZHAO_PULSES
    (a0, a1), (b0, b1) = np.log(np.asarray(ZHAO_DETECTIONS, dtype=np.float64) - dark)
    ta, tb = ZHAO_SHIFTS_NS
    log_ratio0, log_ratio1 = a0 - b0, b1 - a1  # ln s₀(A)/s₀(B), ln s₁(B)/s₁(A)

    def variances(c0: float) -> tuple[float, float]:
        c1 = c0 + delta_ns
        var0 = ((tb - c0) ** 2 - (ta - c0) ** 2) / (2.0 * log_ratio0)
        var1 = ((ta - c1) ** 2 - (tb - c1) ** 2) / (2.0 * log_ratio1)
        return var0, var1

    def peak_gap(c0: float) -> float:
        var0, var1 = variances(c0)
        peak0 = a0 + (ta - c0) ** 2 / (2.0 * var0)
        peak1 = b1 + (tb - c0 - delta_ns) ** 2 / (2.0 * var1)
        return float(peak0 - peak1)

    # Obie wariancje są dodatnie tylko dla c₀ < (A + B)/2 < c₁.
    mid, eps = (ta + tb) / 2.0, 1e-9
    c0 = float(brentq(peak_gap, mid - delta_ns + eps, mid - eps))
    c1 = c0 + delta_ns
    var0, var1 = variances(c0)

    def log_gap(t: float) -> float:
        return float((t - c1) ** 2 / (2.0 * var1) - (t - c0) ** 2 / (2.0 * var0))

    crossing = float(brentq(log_gap, c0, c1))
    return ZhaoFit(
        EfficiencyMismatch(
            centers_ns=(c0 - crossing, c1 - crossing),
            widths_ns=(math.sqrt(var0), math.sqrt(var1)),
        ),
        (ta - crossing, tb - crossing),
        crossing,
    )


def symmetric_mismatch(delta_ns: float, width_ns: float) -> EfficiencyMismatch:
    """Rodzina do przeglądu po Δ: równe szerokości, środki w ∓Δ/2; Δ = 0 to brak niedopasowania."""
    return EfficiencyMismatch((-delta_ns / 2.0, delta_ns / 2.0), (width_ns, width_ns))


def _clicks(mean_right: float, mean_wrong: float, dark_count: float) -> ClickProbabilities:
    silent = 1.0 - dark_count
    right = dark_count + silent * -math.expm1(-mean_right)
    wrong = dark_count + silent * -math.expm1(-mean_wrong)
    return ClickProbabilities(
        right_only=right * (1.0 - wrong),
        wrong_only=wrong * (1.0 - right),
        double=right * wrong,
        none=(1.0 - right) * (1.0 - wrong),
    )


def bit_probabilities(
    mu: float,
    transmittance: float,
    efficiencies: tuple[float, float],
    dark_count: float,
    misalignment: float,
    swap: bool = False,
) -> Floats:
    """p[x, y]: p-stwo detekcji z bitem Boba y przy bicie Alice x, na impuls w zgodnej bazie.

    Dokładna wartość oczekiwana dla detector.py (Poisson, Born, niezależne ciemne zliczenia,
    podwójne kliknięcie = losowy bit). `swap`: Bob losowo zamienia przypisanie bit ↔ detektor.
    """
    probs = np.zeros((2, 2))
    for assignment in (0, 1) if swap else (0,):
        for x in (0, 1):
            right = x ^ assignment  # detektor, do którego trafia foton bez błędu optyki
            clicks = _clicks(
                mu * transmittance * efficiencies[right] * (1.0 - misalignment),
                mu * transmittance * efficiencies[1 - right] * misalignment,
                dark_count,
            )
            probs[x, x] += clicks.right_only + 0.5 * clicks.double
            probs[x, 1 - x] += clicks.wrong_only + 0.5 * clicks.double
    return probs / (2.0 if swap else 1.0)


def balancing_probability(at_a: Floats, at_b: Floats) -> float:
    """p_A, przy którym Bob ma tyle samo zer co jedynek; `at_*` = detekcje (d₀, d₁).

    Bez niedopasowania (nic do zrównoważenia) zwraca ½. Gdy oba przesunięcia faworyzują
    ten sam detektor, równowagi nie ma i wynik jest obcięty do [0, 1].
    """
    gap_a, gap_b = float(at_a[0] - at_a[1]), float(at_b[1] - at_b[0])
    if gap_a + gap_b <= 0.0:
        return 0.5
    return min(max(gap_b / (gap_a + gap_b), 0.0), 1.0)


@dataclass(frozen=True)
class TimeShiftModel:
    """Oczekiwane detekcje w zgodnej bazie na impuls: table[przesunięcie, bit Alice, bit Boba]."""

    table: Floats
    p_a: float

    @property
    def gain(self) -> float:
        return float(self.table.sum())

    @property
    def qber(self) -> float:
        return float(self.table[:, 0, 1].sum() + self.table[:, 1, 0].sum()) / self.gain

    @property
    def eve_information(self) -> float:
        return eve_information(self.table)


def time_shift_model(
    mu: float,
    transmittance: float,
    eta_bob: float,
    dark_count: float,
    misalignment: float,
    mismatch: EfficiencyMismatch,
    shifts_ns: tuple[float, float],
    p_a: float | None = None,
    swap: bool = False,
) -> TimeShiftModel:
    """Wartość oczekiwana symulatora; `p_a` = None: Ewa równoważy zera i jedynki Boba."""
    per_shift = []
    for shift in shifts_ns:
        t = np.array([shift])
        etas = (
            min(eta_bob * float(mismatch.relative(t, 0)[0]), 1.0),
            min(eta_bob * float(mismatch.relative(t, 1)[0]), 1.0),
        )
        per_shift.append(
            0.5 * bit_probabilities(mu, transmittance, etas, dark_count, misalignment, swap)
        )
    if p_a is None:
        p_a = balancing_probability(per_shift[0].sum(axis=0), per_shift[1].sum(axis=0))
    weights = np.array([p_a, 1.0 - p_a])[:, None, None]
    return TimeShiftModel(weights * np.stack(per_shift), p_a)


def eve_information(table: npt.ArrayLike) -> float:
    """1 − Σ_k P(k) h(P(X = 0 | k)) na wykryty bit w zgodnej bazie.

    `table[..., x, y]`: detekcje (liczby albo p-stwa) z bitem Alice x i bitem Boba y; osie
    wiodące to to, co Ewa wie o impulsie (jej przesunięcie, po siftingu także baza). Tak liczy
    górną granicę K_U Zhao et al. 2008 (wzór (3)).
    """
    per_x = np.asarray(table, dtype=np.float64).sum(axis=-1).reshape(-1, 2)
    total = per_x.sum()
    information = 1.0
    for row in per_x:
        detected = row.sum()
        if detected > 0.0:
            information -= detected / total * h2(row[0] / detected)
    return float(information)


@dataclass(frozen=True)
class KeyBounds:
    """Granice długości klucza z Zhao et al. 2008 (wzory (1)–(3))."""

    upper: float  # K_U: Ewa zna niedopasowanie i swoje przesunięcia
    lower: float  # K_L: Alice i Bob ataku nie widzą (nieskończenie wiele decoy, GLLP)
    qber: float
    detected: float

    @property
    def attack_succeeds(self) -> bool:
        """K_L > K_U: Alice i Bob zostawiają więcej klucza, niż jest tajne wobec Ewy."""
        return self.lower > self.upper


def zhao_key_bounds(
    table: npt.ArrayLike, matched: float, mu: float, background: float, f_ec: float
) -> KeyBounds:
    """K_U i K_L z tabeli detekcji w zgodnej bazie (`table[..., x, y]`) i liczby Ñ impulsów.

    K_L: η z Q = Y₀ + 1 − e^(−ημ), e_d z EQ = Y₀/2 + e_d(1 − e^(−ημ)), potem Y₁ i e₁ jak dla
    nieskończenie wielu stanów decoy. Dla p-stw zamiast liczb: `matched` = 1 daje K na impuls.
    """
    counts = np.asarray(table, dtype=np.float64)
    detected = float(counts.sum())
    errors = float(counts[..., 0, 1].sum() + counts[..., 1, 0].sum())
    qber = errors / detected
    error_correction = detected * f_ec * h2(qber)
    upper = detected * (1.0 - eve_information(counts)) - error_correction
    gain = detected / matched
    signal = gain - background  # 1 − e^(−ημ)
    eta = -math.log1p(-signal) / mu
    misalignment = (qber * gain - 0.5 * background) / signal
    y1, e1 = infinite_decoy_singles(eta, background, misalignment)
    singles = y1 * mu * math.exp(-mu) * (1.0 - h2(min(e1, 0.5)))
    lower = matched * (singles + background * math.exp(-mu)) - error_correction
    return KeyBounds(upper, lower, qber, detected)


def gain_matching_transmission(
    mu: float,
    transmittance: float,
    eta_bob: float,
    dark_count: float,
    misalignment: float,
    mismatch: EfficiencyMismatch,
    shifts_ns: tuple[float, float],
) -> float | None:
    """Tłumienie Ewy za linią bezstratną, przy którym zysk Boba jest jak bez ataku.

    `transmittance`: transmisja uczciwego kanału. None, gdy nawet bez tłumienia zysk pod atakiem
    jest niższy, czyli Ewa nie ukryje spadku sprawności na tej odległości.
    """
    honest = time_shift_model(
        mu, transmittance, eta_bob, dark_count, misalignment, mismatch, (0.0, 0.0), 1.0
    ).gain

    def gap(transmission: float) -> float:
        attacked = time_shift_model(
            mu, transmission, eta_bob, dark_count, misalignment, mismatch, shifts_ns
        )
        return attacked.gain - honest

    if gap(1.0) < 0.0:
        return None
    return float(brentq(gap, 0.0, 1.0))
