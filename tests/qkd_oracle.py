"""Jedyny punkt dostępu do pakietu qkd w testach (docs/architecture.md).

Przekłada prywatne funkcje qkd._core na nazwy i konwencje projektu. Domyślne parametry to zestaw
GYS z golden.py. Zmiana API qkd ma dotykać tylko tego pliku.
"""

import math
from collections.abc import Callable

import numpy as np
from qkd import _core, attacks
from scipy.optimize import minimize_scalar

import golden as g


def binary_entropy(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def transmittance(length_km: float) -> float:
    return float(_core.decoy_eta(g.GYS_ALPHA_DB_PER_KM, length_km, g.GYS_ETA_BOB))


def gain_and_qber(mu: float, length_km: float) -> tuple[float, float]:
    """Q_mu i E_mu (Ma et al. 2005, wzór (7) z korektą podwójnego liczenia)."""
    q, e = _core.decoy_gain(mu, transmittance(length_km), g.GYS_Y0_PER_PULSE, g.GYS_E_DETECTOR)
    return float(q), float(e)


def single_photon_ideal(length_km: float) -> tuple[float, float]:
    """Y1 i e1 w granicy nieskończenie wielu stanów decoy."""
    y1, e1 = _core.decoy_ideal(transmittance(length_km), g.GYS_Y0_PER_PULSE, g.GYS_E_DETECTOR)
    return float(y1), float(e1)


def vacuum_weak_bounds(length_km: float, mu: float, nu: float) -> tuple[float, float, float]:
    """Y1^L, e1^U i Q1^L z protokołu vacuum+weak (drugi stan decoy to próżnia)."""
    q_mu, e_mu = gain_and_qber(mu, length_km)
    q_nu, e_nu = gain_and_qber(nu, length_km)
    y1, e1, q1 = _core.decoy_bounds(mu, nu, 0.0, q_mu, e_mu, q_nu, e_nu, g.GYS_Y0_PER_PULSE, g.E0)
    return float(y1), float(e1), float(q1)


def _gllp_rate(q_mu: float, e_mu: float, q1: float, e1: float) -> float:
    return g.Q_SIFT * (-q_mu * g.GYS_F_EC * binary_entropy(e_mu) + q1 * (1.0 - binary_entropy(e1)))


def rate_infinite_decoy(length_km: float, mu: float = g.VACUUM_WEAK_MU) -> float:
    q_mu, e_mu = gain_and_qber(mu, length_km)
    y1, e1 = single_photon_ideal(length_km)
    return _gllp_rate(q_mu, e_mu, y1 * mu * math.exp(-mu), e1)


def rate_vacuum_weak(
    length_km: float, mu: float = g.VACUUM_WEAK_MU, nu: float = g.VACUUM_WEAK_NU
) -> float:
    q_mu, e_mu = gain_and_qber(mu, length_km)
    _y1, e1, q1 = vacuum_weak_bounds(length_km, mu, nu)
    return _gllp_rate(q_mu, e_mu, q1, min(e1, 0.5))


def rate_gllp_no_decoy(length_km: float, mu: float) -> float:
    """GLLP bez decoy: wszystkie impulsy wielofotonowe zna Ewa (najgorszy przypadek)."""
    q_mu, e_mu = gain_and_qber(mu, length_km)
    multi = (1.0 - math.exp(-mu) * (1.0 + mu)) / q_mu
    if multi >= 1.0:
        return -1.0
    single = 1.0 - multi
    return _gllp_rate(q_mu, e_mu, q_mu * single, min(e_mu / single, 0.5))


_MU_GRID = np.geomspace(1e-4, 1.0, 400)


def best_rate_gllp_no_decoy(length_km: float) -> float:
    """Stopa GLLP bez decoy przy optymalnym mu.

    Dodatnia stopa istnieje tylko w wąskim zakresie małych mu, a poza nim funkcja jest płaska,
    więc samo minimize_scalar gubi optimum. Najpierw siatka logarytmiczna, potem dokładna
    optymalizacja między sąsiadami najlepszego punktu.
    """
    rates = [rate_gllp_no_decoy(length_km, float(mu)) for mu in _MU_GRID]
    i = int(np.argmax(rates))
    lo, hi = _MU_GRID[max(i - 1, 0)], _MU_GRID[min(i + 1, len(_MU_GRID) - 1)]
    res = minimize_scalar(
        lambda mu: -rate_gllp_no_decoy(length_km, mu),
        bounds=(float(lo), float(hi)),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return max(float(-res.fun), rates[i])


def max_distance(rate: Callable[[float], float], hi_km: float = 400.0) -> float:
    """Odległość, przy której stopa klucza spada do zera (bisekcja)."""
    lo = 0.0
    if rate(lo) <= 0.0 or rate(hi_km) > 0.0:
        raise ValueError("stopa musi być dodatnia w 0 km i niedodatnia na końcu przedziału")
    for _ in range(60):
        mid = 0.5 * (lo + hi_km)
        if rate(mid) > 0.0:
            lo = mid
        else:
            hi_km = mid
    return lo


def blinding_ratio(always_w: tuple[float, ...], never_w: tuple[float, ...]) -> float:
    """max P_always / min P_never; oślepianie jest doskonałe poniżej 2 (Lydersen, wzór (1))."""
    return float(attacks.blind_ratio(always_w, never_w))
