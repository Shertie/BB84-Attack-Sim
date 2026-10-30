"""Estymacja stanów decoy z liczników Monte Carlo (F3.13–F3.14).

Alice po pomiarze ujawnia, która intensywność była w którym impulsie. Z liczników Boba
powstają Q_μ, E_μ (sygnał), Q_ν, E_ν (słaby decoy) i Y₀, e₀ (próżnia), a z nich granice
vacuum+weak Ma et al. 2005: Y₁^L, e₁^U i stopa GLLP na impuls sygnałowy.

Wersja asymptotyczna: fluktuacje statystyczne estymat nie są tu uwzględnione (F7, Lim et al.
2014). `naive_rate` liczy klucz tak, jakby każda detekcja pochodziła z jednego fotonu; to
właśnie błąd, który atak PNS wykorzystuje.
"""

import math
from dataclasses import dataclass

from qkdsim.analysis.entropy import h2
from qkdsim.analysis.key_rate import gllp_rate, shor_preskill_rate, vacuum_weak_bounds
from qkdsim.metrics import gain, matched_qber
from qkdsim.pipeline import Counts

SIGNAL, DECOY, VACUUM = 0, 1, 2


@dataclass(frozen=True)
class DecoyEstimate:
    q_mu: float
    e_mu: float
    q_nu: float
    e_nu: float
    y0: float
    e0: float
    y1_lower: float
    e1_upper: float
    rate: float  # GLLP z granicami decoy, na impuls sygnałowy; ≤ 0 oznacza przerwanie protokołu
    naive_rate: float  # gdyby wszystkie detekcje były jednofotonowe (niebezpieczne)

    @property
    def aborted(self) -> bool:
        return self.rate <= 0.0


def estimate_decoy(
    counts: Counts, mu: float, nu: float, f_ec: float, q_sift: float = 0.5
) -> DecoyEstimate:
    """Intensywności w kolejności (μ, ν, 0), jak w WCPSource((μ, ν, 0.0), ...)."""
    if counts.sent.shape[0] != 3 or not 0.0 < nu < mu:
        raise ValueError("wymagane trzy intensywności (μ, ν, 0) z 0 < ν < μ")
    q_mu, e_mu = gain(counts, SIGNAL).value, matched_qber(counts, SIGNAL).value
    q_nu, e_nu = gain(counts, DECOY).value, matched_qber(counts, DECOY).value
    vacuum = gain(counts, VACUUM)
    y0 = vacuum.value
    e0 = matched_qber(counts, VACUUM).value if vacuum.successes > 0 else 0.5
    y1, e1 = vacuum_weak_bounds(mu, nu, q_mu, e_mu, q_nu, e_nu, y0, e0)
    q1 = max(y1, 0.0) * mu * math.exp(-mu)
    e1_clipped = min(max(e1, 0.0), 0.5) if y1 > 0.0 else 0.5
    rate = gllp_rate(q_mu, e_mu, q1, e1_clipped, f_ec, q_sift)
    naive = q_sift * q_mu * shor_preskill_rate(e_mu, f_ec) if e_mu < 0.5 else -h2(0.5)
    return DecoyEstimate(q_mu, e_mu, q_nu, e_nu, y0, e0, y1, e1, rate, naive)
