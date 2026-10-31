"""Walidacja ataków 1–3 przy 25 km, parametry GYS.

Wolne testy (`make test-slow`). Atak 1: siatka p × położenie Ewy; atak 2: siatka mocy impulsu
wyzwalającego dla Ewy skalibrowanej i naiwnej; atak 3: estymacja decoy z MC na 0–100 km
i PNS za odległością krytyczną (zawsze przerwanie protokołu).
"""

import math

import numpy as np
import pytest

from conftest import SEED
from qkdsim.analysis.blinding import blinding_model
from qkdsim.analysis.intercept_resend import intercept_resend_model
from qkdsim.analysis.key_rate import GYS, gllp_rate, vacuum_weak_bounds
from qkdsim.analysis.pns import pns_model, pns_strategy
from qkdsim.analysis.wcp import gain as gain_model
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.analysis.wcp import qber as qber_model
from qkdsim.attacks.blinding import DetectorBlinding, matching_send_fraction
from qkdsim.attacks.intercept_resend import InterceptResend
from qkdsim.attacks.pns import PhotonNumberSplitting
from qkdsim.channel import FiberChannel
from qkdsim.defenses.decoy_check import decoy_consistency_z
from qkdsim.detector import CLAVIS2, DetectorPair, dark_count_per_detector
from qkdsim.metrics import eve_known_fraction, gain, matched_qber
from qkdsim.pipeline import Link, run_counts_parallel
from qkdsim.postprocessing.decoy import estimate_decoy
from qkdsim.source import WCPSource

Z_MAX = 4.0
MU, LENGTH = 0.48, 25.0
P_D = dark_count_per_detector(GYS.background)


@pytest.mark.slow
@pytest.mark.parametrize("position", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
def test_intercept_resend_grid(fraction: float, position: float) -> None:
    link = Link(
        WCPSource((MU,)),
        FiberChannel(LENGTH, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
        eve=InterceptResend(fraction, position),
    )
    seed = np.random.SeedSequence([SEED, int(100 * fraction), int(10 * position)])
    counts = run_counts_parallel(link, 20_000_000, seed, chunk_size=5_000_000)
    model = intercept_resend_model(GYS, LENGTH, MU, fraction, position, P_D)
    assert abs(gain(counts).z_score(model.clicks.detected)) < Z_MAX
    assert abs(matched_qber(counts).z_score(model.clicks.qber)) < Z_MAX
    if fraction > 0:
        assert abs(eve_known_fraction(counts).z_score(model.known_fraction)) < Z_MAX


# --- atak 2: oślepianie -----------------------------------------------------------------

BLIND_POWERS = [0.6e-3, 0.85e-3, 1.0e-3, 1.1e-3, 1.25e-3, 1.5e-3, 2.0e-3]


@pytest.mark.slow
@pytest.mark.parametrize("calibrated", [True, False])
@pytest.mark.parametrize("power", BLIND_POWERS)
def test_blinding_grid(power: float, calibrated: bool) -> None:
    detectors = DetectorPair(GYS.eta_bob, P_D, CLAVIS2)
    honest = matched_basis_clicks(MU, GYS.eta(LENGTH), P_D, GYS.misalignment)
    eve = DetectorBlinding(
        power,
        send_fraction=matching_send_fraction(honest.detected, MU),
        mimic_qber=honest.qber,
        compensation_rad=math.asin(math.sqrt(GYS.misalignment)) if calibrated else 0.0,
    )
    link = Link(
        WCPSource((MU,)),
        FiberChannel(LENGTH, GYS.alpha_db_per_km, GYS.misalignment),
        detectors,
        eve=eve,
    )
    seed = np.random.SeedSequence([SEED, int(power * 1e6), int(calibrated)])
    counts = run_counts_parallel(link, 8_000_000, seed, chunk_size=2_000_000)
    model = blinding_model(detectors, eve, MU, GYS.misalignment)
    if model.detected == 0.0:
        # Impuls poniżej P_never obu detektorów: nic nie klika ani w modelu, ani w MC.
        assert counts.detected.sum() == 0
        return
    assert abs(gain(counts).z_score(model.detected)) < Z_MAX
    assert abs(matched_qber(counts).z_score(model.qber)) < Z_MAX


# --- atak 3: PNS i stany decoy -----------------------------------------------------------

DECOY_SOURCE = WCPSource((MU, 0.05, 0.0), (0.7, 0.2, 0.1))
HONEST_PULSES = {0.0: 20_000_000, 25.0: 40_000_000, 50.0: 100_000_000,
                 75.0: 300_000_000, 100.0: 1_000_000_000}  # fmt: skip


def decoy_link(length_km: float, eve: PhotonNumberSplitting | None = None) -> Link:
    return Link(
        DECOY_SOURCE,
        FiberChannel(length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
        eve=eve,
    )


def _vw(
    q_mu: float, e_mu: float, q_nu: float, e_nu: float, y0: float
) -> tuple[float, float, float]:
    """(Y₁^L, e₁^U, R) z pięciu mierzonych wielkości, jak w estimate_decoy."""
    y1, e1 = vacuum_weak_bounds(MU, 0.05, q_mu, e_mu, q_nu, e_nu, y0)
    rate = gllp_rate(q_mu, e_mu, y1 * MU * math.exp(-MU), e1, GYS.f_ec, GYS.q_sift)
    return y1, e1, rate


def _propagated_sigma(values: list[float], sigmas: list[float]) -> tuple[float, float, float]:
    """σ(Y₁^L), σ(e₁^U), σ(R) z propagacji niepewności (pochodne numeryczne)."""
    base = _vw(*values)
    variance = [0.0, 0.0, 0.0]
    for i, sigma in enumerate(sigmas):
        step = sigma * 1e-3
        shifted = list(values)
        shifted[i] += step
        moved = _vw(*shifted)
        for j in range(3):
            variance[j] += ((moved[j] - base[j]) / step * sigma) ** 2
    return math.sqrt(variance[0]), math.sqrt(variance[1]), math.sqrt(variance[2])


@pytest.mark.slow
@pytest.mark.parametrize("length_km", sorted(HONEST_PULSES))
def test_decoy_estimate_from_monte_carlo(length_km: float) -> None:
    """Y₁^L, e₁^U i R z danych MC zgodne z analityką w granicach 4σ propagowanej niepewności.

    Słaby decoy (ν = 0,05) daje mało detekcji, więc e₁^U i R są dużo bardziej zaszumione niż
    Q_μ; dlatego tolerancje wynikają z propagacji niepewności, a nie są stałe.
    """
    seed = np.random.SeedSequence([SEED, 3, int(length_km)])
    counts = run_counts_parallel(decoy_link(length_km), HONEST_PULSES[length_km], seed)
    estimate = estimate_decoy(counts, MU, 0.05, GYS.f_ec)
    eta = GYS.eta(length_km)
    expected = _vw(
        gain_model(MU, eta, GYS.background),
        qber_model(MU, eta, GYS.background, GYS.misalignment),
        gain_model(0.05, eta, GYS.background),
        qber_model(0.05, eta, GYS.background, GYS.misalignment),
        GYS.background,
    )
    measured = [gain(counts, 0), matched_qber(counts, 0), gain(counts, 1), matched_qber(counts, 1)]
    vacuum = gain(counts, 2)
    sigmas = _propagated_sigma(
        [m.value for m in measured] + [vacuum.value], [m.sigma for m in measured] + [vacuum.sigma]
    )
    observed = (estimate.y1_lower, estimate.e1_upper, estimate.rate)
    for got, want, sigma in zip(observed, expected, sigmas, strict=True):
        assert abs(got - want) < Z_MAX * sigma
    assert abs(decoy_consistency_z(counts, MU, 0.05)) < Z_MAX


@pytest.mark.slow
@pytest.mark.parametrize("length_km", [40.0, 60.0, 80.0, 100.0])
def test_pns_beyond_critical_distance_is_caught(length_km: float) -> None:
    honest = matched_basis_clicks(MU, GYS.eta(length_km), P_D, GYS.misalignment)
    block, forward = pns_strategy(honest.detected, MU, GYS.eta_bob, P_D)
    seed = np.random.SeedSequence([SEED, 33, int(length_km)])
    eve = PhotonNumberSplitting(block, forward)
    counts = run_counts_parallel(decoy_link(length_km, eve), 300_000_000, seed)
    model = pns_model(MU, block, forward, GYS.eta_bob, P_D, GYS.misalignment)
    assert abs(gain(counts).z_score(model.clicks.detected)) < Z_MAX
    assert abs(gain(counts).z_score(honest.detected)) < Z_MAX  # zysk sygnału jak bez ataku
    assert eve_known_fraction(counts).value > 0.95
    estimate = estimate_decoy(counts, MU, 0.05, GYS.f_ec)
    assert estimate.aborted and estimate.naive_rate > 0.0
    assert decoy_consistency_z(counts, MU, 0.05) < -Z_MAX
