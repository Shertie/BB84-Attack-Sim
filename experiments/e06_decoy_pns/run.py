"""E06: atak PNS i stany decoy.

Wynik: figures/e06_decoy_pns.png i results/e06_decoy_pns.json.
- R(L): GLLP bez decoy, decoy vacuum+weak (wzór i estymata z MC), granica decoy;
- PNS za odległością krytyczną: naiwny rachunek daje klucz, a decoy przerywa protokół;
- wiedza Ewy pod PNS i ROC testu spójności zysków sygnału i decoy.
Kończy się błędem, gdy PNS za odległością krytyczną nie prowadzi do przerwania protokołu.
"""

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qkdsim.analysis.key_rate import (
    GYS,
    best_over_mu,
    max_distance,
    rate_gllp_no_decoy,
    rate_infinite_decoy,
    rate_vacuum_weak,
)
from qkdsim.analysis.pns import critical_transmittance, pns_model, pns_strategy
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.attacks.pns import PhotonNumberSplitting
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.defenses.decoy_check import decoy_consistency_z
from qkdsim.defenses.roc import auc
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.metrics import eve_known_fraction
from qkdsim.pipeline import Link, run_counts, run_counts_parallel
from qkdsim.postprocessing.decoy import estimate_decoy
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
P_D = dark_count_per_detector(GYS.background)


@dataclass(frozen=True)
class Config:
    seed: int
    mu: float
    nu: float
    probabilities: tuple[float, float, float]
    honest_lengths_km: tuple[float, ...]
    honest_pulses: tuple[int, ...]
    pns_lengths_km: tuple[float, ...]
    pns_pulses: int
    roc_length_km: float
    roc_session_pulses: int
    roc_sessions: int


def strategy(cfg: Config, length_km: float) -> tuple[float, float]:
    honest = matched_basis_clicks(cfg.mu, GYS.eta(length_km), P_D, GYS.misalignment)
    return pns_strategy(honest.detected, cfg.mu, GYS.eta_bob, P_D)


def link(cfg: Config, length_km: float, attacked: bool) -> Link:
    eve = PhotonNumberSplitting(*strategy(cfg, length_km)) if attacked else None
    return Link(
        WCPSource((cfg.mu, cfg.nu, 0.0), cfg.probabilities),
        FiberChannel(length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
        eve=eve,
    )


def measure(cfg: Config, length_km: float, pulses: int, attacked: bool) -> dict[str, float]:
    seed = np.random.SeedSequence([cfg.seed, int(length_km), int(attacked)])
    counts = run_counts_parallel(link(cfg, length_km, attacked), pulses, seed)
    estimate = estimate_decoy(counts, cfg.mu, cfg.nu, GYS.f_ec)
    return {
        "length_km": length_km,
        "attacked": attacked,
        "rate": estimate.rate,
        "naive_rate": estimate.naive_rate,
        "y1_lower": estimate.y1_lower,
        "e1_upper": estimate.e1_upper,
        "qber": estimate.e_mu,
        "eve_known": eve_known_fraction(counts).value,
        "consistency_z": decoy_consistency_z(counts, cfg.mu, cfg.nu),
    }


def consistency_auc(cfg: Config) -> float:
    scores: dict[bool, list[float]] = {False: [], True: []}
    for attacked in (False, True):
        for s in range(cfg.roc_sessions):
            streams = RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, 99, s, attacked]))
            counts = run_counts(
                link(cfg, cfg.roc_length_km, attacked), cfg.roc_session_pulses, streams
            )
            scores[attacked].append(-decoy_consistency_z(counts, cfg.mu, cfg.nu))
    return auc(np.array(scores[False]), np.array(scores[True]))


def plot(
    cfg: Config,
    honest: list[dict[str, float]],
    pns: list[dict[str, float]],
    roc_auc: float,
    path: Path,
) -> None:
    grid = np.linspace(0.0, 150.0, 301)
    l_crit = -10 * math.log10(critical_transmittance(cfg.mu, GYS.eta_bob)) / GYS.alpha_db_per_km
    fig, (ax_r, ax_k) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    def positive(values: list[float]) -> list[float]:
        return [v if v > 0 else np.nan for v in values]

    ax_r.semilogy(
        grid,
        positive([rate_infinite_decoy(GYS, x, cfg.mu) for x in grid]),
        "k:",
        label="decoy, granica (nieskończenie wiele stanów)",
    )
    ax_r.semilogy(
        grid,
        positive([rate_vacuum_weak(GYS, x, cfg.mu, cfg.nu) for x in grid]),
        color="C0",
        label=f"decoy vacuum+weak (ν = {cfg.nu})",
    )
    ax_r.semilogy(
        grid,
        positive([best_over_mu(lambda m, x=x: rate_gllp_no_decoy(GYS, x, m))[0] for x in grid]),
        color="C1",
        label="GLLP bez decoy (μ optymalne)",
    )
    ax_r.semilogy(
        [p["length_km"] for p in honest],
        positive([p["rate"] for p in honest]),
        "o",
        color="C0",
        label="decoy z Monte Carlo, bez ataku",
    )
    ax_r.semilogy(
        [p["length_km"] for p in pns],
        positive([p["naive_rate"] for p in pns]),
        "x",
        color="C3",
        markersize=8,
        label="PNS: rachunek naiwny (Ewa zna klucz)",
    )
    floor = 1e-7
    ax_r.plot(
        [p["length_km"] for p in pns],
        [floor] * len(pns),
        "v",
        color="C3",
        markersize=8,
        label="PNS: decoy przerywa protokół (R ≤ 0)",
    )
    ax_r.axvline(l_crit, color="0.5", linestyle="--", linewidth=1)
    ax_r.text(l_crit + 1, 3e-3, f"L$_{{kryt}}$ = {l_crit:.1f} km", fontsize=8)
    ax_r.set(
        xlabel="długość światłowodu L [km]",
        ylabel="stopa klucza R [bit na impuls sygnałowy]",
        title="Stopa klucza i atak PNS",
        ylim=(floor / 2, 2e-2),
    )
    ax_r.legend(fontsize=7, loc="upper right")

    model_known = []
    for x in grid:
        block, forward = strategy(cfg, x)
        model_known.append(
            pns_model(cfg.mu, block, forward, GYS.eta_bob, P_D, GYS.misalignment).known_fraction
        )
    ax_k.plot(grid, model_known, color="C3", label="model: bity znane Ewie (sygnał)")
    ax_k.plot(
        [p["length_km"] for p in pns],
        [p["eve_known"] for p in pns],
        "o",
        color="C3",
        label="Monte Carlo",
    )
    ax_k.axvline(l_crit, color="0.5", linestyle="--", linewidth=1)
    ax_k.set(
        xlabel="długość światłowodu L [km]",
        ylabel="bity znane Ewie / bity po siftingu",
        title=(
            f"Wiedza Ewy pod PNS; test spójności decoy przy {cfg.roc_length_km:g} km: "
            f"AUC = {roc_auc:.3f}"
        ),
    )
    ax_k.legend(fontsize=8, loc="lower right")
    for ax in (ax_r, ax_k):
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"PNS i stany decoy, GYS, μ = {cfg.mu}, ν = {cfg.nu}, f = {GYS.f_ec}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    honest = [
        measure(cfg, length, pulses, attacked=False)
        for length, pulses in zip(cfg.honest_lengths_km, cfg.honest_pulses, strict=True)
    ]
    pns = [measure(cfg, length, cfg.pns_pulses, attacked=True) for length in cfg.pns_lengths_km]
    roc_auc = consistency_auc(cfg)
    l_vw = max_distance(lambda x: rate_vacuum_weak(GYS, x, cfg.mu, cfg.nu))
    header = f"{'L [km]':>7} {'atak':>5} {'R decoy':>10} {'R naiwne':>10} {'QBER':>7}"
    print(f"{header} {'wiedza Ewy':>10} {'z spójn.':>9}")
    for p in honest + pns:
        print(
            f"{p['length_km']:>7.0f} {'PNS' if p['attacked'] else '—':>5} {p['rate']:>10.2e} "
            f"{p['naive_rate']:>10.2e} {100 * p['qber']:>6.2f}% {p['eve_known']:>10.3f} "
            f"{p['consistency_z']:>9.1f}"
        )
    print(f"AUC testu spójności przy {cfg.roc_length_km:g} km: {roc_auc:.3f}")
    print(f"L_max vacuum+weak = {l_vw:.2f} km")
    plot(cfg, honest, pns, roc_auc, ROOT / "figures" / f"{NAME}.png")
    out = ROOT / "results" / f"{NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"config": asdict(cfg), "honest": honest, "pns": pns, "consistency_auc": roc_auc},
            indent=2,
        ),
        "utf-8",
    )
    ok = all(p["rate"] <= 0.0 and p["naive_rate"] > 0.0 for p in pns)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
