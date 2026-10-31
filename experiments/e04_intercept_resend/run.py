"""E04: atak intercept-resend: QBER(p), zysk jako drugi detektor ataku, I_AB i I_AE.

Wynik: figures/e04_intercept_resend.png i results/e04_intercept_resend.json.
Kończy się błędem, jeśli któryś punkt odbiega od modelu analitycznego o |z| ≥ z_max.
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qkdsim.analysis.entropy import h2
from qkdsim.analysis.intercept_resend import intercept_resend_model, qber_line
from qkdsim.analysis.key_rate import GYS
from qkdsim.attacks.intercept_resend import InterceptResend
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.metrics import eve_known_fraction, gain, matched_qber
from qkdsim.pipeline import Link, run_counts_parallel
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
P_D = dark_count_per_detector(GYS.background)
LABELS = {0.0: "Ewa tuż za Alice", 1.0: "Ewa tuż przed Bobem"}


@dataclass(frozen=True)
class Config:
    seed: int
    mu: float
    length_km: float
    fractions: tuple[float, ...]
    positions: tuple[float, ...]
    pulses: int
    chunk_size: int
    z_max: float


def measure(cfg: Config, i: int, j: int) -> dict[str, float]:
    fraction, position = cfg.fractions[i], cfg.positions[j]
    link = Link(
        WCPSource((cfg.mu,)),
        FiberChannel(cfg.length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
        eve=InterceptResend(fraction, position),
    )
    counts = run_counts_parallel(
        link, cfg.pulses, np.random.SeedSequence([cfg.seed, i, j]), cfg.chunk_size
    )
    model = intercept_resend_model(GYS, cfg.length_km, cfg.mu, fraction, position, P_D)
    g, e, k = gain(counts), matched_qber(counts), eve_known_fraction(counts)
    return {
        "fraction": fraction,
        "position": position,
        "gain": g.value,
        "gain_sigma": g.sigma,
        "gain_model": model.clicks.detected,
        "gain_z": g.z_score(model.clicks.detected),
        "qber": e.value,
        "qber_sigma": e.sigma,
        "qber_model": model.clicks.qber,
        "qber_z": e.z_score(model.clicks.qber),
        "eve_known": k.value,
        "eve_known_sigma": k.sigma,
        "eve_known_model": model.known_fraction,
    }


def plot(cfg: Config, points: list[dict[str, float]], path: Path) -> None:
    grid = np.linspace(0.0, 1.0, 101)
    fig, (ax_e, ax_q, ax_i) = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    honest = intercept_resend_model(GYS, cfg.length_km, cfg.mu, 0.0, 1.0, P_D).clicks.detected
    for color, position in zip(("C0", "C1"), cfg.positions, strict=True):
        pts = [p for p in points if p["position"] == position]
        x = [p["fraction"] for p in pts]
        models = [
            intercept_resend_model(GYS, cfg.length_km, cfg.mu, f, position, P_D) for f in grid
        ]
        label = LABELS.get(position, f"position = {position}")
        ax_e.plot(grid, [100 * m.clicks.qber for m in models], color=color, label=f"model: {label}")
        ax_e.errorbar(
            x, [100 * p["qber"] for p in pts], yerr=[100 * p["qber_sigma"] for p in pts],
            fmt="o", color=color, markersize=4,
        )  # fmt: skip
        ax_q.plot(grid, [m.clicks.detected / honest for m in models], color=color, label=label)
        ax_q.errorbar(
            x, [p["gain"] / honest for p in pts], yerr=[p["gain_sigma"] / honest for p in pts],
            fmt="o", color=color, markersize=4,
        )  # fmt: skip
    ax_e.plot(
        grid, [100 * qber_line(f, GYS.misalignment) for f in grid],
        "k--", linewidth=1, label="jednofotonowo: e$_d$ + p(¼ − e$_d$/2)",
    )  # fmt: skip
    ax_e.set_xlabel("ułamek przechwyconych impulsów p")
    ax_e.set_ylabel("QBER [%]")
    ax_e.set_title("QBER: fizyka wykrywa podsłuch")
    ax_e.legend(fontsize=8)
    ax_q.set_xlabel("ułamek przechwyconych impulsów p")
    ax_q.set_ylabel("zysk / zysk bez ataku")
    ax_q.set_title("Zysk: drugi detektor ataku")
    ax_q.legend(fontsize=8)

    near_bob = [p for p in points if p["position"] == 1.0]
    models = [intercept_resend_model(GYS, cfg.length_km, cfg.mu, f, 1.0, P_D) for f in grid]
    ax_i.plot(
        grid, [1 - h2(m.clicks.qber) for m in models], color="C2", label="I$_{AB}$ = 1 − h(E)"
    )
    ax_i.plot(
        grid, [m.known_fraction for m in models], color="C3", label="I$_{AE}$ (bity znane Ewie)"
    )
    ax_i.errorbar(
        [p["fraction"] for p in near_bob], [p["eve_known"] for p in near_bob],
        yerr=[p["eve_known_sigma"] for p in near_bob], fmt="o", color="C3", markersize=4,
    )  # fmt: skip
    ax_i.errorbar(
        [p["fraction"] for p in near_bob], [1 - h2(p["qber"]) for p in near_bob],
        fmt="o", color="C2", markersize=4,
    )  # fmt: skip
    ax_i.set_xlabel("ułamek przechwyconych impulsów p")
    ax_i.set_ylabel("informacja [bit na bit po siftingu]")
    ax_i.set_title("Informacja Boba i Ewy (Ewa tuż przed Bobem)")
    ax_i.legend(fontsize=8)
    for ax in (ax_e, ax_q, ax_i):
        ax.grid(True, alpha=0.3)
    pulses_text = f"{cfg.pulses:,}".replace(",", " ")
    fig.suptitle(
        f"Intercept-resend, GYS, L = {cfg.length_km:g} km, μ = {cfg.mu}; "
        f"punkty: Monte Carlo, {pulses_text} impulsów na punkt (±1σ)"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    points = [
        measure(cfg, i, j) for j in range(len(cfg.positions)) for i in range(len(cfg.fractions))
    ]
    print(f"{'poz.':>5} {'p':>5} {'QBER [%]':>9} {'z':>6} {'Q/Q₀':>6} {'z':>6} {'I_AE':>6}")
    honest = intercept_resend_model(GYS, cfg.length_km, cfg.mu, 0.0, 1.0, P_D).clicks.detected
    for p in points:
        print(
            f"{p['position']:>5.1f} {p['fraction']:>5.1f} {100 * p['qber']:>9.3f} "
            f"{p['qber_z']:>6.2f} {p['gain'] / honest:>6.3f} {p['gain_z']:>6.2f} "
            f"{p['eve_known']:>6.3f}"
        )
    plot(cfg, points, ROOT / "figures" / f"{NAME}.png")
    out = ROOT / "results" / f"{NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": asdict(cfg), "points": points}, indent=2), "utf-8")
    worst = max(max(abs(p["gain_z"]), abs(p["qber_z"])) for p in points)
    print(f"max |z| = {worst:.2f}")
    return 0 if worst < cfg.z_max else 1


if __name__ == "__main__":
    sys.exit(main())
