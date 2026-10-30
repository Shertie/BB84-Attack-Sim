"""E01: zysk Q_μ(L) i QBER E_μ(L) z Monte Carlo na tle wzorów Ma et al. 2005 (F1.12).

Wynik: figures/e01_gain_qber_vs_L.png i results/e01_gain_qber_vs_L.json.
Kończy się błędem, jeśli którykolwiek punkt odbiega od modelu o |z| ≥ z_max (bramka G1).
"""

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qkdsim.analysis.key_rate import LinkParameters
from qkdsim.analysis.wcp import gain, matched_basis_clicks, qber
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.metrics import gain as measured_gain
from qkdsim.metrics import matched_qber
from qkdsim.pipeline import Link, run_counts_parallel
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name


@dataclass(frozen=True)
class Config:
    seed: int
    mu: float
    link: LinkParameters
    lengths_km: tuple[float, ...]
    pulses: tuple[int, ...]
    chunk_size: int
    z_max: float

    def __post_init__(self) -> None:
        if len(self.lengths_km) != len(self.pulses):
            raise ValueError("lengths_km i pulses muszą mieć tę samą długość")


def simulate_point(cfg: Config, index: int) -> dict[str, float]:
    length, n_pulses = cfg.lengths_km[index], cfg.pulses[index]
    p = cfg.link
    p_d = dark_count_per_detector(p.background)
    link = Link(
        WCPSource((cfg.mu,)),
        FiberChannel(length, p.alpha_db_per_km, p.misalignment),
        DetectorPair(p.eta_bob, p_d),
    )
    counts = run_counts_parallel(
        link, n_pulses, np.random.SeedSequence([cfg.seed, index]), cfg.chunk_size
    )
    eta = p.eta(length)
    model = matched_basis_clicks(cfg.mu, eta, p_d, p.misalignment)
    q_est, e_est = measured_gain(counts), matched_qber(counts)
    return {
        "length_km": length,
        "pulses": n_pulses,
        "detections": q_est.successes,
        "gain": q_est.value,
        "gain_sigma": q_est.sigma,
        "gain_model": model.detected,
        "gain_paper": gain(cfg.mu, eta, p.background),
        "gain_z": q_est.z_score(model.detected),
        "qber": e_est.value,
        "qber_sigma": e_est.sigma,
        "qber_model": model.qber,
        "qber_paper": qber(cfg.mu, eta, p.background, p.misalignment, p.e0),
        "qber_z": e_est.z_score(model.qber),
    }


def plot(cfg: Config, points: list[dict[str, float]], path: Path) -> None:
    p = cfg.link
    grid = np.linspace(0.0, max(cfg.lengths_km) + 10.0, 400)
    eta = [p.eta(x) for x in grid]
    curve_q = [gain(cfg.mu, e, p.background) for e in eta]
    curve_e = [100 * qber(cfg.mu, e, p.background, p.misalignment, p.e0) for e in eta]
    x = [pt["length_km"] for pt in points]

    fig, (ax_q, ax_e) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax_q.semilogy(grid, curve_q, color="0.25", label="wzór (Ma et al. 2005)")
    ax_q.errorbar(
        x, [pt["gain"] for pt in points], yerr=[pt["gain_sigma"] for pt in points],
        fmt="o", color="C0", label="Monte Carlo (±1σ)",
    )  # fmt: skip
    ax_q.set_xlabel("długość światłowodu L [km]")
    ax_q.set_ylabel("zysk $Q_\\mu$ [na impuls]")
    ax_q.set_title("Zysk")
    ax_q.grid(True, which="both", alpha=0.3)
    ax_q.legend()

    ax_e.plot(grid, curve_e, color="0.25", label="wzór (Ma et al. 2005)")
    ax_e.errorbar(
        x, [100 * pt["qber"] for pt in points], yerr=[100 * pt["qber_sigma"] for pt in points],
        fmt="o", color="C1", label="Monte Carlo (±1σ)",
    )  # fmt: skip
    ax_e.set_xlabel("długość światłowodu L [km]")
    ax_e.set_ylabel("QBER $E_\\mu$ [%]")
    ax_e.set_title("QBER po siftingu")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(loc="upper left")

    exponent = int(np.floor(np.log10(p.background)))
    mantissa = p.background / 10.0**exponent
    fig.suptitle(
        f"GYS: α = {p.alpha_db_per_km} dB/km, η$_{{Bob}}$ = {100 * p.eta_bob:.1f}%, "
        f"Y$_0$ = ${mantissa:.1f}\\cdot10^{{{exponent}}}$, "
        f"e$_d$ = {100 * p.misalignment:.1f}%, μ = {cfg.mu}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or "nieznany"
    except OSError:
        return "nieznany"


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    points = [simulate_point(cfg, i) for i in range(len(cfg.lengths_km))]

    print(f"{'L [km]':>7} {'impulsy':>14} {'detekcje':>9} {'Q z':>6} {'E [%]':>7} {'E z':>6}")
    for pt in points:
        print(
            f"{pt['length_km']:>7.0f} {pt['pulses']:>14,d} {pt['detections']:>9,d} "
            f"{pt['gain_z']:>6.2f} {100 * pt['qber']:>7.3f} {pt['qber_z']:>6.2f}"
        )

    plot(cfg, points, ROOT / "figures" / f"{NAME}.png")
    results = ROOT / "results" / f"{NAME}.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    meta = {"commit": git_commit(), "python": platform.python_version(), "numpy": np.__version__}
    results.write_text(
        json.dumps({"meta": meta, "config": asdict(cfg), "points": points}, indent=2),
        encoding="utf-8",
    )

    worst = max(max(abs(pt["gain_z"]), abs(pt["qber_z"])) for pt in points)
    if worst >= cfg.z_max:
        print(f"BŁĄD: MC odbiega od modelu (max |z| = {worst:.2f} ≥ {cfg.z_max})")
        return 1
    print(f"OK: max |z| = {worst:.2f} < {cfg.z_max}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
