"""E02: sprawność Cascade f(Q) na kluczach syntetycznych.

Wynik: figures/e02_cascade_efficiency.png i results/e02_cascade_efficiency.json.
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
from qkdsim.classical import ClassicalChannel
from qkdsim.config import load_yaml
from qkdsim.postprocessing.cascade import cascade

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name


@dataclass(frozen=True)
class Config:
    seed: int
    key_bits: int
    runs: int
    qber: tuple[float, ...]
    passes: int


def measure(cfg: Config, index: int) -> dict[str, float]:
    rng = np.random.default_rng([cfg.seed, index])
    target = cfg.qber[index]
    efficiencies, residual = [], 0
    for _ in range(cfg.runs):
        alice = rng.integers(0, 2, cfg.key_bits, dtype=np.uint8)
        bob = alice ^ (rng.random(cfg.key_bits) < target).astype(np.uint8)
        q = float(np.mean(alice != bob))
        result = cascade(alice, bob, q, rng, ClassicalChannel(), passes=cfg.passes)
        efficiencies.append(result.leaked_bits / (cfg.key_bits * h2(q)))
        residual += int(not np.array_equal(result.corrected, alice))
    return {
        "qber": target,
        "f_mean": float(np.mean(efficiencies)),
        "f_std": float(np.std(efficiencies, ddof=1)),
        "runs_with_residual_errors": residual,
    }


def plot(cfg: Config, points: list[dict[str, float]], path: Path) -> None:
    q = [100 * p["qber"] for p in points]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.axhspan(1.1, 1.2, color="C2", alpha=0.12, label="typowe f Cascade (1,1–1,2)")
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1, label="granica Shannona (f = 1)")
    ax.errorbar(
        q, [p["f_mean"] for p in points], yerr=[p["f_std"] for p in points],
        fmt="o-", color="C0", label=f"Cascade, {cfg.passes} przebiegi (±1σ)",
    )  # fmt: skip
    ax.set_xlabel("QBER Q [%]")
    ax.set_ylabel("sprawność f = leak$_{EC}$ / (n·h(Q))")
    n_text = f"{cfg.key_bits:,}".replace(",", " ")
    ax.set_title(f"Sprawność korekcji błędów, n = {n_text} bitów, {cfg.runs} przebiegów")
    ax.set_ylim(0.95, 1.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    points = [measure(cfg, i) for i in range(len(cfg.qber))]
    print(f"{'Q [%]':>6} {'f':>7} {'σ_f':>7} {'z błędami':>10}")
    for p in points:
        print(
            f"{100 * p['qber']:>6.1f} {p['f_mean']:>7.3f} {p['f_std']:>7.3f} "
            f"{p['runs_with_residual_errors']:>10}"
        )
    plot(cfg, points, ROOT / "figures" / f"{NAME}.png")
    out = ROOT / "results" / f"{NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": asdict(cfg), "points": points}, indent=2), "utf-8")
    in_target = all(1.1 <= p["f_mean"] <= 1.2 for p in points)  # typowe f Cascade
    return 0 if in_target else 1


if __name__ == "__main__":
    sys.exit(main())
