"""E07: skończony klucz.

Wynik: figures/e07_finite_key.png i results/e07_finite_key.json.
(a) stopa klucza w funkcji bloku n vs Tabela 1 Tomamichela et al.;
(b) ile uproszczenie (Hoeffding na samej próbce) zawyża długość klucza;
(c) „Ewa w szumie”: p-stwo wykrycia intercept-resend przy małym p dla różnych k (MC i analityka);
(d) kompromis: ułamek klucza w funkcji N = n + k i ułamka próbki k/N.
"""

import dataclasses
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qkdsim.analysis.entropy import h2
from qkdsim.analysis.intercept_resend import intercept_resend_model
from qkdsim.analysis.key_rate import GYS
from qkdsim.attacks.intercept_resend import InterceptResend
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.defenses.qber_test import tolerance
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link, run_bits
from qkdsim.postprocessing.finite_key import (
    abort_probability,
    key_length,
    key_length_sample_hoeffding,
    optimal_key_rate,
)
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
P_D = dark_count_per_detector(GYS.background)
TABLE_1 = {0.01: {1e4: 0.117, 1e5: 0.304, 1e6: 0.478}, 0.025: {1e4: 0.068, 1e5: 0.215, 1e6: 0.357}}


@dataclass(frozen=True)
class Config:
    seed: int
    eps_sec: float
    eps_cor: float
    qbers: tuple[float, ...]
    block_sizes: tuple[int, ...]
    compare_qber: float
    noise_misalignment: float
    noise_fractions: tuple[float, ...]
    noise_samples: tuple[int, ...]
    mc_sample: int
    mc_fractions: tuple[float, ...]
    mc_sessions: int
    mc_session_pulses: int


def noise_qber(cfg: Config, fraction: float) -> float:
    params = dataclasses.replace(GYS, misalignment=cfg.noise_misalignment)
    return intercept_resend_model(params, 0.0, 0.48, fraction, 1.0, P_D).clicks.qber


def mc_detection(cfg: Config) -> dict[float, float]:
    k, q_tol = cfg.mc_sample, tolerance(cfg.mc_sample, noise_qber(cfg, 0.0))
    out = {}
    for fraction in cfg.mc_fractions:
        link = Link(
            WCPSource((0.48,)),
            FiberChannel(0.0, GYS.alpha_db_per_km, cfg.noise_misalignment),
            DetectorPair(GYS.eta_bob, P_D),
            eve=InterceptResend(fraction, 1.0),
        )
        aborted = 0
        for s in range(cfg.mc_sessions):
            streams = RandomStreams.from_seed(
                np.random.SeedSequence([cfg.seed, int(1000 * fraction), s])
            )
            key = run_bits(link, cfg.mc_session_pulses, streams).key
            errors = int(np.count_nonzero(key.alice[:k] != key.bob[:k]))
            aborted += errors > math.floor(q_tol * k)
        out[fraction] = aborted / cfg.mc_sessions
    return out


def plot(cfg: Config, mc: dict[float, float], path: Path) -> dict[str, object]:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    (ax_r, ax_h), (ax_d, ax_m) = axes
    summary: dict[str, object] = {}

    for color, q in zip(("C0", "C1", "C2"), cfg.qbers, strict=True):
        rates = [optimal_key_rate(n, q, cfg.eps_sec, cfg.eps_cor)[0] for n in cfg.block_sizes]
        asym = 1 - 2 * h2(q)
        ax_r.semilogx(
            cfg.block_sizes,
            [100 * r for r in rates],
            "o-",
            color=color,
            markersize=3,
            label=f"Q = {100 * q:g}%",
        )
        ax_r.axhline(100 * asym, color=color, linestyle=":", linewidth=1)
        if q in TABLE_1:
            ns, vals = zip(*TABLE_1[q].items(), strict=True)
            ax_r.plot(
                ns, [100 * v for v in vals], "s", color=color, markerfacecolor="none", markersize=8
            )
        summary[f"rates_{q}"] = rates
    ax_r.plot([], [], "s", color="k", markerfacecolor="none", label="Tabela 1 artykułu")
    ax_r.plot([], [], ":", color="k", label="granica 1 − 2h(Q)")
    ax_r.set(
        xlabel="blok klucza n",
        ylabel="oczekiwana stopa klucza [% wykrytych sygnałów]",
        title="Stopa klucza w funkcji bloku (optymalne k i Q_tol)",
    )
    ax_r.legend(fontsize=8)

    q = cfg.compare_qber
    ratios = []
    for n in cfg.block_sizes:
        _, k, _ = optimal_key_rate(n, q, cfg.eps_sec, cfg.eps_cor)
        leak = 1.1 * n * h2(q)
        exact = key_length(n, k, q, leak, cfg.eps_sec, cfg.eps_cor)
        naive = key_length_sample_hoeffding(n, k, q, leak, cfg.eps_sec, cfg.eps_cor)
        ratios.append(100 * (naive / exact - 1) if exact > 0 else float("nan"))
    ax_h.semilogx(cfg.block_sizes, ratios, "o-", color="C3")
    ax_h.set(
        xlabel="blok klucza n",
        ylabel="zawyżenie ℓ przez uproszczenie [%]",
        title=f"Hoeffding na próbce vs twierdzenie (Q = {100 * q:g}%)",
    )
    summary["naive_overestimate_percent"] = ratios

    fractions = np.linspace(0, max(cfg.noise_fractions), 101)
    honest = noise_qber(cfg, 0.0)
    for color, k in zip(("C0", "C1", "C2"), cfg.noise_samples, strict=True):
        q_tol = tolerance(k, honest)
        ax_d.plot(
            100 * fractions,
            [abort_probability(k, noise_qber(cfg, f), q_tol) for f in fractions],
            color=color,
            label=f"k = {k:,}".replace(",", " "),
        )
    ax_d.plot(
        [100 * f for f in mc],
        list(mc.values()),
        "o",
        color="C0",
        label=f"Monte Carlo, k = {cfg.mc_sample}, {cfg.mc_sessions} sesji",
    )
    ax_d.set(
        xlabel="ułamek przechwyconych impulsów p [%]",
        ylabel="p-stwo wykrycia (przerwania)",
        title=f"Ewa w szumie: QBER tła {100 * honest:.1f}%, 1% fałszywych alarmów",
    )
    ax_d.legend(fontsize=8)
    summary["mc_detection"] = mc

    totals = np.geomspace(1e4, 1e8, 33)
    sample_fractions = np.geomspace(0.003, 0.5, 33)
    grid = np.zeros((len(sample_fractions), len(totals)))
    for j, total in enumerate(totals):
        for i, s in enumerate(sample_fractions):
            k, n = max(10, round(s * total)), round((1 - s) * total)
            best = 0.0
            for q_tol in np.linspace(q, q + 0.04, 25):
                length = key_length(
                    n, k, float(q_tol), 1.1 * n * h2(float(q_tol)), cfg.eps_sec, cfg.eps_cor
                )
                best = max(best, (1 - abort_probability(k, q, float(q_tol))) * length / total)
            grid[i, j] = best
    mesh = ax_m.pcolormesh(totals, sample_fractions, 100 * grid, shading="auto", cmap="viridis")
    ax_m.plot(
        totals, sample_fractions[np.argmax(grid, axis=0)], "w.-", label="optimum k/N dla danego N"
    )
    ax_m.set(
        xscale="log",
        yscale="log",
        xlabel="N = n + k (bity po siftingu)",
        ylabel="ułamek próbki k/N",
        title=f"Kompromis próbki (Q = {100 * q:g}%)",
    )
    fig.colorbar(mesh, ax=ax_m, label="ℓ / N [%]")
    ax_m.legend(fontsize=8, loc="upper right")
    for ax in (ax_r, ax_h, ax_d):
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(
        f"Skończony klucz (Tomamichel et al. 2012), ε_sec = {cfg.eps_sec:g}, "
        f"ε_cor = {cfg.eps_cor:g}, leak_EC = 1,1·n·h(Q_tol)"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    summary["optimal_sample_fraction"] = sample_fractions[np.argmax(grid, axis=0)].tolist()
    return summary


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    mc = mc_detection(cfg)
    k, honest = cfg.mc_sample, noise_qber(cfg, 0.0)
    q_tol = tolerance(k, honest)
    worst = 0.0
    for fraction, observed in mc.items():
        expected = abort_probability(k, noise_qber(cfg, fraction), q_tol)
        sigma = math.sqrt(max(expected * (1 - expected), 1e-4) / cfg.mc_sessions)
        worst = max(worst, abs(observed - expected) / sigma)
        print(f"p = {fraction:.2f}: wykrycie MC {observed:.3f}, analitycznie {expected:.3f}")
    summary = plot(cfg, mc, ROOT / "figures" / f"{NAME}.png")
    for q in cfg.qbers:
        print(
            f"Q = {100 * q:g}%: stopy "
            + ", ".join(f"{100 * r:.1f}%" for r in summary[f"rates_{q}"])
        )  # type: ignore[union-attr]
    print(
        "zawyżenie przez uproszczenie [%]: "
        + ", ".join(f"{r:.1f}" for r in summary["naive_overestimate_percent"])
    )  # type: ignore[union-attr]
    out = ROOT / "results" / f"{NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"config": asdict(cfg), "summary": summary}, indent=2, default=float), "utf-8"
    )
    print(f"max |z| MC vs analityka = {worst:.2f}")
    return 0 if worst < 4.0 else 1


if __name__ == "__main__":
    sys.exit(main())
