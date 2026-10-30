"""E05: oślepianie detektorów i obrony (F3.8–F3.9).

Wyniki: figures/e05_blinding.png (atak w funkcji mocy impulsu wyzwalającego),
figures/e05_blinding_roc.png (krzywe ROC obron) i results/e05_blinding.json.
Kończy się błędem, gdy któryś punkt odbiega od modelu analitycznego o |z| ≥ 4.
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

from qkdsim.analysis.blinding import blinding_model
from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.wcp import gain as gain_model
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.attacks.blinding import DetectorBlinding, matching_send_fraction
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.defenses.power_monitor import PowerMonitor
from qkdsim.defenses.random_attenuation import RandomAttenuator, attenuation_z
from qkdsim.defenses.roc import auc, roc_curve
from qkdsim.detector import CLAVIS2, DetectorPair, dark_count_per_detector
from qkdsim.metrics import attenuation_tally, eve_known_fraction, gain, matched_qber
from qkdsim.pipeline import Link, run_counts, run_counts_parallel
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
P_D = dark_count_per_detector(GYS.background)
DETECTORS = DetectorPair(GYS.eta_bob, P_D, CLAVIS2)
THETA_E = math.asin(math.sqrt(GYS.misalignment))
Z_MAX = 4.0


@dataclass(frozen=True)
class Config:
    seed: int
    mu: float
    length_km: float
    trigger_powers_uw: tuple[float, ...]
    pulses: int
    attenuator_fraction: float
    attenuator_factor: float
    roc_trigger_uw: float
    roc_session_pulses: tuple[int, ...]
    roc_sessions: int
    monitor_tap: float
    monitor_noise_w: float
    monitor_cw_w: tuple[float, ...]
    monitor_windows: int


def make_attack(cfg: Config, power_w: float, calibrated: bool) -> DetectorBlinding:
    honest = matched_basis_clicks(cfg.mu, GYS.eta(cfg.length_km), P_D, GYS.misalignment)
    return DetectorBlinding(
        power_w,
        send_fraction=matching_send_fraction(honest.detected, cfg.mu),
        mimic_qber=honest.qber,
        compensation_rad=THETA_E if calibrated else 0.0,
    )


def make_link(cfg: Config, eve: DetectorBlinding | None, attenuate: bool) -> Link:
    attenuator = RandomAttenuator(cfg.attenuator_fraction, cfg.attenuator_factor)
    return Link(
        WCPSource((cfg.mu,)),
        FiberChannel(cfg.length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DETECTORS,
        eve=eve,
        attenuator=attenuator if attenuate else None,
    )


def expected_ratio(cfg: Config) -> float:
    eta = GYS.eta(cfg.length_km)
    return gain_model(cfg.mu, cfg.attenuator_factor * eta, GYS.background) / gain_model(
        cfg.mu, eta, GYS.background
    )


def sweep(cfg: Config) -> list[dict[str, float]]:
    points = []
    for i, power_uw in enumerate(cfg.trigger_powers_uw):
        for calibrated in (True, False):
            eve = make_attack(cfg, power_uw * 1e-6, calibrated)
            seed = np.random.SeedSequence([cfg.seed, i, int(calibrated)])
            counts = run_counts_parallel(make_link(cfg, eve, False), cfg.pulses, seed, 1_000_000)
            model = blinding_model(DETECTORS, eve, cfg.mu, GYS.misalignment)
            point = {"power_uw": power_uw, "calibrated": calibrated, "detected": model.detected}
            if model.detected > 0.0:
                g, e, k = gain(counts), matched_qber(counts), eve_known_fraction(counts)
                point |= {
                    "gain": g.value,
                    "gain_model": model.detected,
                    "gain_z": g.z_score(model.detected),
                    "qber": e.value,
                    "qber_sigma": e.sigma,
                    "qber_model": model.qber,
                    "qber_z": e.z_score(model.qber),
                    "known": k.value,
                    "known_model": model.known_fraction,
                }
            else:
                point |= {
                    "gain": 0.0,
                    "gain_z": 0.0,
                    "qber_z": 0.0,
                    "mc_detected": int(counts.detected.sum()),
                }
            # Bez kliknięć test tłumienia nie ma sensu (atak i tak zdradza zerowy zysk).
            point["attenuation_z"] = float("nan")
            if calibrated and model.detected > 0.0:
                seed = np.random.SeedSequence([cfg.seed, i, 7])
                guarded = run_counts_parallel(
                    make_link(cfg, eve, True), cfg.pulses, seed, 1_000_000
                )
                point["attenuation_z"] = attenuation_z(
                    attenuation_tally(guarded), expected_ratio(cfg)
                )
            points.append(point)
    return points


def attenuation_roc(cfg: Config) -> dict[int, tuple[list[float], list[float], float]]:
    eve = make_attack(cfg, cfg.roc_trigger_uw * 1e-6, calibrated=True)
    ratio, out = expected_ratio(cfg), {}
    for size in cfg.roc_session_pulses:
        scores: dict[bool, list[float]] = {False: [], True: []}
        for attacked in (False, True):
            for s in range(cfg.roc_sessions):
                streams = RandomStreams.from_seed(
                    np.random.SeedSequence([cfg.seed, size, s, attacked])
                )
                link = make_link(cfg, eve if attacked else None, True)
                tally = attenuation_tally(run_counts(link, size, streams))
                # Podejrzane są mniejsze z niż oczekiwane, więc wynikiem jest −z.
                scores[attacked].append(-attenuation_z(tally, ratio))
        honest, bad = np.array(scores[False]), np.array(scores[True])
        fpr, tpr = roc_curve(honest, bad)
        out[size] = (fpr.tolist(), tpr.tolist(), auc(honest, bad))
    return out


def monitor_roc(cfg: Config) -> dict[float, tuple[list[float], list[float], float]]:
    monitor = PowerMonitor(cfg.monitor_tap, cfg.monitor_noise_w)
    rng = np.random.default_rng([cfg.seed, 5])
    honest = monitor.readings(0.0, cfg.monitor_windows, rng)
    out = {}
    for cw in cfg.monitor_cw_w:
        attacked = monitor.readings(cw, cfg.monitor_windows, rng)
        fpr, tpr = roc_curve(honest, attacked)
        out[cw] = (fpr.tolist(), tpr.tolist(), auc(honest, attacked))
    return out


def plot_sweep(cfg: Config, points: list[dict[str, float]], path: Path) -> None:
    grid = np.linspace(min(cfg.trigger_powers_uw), max(cfg.trigger_powers_uw), 300)
    honest = matched_basis_clicks(cfg.mu, GYS.eta(cfg.length_km), P_D, GYS.misalignment)
    low, high = (1e6 * w for w in CLAVIS2.perfect_window_w)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    (ax_e, ax_k), (ax_q, ax_z) = axes
    for calibrated, color, label in ((True, "C0", "Ewa skalibrowana"), (False, "C3", "Ewa naiwna")):
        models = [
            blinding_model(
                DETECTORS, make_attack(cfg, p * 1e-6, calibrated), cfg.mu, GYS.misalignment
            )
            for p in grid
        ]
        alive = [m.detected > 0 for m in models]
        pts = [p for p in points if p["calibrated"] == calibrated and p["detected"] > 0]
        x = [p["power_uw"] for p in pts]
        ax_e.plot(
            grid[alive],
            [100 * m.qber for m, a in zip(models, alive, strict=True) if a],
            color=color,
            label=label,
        )
        ax_e.errorbar(
            x,
            [100 * p["qber"] for p in pts],
            yerr=[100 * p["qber_sigma"] for p in pts],
            fmt="o",
            color=color,
            markersize=4,
        )
        ax_k.plot(
            grid[alive],
            [m.known_fraction for m, a in zip(models, alive, strict=True) if a],
            color=color,
            label=label,
        )
        ax_k.plot(x, [p["known"] for p in pts], "o", color=color, markersize=4)
        ax_q.plot(grid, [m.detected / honest.detected for m in models], color=color, label=label)
        ax_q.plot(
            [p["power_uw"] for p in points if p["calibrated"] == calibrated],
            [p["gain"] / honest.detected for p in points if p["calibrated"] == calibrated],
            "o",
            color=color,
            markersize=4,
        )
    calibrated_pts = [p for p in points if p["calibrated"]]
    ax_z.plot(
        [p["power_uw"] for p in calibrated_pts],
        [p["attenuation_z"] for p in calibrated_pts],
        "o-",
        color="C0",
        label="Ewa skalibrowana",
    )
    ax_z.axhline(
        -Z_MAX, color="k", linestyle="--", linewidth=1, label=f"próg alarmu z = −{Z_MAX:g}"
    )
    ax_z.axhline(Z_MAX, color="k", linestyle="--", linewidth=1)
    ax_e.axhline(100 * honest.qber, color="0.4", linestyle=":", label="QBER bez ataku")
    ax_q.axhline(1.0, color="0.4", linestyle=":", label="zysk bez ataku")
    for ax in (ax_e, ax_k, ax_q, ax_z):
        ax.axvspan(low, high, color="C2", alpha=0.12)
        ax.set_xlabel("moc impulsu wyzwalającego P [µW]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    ax_e.set(ylabel="QBER [%]", title="QBER (zielone: okno doskonałego ataku)")
    ax_k.set(ylabel="bity znane Ewie / bity po siftingu", title="Wiedza Ewy")
    ax_q.set(ylabel="zysk / zysk bez ataku", title="Zysk (Ewa dobiera częstość impulsów)")
    ax_z.set(
        ylabel="z testu losowego tłumienia", title="Obrona (b): losowe tłumienie 3 dB w 10% bramek"
    )
    ax_z.set_yscale("symlog", linthresh=10)
    fig.suptitle(
        f"Oślepianie detektorów (progi Clavis2), GYS, L = {cfg.length_km:g} km, μ = {cfg.mu}"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(
    cfg: Config,
    att: dict[int, tuple[list[float], list[float], float]],
    mon: dict[float, tuple[list[float], list[float], float]],
    path: Path,
) -> None:
    fig, (ax_a, ax_m) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for size, (fpr, tpr, area) in att.items():
        ax_a.step(
            fpr,
            tpr,
            where="post",
            label=f"{size:,} impulsów na sesję, AUC = {area:.3f}".replace(",", " "),
        )
    for cw, (fpr, tpr, area) in mon.items():
        ax_m.step(fpr, tpr, where="post", label=f"CW = {cw * 1e6:g} µW, AUC = {area:.3f}")
    for ax in (ax_a, ax_m):
        ax.plot([0, 1], [0, 1], color="0.6", linestyle=":")
        ax.set(xlabel="odsetek fałszywych alarmów (FPR)", ylabel="odsetek wykrytych ataków (TPR)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    ax_a.set_title(f"Losowe tłumienie, Ewa skalibrowana, P = {cfg.roc_trigger_uw:g} µW")
    ax_m.set_title(
        f"Monitor mocy: odczep {100 * cfg.monitor_tap:g}%, szum {cfg.monitor_noise_w * 1e9:g} nW"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    points = sweep(cfg)
    att, mon = attenuation_roc(cfg), monitor_roc(cfg)
    header = f"{'P [µW]':>7} {'Ewa':>12} {'QBER [%]':>9} {'z':>6} {'zysk z':>7}"
    print(f"{header} {'wiedza':>7} {'z tłum.':>8}")
    for p in points:
        print(
            f"{p['power_uw']:>7.0f} {'skalibrowana' if p['calibrated'] else 'naiwna':>12} "
            f"{100 * p.get('qber', float('nan')):>9.2f} {p['qber_z']:>6.2f} {p['gain_z']:>7.2f} "
            f"{p.get('known', float('nan')):>7.3f} {p.get('attenuation_z', float('nan')):>8.1f}"
        )
    for size, (_, _, area) in att.items():
        print(f"ROC losowego tłumienia, {size} impulsów na sesję: AUC = {area:.3f}")
    for cw, (_, _, area) in mon.items():
        print(f"ROC monitora mocy, CW = {cw:.2e} W: AUC = {area:.3f}")
    plot_sweep(cfg, points, ROOT / "figures" / f"{NAME}.png")
    plot_roc(cfg, att, mon, ROOT / "figures" / f"{NAME}_roc.png")
    out = ROOT / "results" / f"{NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "points": points,
        "attenuation_auc": {str(k): v[2] for k, v in att.items()},
        "monitor_auc": {str(k): v[2] for k, v in mon.items()},
    }
    out.write_text(json.dumps(payload, indent=2), "utf-8")
    worst = max(max(abs(p["gain_z"]), abs(p["qber_z"])) for p in points)
    print(f"max |z| względem modelu = {worst:.2f}")
    return 0 if worst < Z_MAX else 1


if __name__ == "__main__":
    sys.exit(main())
