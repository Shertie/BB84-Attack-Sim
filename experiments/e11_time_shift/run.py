"""E11: atak 4, time-shift.

Wynik: figures/e11_time_shift.png, figures/e11_time_shift.md i results/e11_time_shift.json.
(a) krzywe sprawności dopasowane do Tabeli I(a) Zhao et al. 2008 i przesunięcia Ewy A, B;
(b) informacja Ewy na wykryty bit w funkcji rozsunięcia krzywych Δ: model i Monte Carlo,
    z obroną (losowe przypisanie bit ↔ detektor) i bez;
(c) QBER pod atakiem na tle QBER bez ataku (Ewa wyrównuje zysk linią bezstratną);
(d) granice K_U i K_L Zhao et al. na wykryty bit: od jakiego Δ atak łamie bezpieczeństwo.
Do tego odtworzenie liczb z Tabeli I(c) artykułu z jego własnych danych.
"""

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.time_shift import (
    ZHAO_BACKGROUND,
    ZHAO_DETECTIONS,
    ZHAO_F_EC,
    ZHAO_MATCHED,
    ZHAO_MU,
    ZHAO_P_A,
    ZHAO_TABLE,
    TimeShiftModel,
    balancing_probability,
    eve_information,
    fit_zhao,
    gain_matching_transmission,
    symmetric_mismatch,
    time_shift_model,
    zhao_key_bounds,
)
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.attacks.time_shift import TimeShift, detection_table
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.detector import DetectorPair, EfficiencyMismatch, dark_count_per_detector
from qkdsim.pipeline import Link
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
P_D = dark_count_per_detector(GYS.background)
FIT = fit_zhao()
WIDTH_NS = float(np.mean(FIT.mismatch.widths_ns))  # rodzina symetryczna do przeglądu po Δ
# Zhao et al. 2008, Tabela I(c).
PAPER = {"p_a": 0.230, "qber": 0.0568, "k_upper": 1131, "k_lower": 1297}


@dataclass(frozen=True)
class Config:
    seed: int
    length_km: float
    mu: float
    deltas_ps: tuple[float, ...]
    mc_pulses: int
    criterion_mus: tuple[float, ...]


def transmittance(length_km: float) -> float:
    return float(10.0 ** (-GYS.alpha_db_per_km * length_km / 10.0))


def attacked(
    mu: float, length_km: float, mismatch: EfficiencyMismatch, swap: bool = False
) -> tuple[TimeShiftModel, float]:
    """Model ataku z tłumieniem Ewy, które wyrównuje zysk Boba."""
    transmission = gain_matching_transmission(
        mu, transmittance(length_km), GYS.eta_bob, P_D, GYS.misalignment, mismatch, FIT.shifts_ns
    )
    if transmission is None:
        raise ValueError(f"Ewa nie wyrówna zysku przy {length_km} km")
    plain = time_shift_model(
        mu, transmission, GYS.eta_bob, P_D, GYS.misalignment, mismatch, FIT.shifts_ns
    )
    model = time_shift_model(
        mu,
        transmission,
        GYS.eta_bob,
        P_D,
        GYS.misalignment,
        mismatch,
        FIT.shifts_ns,
        plain.p_a,
        swap,
    )
    return model, transmission


def mc_point(
    cfg: Config, mismatch: EfficiencyMismatch, swap: bool, index: int
) -> tuple[list[int], int]:
    model, transmission = attacked(cfg.mu, cfg.length_km, mismatch)
    link = Link(
        WCPSource((cfg.mu,)),
        FiberChannel(cfg.length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D, timing=mismatch),
        eve=TimeShift(*FIT.shifts_ns, p_a=model.p_a, transmission=transmission),
        random_assignment=swap,
    )
    streams = RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, index, int(swap)]))
    table, matched = detection_table(link, cfg.mc_pulses, streams)
    return table.ravel().tolist(), matched


def information_sigma(table: npt.NDArray[np.float64]) -> float:
    """Niepewność estymaty 1 − Σ P(k)h(p_k) metodą delta (bez zmienności P(k))."""
    per_x = table.sum(axis=-1).reshape(-1, 2)
    total = per_x.sum()
    variance = 0.0
    for zeros, ones in per_x:
        n = zeros + ones
        p = zeros / n
        if 0.0 < p < 1.0:
            slope = math.log2((1.0 - p) / p)
            variance += (n / total) ** 2 * slope**2 * p * (1.0 - p) / n
    return math.sqrt(variance)


def qber_of(table: npt.NDArray[np.float64]) -> float:
    return float(table[..., 0, 1].sum() + table[..., 1, 0].sum()) / float(table.sum())


def criterion_delta(mu: float, length_km: float) -> float:
    """Najmniejsze Δ, przy którym K_L > K_U (atak łamie bezpieczeństwo); rodzina symetryczna."""

    def gap(delta_ns: float) -> float:
        model, _ = attacked(mu, length_km, symmetric_mismatch(delta_ns, WIDTH_NS))
        bounds = zhao_key_bounds(model.table, 1.0, mu, GYS.background, GYS.f_ec)
        return bounds.lower - bounds.upper

    return float(brentq(gap, 1e-4, 0.3))


def hiding_distance(mu: float) -> float:
    """Najmniejsza odległość, na której linia bezstratna wyrównuje spadek zysku (krzywe Zhao)."""

    def gap(length_km: float) -> float:
        honest = matched_basis_clicks(mu, GYS.eta(length_km), P_D, GYS.misalignment).detected
        full = time_shift_model(
            mu, 1.0, GYS.eta_bob, P_D, GYS.misalignment, FIT.mismatch, FIT.shifts_ns
        )
        return full.gain - honest

    return float(brentq(gap, 0.1, 150.0))


def comma(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def scientific(value: float) -> str:
    return f"{value:.3e}".replace(".", ",")


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    deltas_ns = [d / 1000.0 for d in cfg.deltas_ps]
    tasks = [(symmetric_mismatch(d, WIDTH_NS), swap) for swap in (False, True) for d in deltas_ns]
    tasks += [(FIT.mismatch, False), (FIT.mismatch, True)]
    with ProcessPoolExecutor() as pool:
        futures = [
            pool.submit(mc_point, cfg, mismatch, swap, i)
            for i, (mismatch, swap) in enumerate(tasks)
        ]
        results = [f.result() for f in futures]
    tables = [np.array(t, dtype=np.float64).reshape(2, 2, 2, 2) for t, _ in results]
    matched = [m for _, m in results]
    n = len(deltas_ns)
    attack_tables, defense_tables = tables[:n], tables[n : 2 * n]
    zhao_attack, zhao_defense = tables[2 * n], tables[2 * n + 1]

    # Odtworzenie Tabeli I(c) z danych artykułu.
    at_a, at_b = (np.array(d, dtype=np.float64) for d in ZHAO_DETECTIONS)
    p_a = balancing_probability(at_a, at_b)
    mixed = np.stack([ZHAO_P_A * ZHAO_TABLE[0], (1.0 - ZHAO_P_A) * ZHAO_TABLE[1]])
    paper_bounds = zhao_key_bounds(mixed, ZHAO_MATCHED, ZHAO_MU, ZHAO_BACKGROUND, ZHAO_F_EC)
    paper_info = eve_information(mixed)

    honest = matched_basis_clicks(cfg.mu, GYS.eta(cfg.length_km), P_D, GYS.misalignment)
    zhao_model, zhao_transmission = attacked(cfg.mu, cfg.length_km, FIT.mismatch)
    zhao_mc_bounds = zhao_key_bounds(zhao_attack, matched[2 * n], cfg.mu, GYS.background, GYS.f_ec)
    model_bounds = zhao_key_bounds(zhao_model.table, 1.0, cfg.mu, GYS.background, GYS.f_ec)
    thresholds = {mu: criterion_delta(mu, cfg.length_km) for mu in cfg.criterion_mus}
    zhao_curves = {
        mu: zhao_key_bounds(
            attacked(mu, cfg.length_km, FIT.mismatch)[0].table, 1.0, mu, GYS.background, GYS.f_ec
        )
        for mu in cfg.criterion_mus
    }
    hiding = hiding_distance(cfg.mu)

    lines = [
        "## Odtworzenie Tabeli I(c) Zhao et al. 2008 z danych artykułu",
        "",
        "| Wielkość | Artykuł | Z Tabeli I(a)–(b) |",
        "| --- | --- | --- |",
        f"| p_A (zera i jedynki Boba w równowadze) | {comma(100 * PAPER['p_a'], 1)}% | "
        f"{comma(100 * p_a, 2)}% |",
        f"| QBER E | {comma(100 * PAPER['qber'], 2)}% | {comma(100 * paper_bounds.qber, 3)}% |",
        f"| K_U (Ewa zna niedopasowanie) | {PAPER['k_upper']} bitów | "
        f"{comma(paper_bounds.upper, 1)} bitów |",
        f"| K_L (Alice i Bob nie widzą ataku) | {PAPER['k_lower']} bitów | "
        f"{comma(paper_bounds.lower, 1)} bitów |",
        f"| informacja Ewy na wykryty bit | — | {comma(paper_info, 3)} |",
        "",
        "Krzywe Gaussa dopasowane do Tabeli I(a) (Δ = 100 ps, równe szczyty, po odjęciu ciemnych "
        f"zliczeń): środki {comma(1000 * FIT.mismatch.centers_ns[0], 1)} i "
        f"{comma(1000 * FIT.mismatch.centers_ns[1], 1)} ps, szerokości "
        f"{comma(1000 * FIT.mismatch.widths_ns[0], 1)} i "
        f"{comma(1000 * FIT.mismatch.widths_ns[1], 1)} ps względem nominalnego przyjścia "
        f"(przecięcia krzywych); przesunięcia Ewy A = {comma(1000 * FIT.shifts_ns[0], 1)} ps, "
        f"B = {comma(1000 * FIT.shifts_ns[1], 1)} ps.",
        "",
        f"## Te same krzywe w łączu GYS ({cfg.length_km:g} km, μ = {comma(cfg.mu, 2)}, "
        f"{cfg.mc_pulses:.0e} impulsów)".replace("e+0", "·10^"),
        "",
        "| Wielkość | Bez ataku | Atak (model) | Atak (MC) | Atak + obrona (MC) |",
        "| --- | --- | --- | --- | --- |",
        f"| zysk w zgodnej bazie | {scientific(honest.detected)} | "
        f"{scientific(zhao_model.gain)} | {scientific(zhao_attack.sum() / matched[2 * n])} | "
        f"{scientific(zhao_defense.sum() / matched[2 * n + 1])} |",
        f"| QBER | {comma(100 * honest.qber, 2)}% | {comma(100 * zhao_model.qber, 2)}% | "
        f"{comma(100 * qber_of(zhao_attack), 2)}% | {comma(100 * qber_of(zhao_defense), 2)}% |",
        f"| informacja Ewy na wykryty bit | 0 | {comma(zhao_model.eve_information, 3)} | "
        f"{comma(eve_information(zhao_attack), 3)} ± {comma(information_sigma(zhao_attack), 3)} | "
        f"{comma(eve_information(zhao_defense), 3)} ± "
        f"{comma(information_sigma(zhao_defense), 3)} |",
        f"| K_U / K_L na wykryty bit | — | {comma(model_bounds.upper / model_bounds.detected, 3)}"
        f" / {comma(model_bounds.lower / model_bounds.detected, 3)} | "
        f"{comma(zhao_mc_bounds.upper / zhao_mc_bounds.detected, 3)} / "
        f"{comma(zhao_mc_bounds.lower / zhao_mc_bounds.detected, 3)} | — |",
        "",
        f"Ewa przesuwa impuls do A z p-stwem {comma(100 * zhao_model.p_a, 1)}% i tłumi do "
        f"{comma(100 * zhao_transmission, 1)}% za linią bezstratną. Spadek sprawności przy "
        f"przesunięciu ukrywa od {comma(hiding, 1)} km; bliżej Bob widzi niższy zysk.",
        "",
        "## Od jakiego Δ atak łamie bezpieczeństwo (K_L > K_U), rodzina symetryczna",
        "",
        "| μ | Δ* | Krzywe Zhao (Δ = 100 ps): K_U / K_L na wykryty bit |",
        "| --- | --- | --- |",
    ]
    for mu, delta in thresholds.items():
        b = zhao_curves[mu]
        verdict = "atak łamie" if b.attack_succeeds else "atak nie łamie"
        lines.append(
            f"| {comma(mu, 2)} | {comma(1000 * delta, 0)} ps | "
            f"{comma(b.upper / b.detected, 3)} / {comma(b.lower / b.detected, 3)} ({verdict}) |"
        )
    lines += [
        "",
        "Przy większym μ GLLP odrzuca więcej klucza z impulsów wielofotonowych, więc time-shift "
        "musi dać Ewie więcej, zanim przekroczy to, co Alice i Bob i tak usuwają.",
    ]

    grid_ns = np.linspace(0.0, 0.3, 61)
    info_line = [
        attacked(cfg.mu, cfg.length_km, symmetric_mismatch(d, WIDTH_NS))[0].eve_information
        for d in grid_ns
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    ax_c, ax_i, ax_q, ax_k = axes.ravel()

    t = np.linspace(-0.6, 0.6, 241)
    for j, color in ((0, "C0"), (1, "C3")):
        ax_c.plot(1000 * t, FIT.mismatch.relative(t, j), color=color, label=f"detektor {j}")
    for name, shift in zip(("A", "B"), FIT.shifts_ns, strict=True):
        ax_c.axvline(1000 * shift, color="0.4", linestyle=":")
        ax_c.annotate(name, (1000 * shift, 1.05), ha="center")
    ax_c.set(
        xlabel="czas przyjścia względem nominalnego [ps]",
        ylabel="sprawność η_j(t) / η",
        title="Krzywe dopasowane do Tabeli I(a) Zhao et al. 2008 (Δ = 100 ps)",
        ylim=(0, 1.15),
    )
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3)

    ps = np.array(cfg.deltas_ps)
    ax_i.plot(1000 * grid_ns, info_line, color="C3", label="model: atak")
    ax_i.axhline(0.0, color="C2", label="model: atak + losowe przypisanie")
    ax_i.errorbar(
        ps,
        [eve_information(t) for t in attack_tables],
        yerr=[information_sigma(t) for t in attack_tables],
        fmt="o",
        color="C3",
        label="Monte Carlo: atak",
    )
    ax_i.errorbar(
        ps,
        [eve_information(t) for t in defense_tables],
        yerr=[information_sigma(t) for t in defense_tables],
        fmt="s",
        color="C2",
        label="Monte Carlo: atak + obrona",
    )
    ax_i.errorbar(
        [100.0],
        [eve_information(zhao_attack)],
        yerr=[information_sigma(zhao_attack)],
        fmt="*",
        markersize=11,
        color="k",
        label="krzywe Zhao, Monte Carlo",
    )
    ax_i.plot([100.0], [paper_info], "D", color="C1", label="Zhao et al. 2008, Tabela I(b)")
    ax_i.set(
        xlabel="rozsunięcie krzywych Δ [ps]",
        ylabel="informacja Ewy [bit na wykryty bit]",
        title=f"Wiedza Ewy przy niezmienionym QBER i zysku ({cfg.length_km:g} km)",
    )
    ax_i.legend(fontsize=7)
    ax_i.grid(True, alpha=0.3)

    ax_q.axhline(100 * honest.qber, color="k", linestyle="--", label="bez ataku (model)")
    ax_q.errorbar(
        ps,
        [100 * qber_of(t) for t in attack_tables],
        yerr=[100 * math.sqrt(honest.qber * (1 - honest.qber) / t.sum()) for t in attack_tables],
        fmt="o",
        color="C3",
        label="atak, Monte Carlo",
    )
    ax_q.set(
        xlabel="rozsunięcie krzywych Δ [ps]",
        ylabel="QBER [%]",
        title="QBER pod atakiem (Ewa wyrównuje zysk)",
        ylim=(0, 2 * 100 * honest.qber),
    )
    ax_q.legend(fontsize=8)
    ax_q.grid(True, alpha=0.3)

    for mu, color in zip(cfg.criterion_mus, ("C0", "C1"), strict=True):
        upper, lower = [], []
        for d in grid_ns:
            model, _ = attacked(mu, cfg.length_km, symmetric_mismatch(d, WIDTH_NS))
            b = zhao_key_bounds(model.table, 1.0, mu, GYS.background, GYS.f_ec)
            upper.append(b.upper / b.detected)
            lower.append(b.lower / b.detected)
        label = comma(mu, 2)
        ax_k.plot(1000 * grid_ns, upper, color=color, label=f"K_U, μ = {label}")
        ax_k.plot(1000 * grid_ns, lower, color=color, linestyle="--", label=f"K_L, μ = {label}")
        ax_k.axvline(1000 * thresholds[mu], color=color, linestyle=":")
    ax_k.axhline(0.0, color="0.5", linewidth=0.8)
    ax_k.set(
        xlabel="rozsunięcie krzywych Δ [ps]",
        ylabel="klucz na wykryty bit",
        title="Atak łamie bezpieczeństwo, gdy K_L > K_U (linie kropkowane: Δ*)",
    )
    ax_k.legend(fontsize=8)
    ax_k.grid(True, alpha=0.3)
    fig.savefig(ROOT / "figures" / f"{NAME}.png", dpi=150)
    plt.close(fig)

    text = "\n".join(lines) + "\n"
    (ROOT / "figures" / f"{NAME}.md").write_text(text, "utf-8")
    print(text)
    out = ROOT / "results" / f"{NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "paper_reproduction": {
                    "p_a": p_a,
                    "qber": paper_bounds.qber,
                    "k_upper": paper_bounds.upper,
                    "k_lower": paper_bounds.lower,
                },
                "information": {
                    "attack": [eve_information(t) for t in attack_tables],
                    "defense": [eve_information(t) for t in defense_tables],
                    "zhao_curves_mc": eve_information(zhao_attack),
                    "zhao_curves_model": zhao_model.eve_information,
                },
                "criterion_delta_ns": {str(mu): d for mu, d in thresholds.items()},
                "hiding_distance_km": hiding,
            },
            indent=2,
        ),
        "utf-8",
    )

    def z_info(table: npt.NDArray[np.float64], expected: float) -> float:
        return (eve_information(table) - expected) / information_sigma(table)

    model_info = [
        attacked(cfg.mu, cfg.length_km, symmetric_mismatch(d, WIDTH_NS))[0].eve_information
        for d in deltas_ns
    ]
    checks = [
        abs(p_a - PAPER["p_a"]) < 5e-4,
        abs(paper_bounds.upper - PAPER["k_upper"]) <= 1.0,
        abs(paper_bounds.lower - PAPER["k_lower"]) <= 1.0,
        abs(z_info(zhao_attack, zhao_model.eve_information)) < 4.0,
        all(
            abs(z_info(t, m)) < 4.0
            for t, m in zip(attack_tables, model_info, strict=True)
            if m > 0.01
        ),
        all(eve_information(t) < 4.0 * information_sigma(t) + 1e-3 for t in defense_tables),
        all(
            abs(qber_of(t) - honest.qber) < 4.0 * math.sqrt(honest.qber / t.sum())
            for t in attack_tables
        ),
    ]
    print("sprawdzenia:", checks)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
