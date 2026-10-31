"""E20: uzgadnianie LDPC z syndromem zamiast Cascade (F7).

Wynik: figures/e20_ldpc.png, figures/e20_ldpc.md i results/e20_ldpc.json.
(a) ewolucja gęstości: progi BP kodów o d_v = 3 i różnych sprawnościach R; dla (3,6) próg 0,084
    z literatury (Richardson, Urbanke 2001); sprawność asymptotyczna f = (1 − R)/h(Q*);
(b) wodospad: p-stwo błędu ramki (FER) kodu (3,6) w funkcji Q dla rosnącej długości ramki;
(c) reżim QKD: FER w funkcji zakładanej sprawności f przy kilku Q, f potrzebne do FER = 10%
    na tle Cascade (wyciek, liczba wiadomości) i granicy z ewolucji gęstości;
(d) pełny postprocessing na torze GYS: Cascade a LDPC.
"""

import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qkdsim.analysis.entropy import h2
from qkdsim.analysis.key_rate import GYS
from qkdsim.channel import FiberChannel
from qkdsim.classical import ClassicalChannel
from qkdsim.config import load_yaml
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link, run_sifted
from qkdsim.postprocessing.cascade import cascade
from qkdsim.postprocessing.ldpc import (
    bp_threshold,
    check_degrees,
    decode,
    random_code,
    syndrome_length,
)
from qkdsim.postprocessing.run import KeyBalance, PostprocessingConfig, postprocess
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
LITERATURE_36 = 0.084  # próg BP kodu (3,6) na BSC (Richardson, Urbanke 2001)


@dataclass(frozen=True)
class Config:
    seed: int
    de_rates: tuple[float, ...]
    waterfall_qbers: tuple[float, ...]
    waterfall_lengths: tuple[int, ...]
    waterfall_trials: tuple[int, ...]
    qbers: tuple[float, ...]
    efficiencies: tuple[float, ...]
    frame: int
    trials: int
    max_iterations: int
    target_fer: float
    cascade_runs: int
    length_km: float
    mu: float
    pulses: int


@dataclass(frozen=True)
class QkdRow:
    qber: float
    fer: tuple[float, ...]
    ldpc_efficiency: float  # f przy FER = target (interpolacja liniowa)
    asymptotic_efficiency: float
    cascade_efficiency: float
    cascade_messages: float
    ldpc_iterations: float


def comma(value: float, digits: int) -> str:
    return f"{round(value, digits) + 0.0:.{digits}f}".replace(".", ",")


def frame_errors(
    n: int, m: int, qber: float, trials: int, iterations: int, rng: np.random.Generator
) -> tuple[float, float]:
    """(FER, średnia liczba iteracji); nowy kod co 10 prób."""
    failures = total_iterations = 0
    code = random_code(n, m, rng)
    for trial in range(trials):
        if trial and trial % 10 == 0:
            code = random_code(n, m, rng)
        x = rng.integers(0, 2, n, dtype=np.uint8)
        y = x ^ (rng.random(n) < qber).astype(np.uint8)
        result = decode(code, y, code.syndrome(x), qber, iterations)
        failures += not (result.success and np.array_equal(result.bits, x))
        total_iterations += result.iterations
    return failures / trials, total_iterations / trials


def crossing(xs: tuple[float, ...], fers: tuple[float, ...], target: float) -> float:
    """Najmniejsze f z FER ≤ target (interpolacja liniowa między punktami siatki)."""
    for i in range(1, len(xs)):
        if fers[i] <= target < fers[i - 1]:
            return xs[i - 1] + (fers[i - 1] - target) / (fers[i - 1] - fers[i]) * (
                xs[i] - xs[i - 1]
            )
    return xs[0] if fers[0] <= target else math.inf


def asymptotic_efficiency(qber: float, rates: tuple[float, ...], thresholds: list[float]) -> float:
    """(1 − R)/h(Q) dla R, przy którym próg BP równa się Q (interpolacja po R)."""
    rate = float(np.interp(qber, thresholds[::-1], rates[::-1]))
    return (1.0 - rate) / h2(qber)


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    rng = np.random.default_rng(cfg.seed)

    # (a) Ewolucja gęstości.
    thresholds = [bp_threshold(3, check_degrees(rate)) for rate in cfg.de_rates]

    # (b) Wodospad kodu (3,6).
    waterfall = {
        n: [
            frame_errors(n, n // 2, q, trials, cfg.max_iterations, rng)[0]
            for q in cfg.waterfall_qbers
        ]
        for n, trials in zip(cfg.waterfall_lengths, cfg.waterfall_trials, strict=True)
    }

    # (c) Reżim QKD.
    rows = []
    for qber in cfg.qbers:
        results = [
            frame_errors(
                cfg.frame,
                syndrome_length(cfg.frame, qber, f),
                qber,
                cfg.trials,
                cfg.max_iterations,
                rng,
            )
            for f in cfg.efficiencies
        ]
        fers = tuple(r[0] for r in results)
        leaks, messages = [], []
        for _ in range(cfg.cascade_runs):
            x = rng.integers(0, 2, cfg.frame, dtype=np.uint8)
            errors = (rng.random(cfg.frame) < qber).astype(np.uint8)
            channel = ClassicalChannel()
            result = cascade(x, x ^ errors, qber, rng, channel)
            leaks.append(result.leaked_bits / (cfg.frame * h2(errors.mean())))
            messages.append(len(channel.transcript))
        rows.append(
            QkdRow(
                qber,
                fers,
                crossing(cfg.efficiencies, fers, cfg.target_fer),
                asymptotic_efficiency(qber, cfg.de_rates, thresholds),
                float(np.mean(leaks)),
                float(np.mean(messages)),
                float(np.mean([r[1] for r in results])),
            )
        )

    # (d) Pełny postprocessing na torze GYS.
    link = Link(
        WCPSource((cfg.mu,)),
        FiberChannel(cfg.length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, dark_count_per_detector(GYS.background)),
    )
    key = run_sifted(
        link, cfg.pulses, RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, 1]))
    )
    base = PostprocessingConfig(ldpc_frame=cfg.frame)
    balances: dict[str, KeyBalance] = {}
    message_counts: dict[str, int] = {}
    for method in ("cascade", "ldpc"):
        processed = postprocess(
            key, np.random.default_rng([cfg.seed, 2]), replace(base, reconciliation=method)
        )
        balances[method] = processed.balance
        message_counts[method] = len(processed.channel.messages(f"{method}_"))

    i36 = cfg.de_rates.index(0.5)
    lines = [
        "## Ewolucja gęstości: progi BP kodów o d_v = 3",
        "",
        "| R | Stopnie kontroli (ułamki krawędzi) | Próg BP Q* | f = (1 − R)/h(Q*) |",
        "| --- | --- | --- | --- |",
    ]
    for rate, threshold in zip(cfg.de_rates, thresholds, strict=True):
        degrees = ", ".join(f"{d}: {comma(w, 2)}" for d, w in check_degrees(rate).items())
        lines.append(
            f"| {comma(rate, 2)} | {degrees} | {comma(threshold, 4)} | "
            f"{comma((1 - rate) / h2(threshold), 3)} |"
        )
    lines += [
        "",
        f"Kod (3,6): próg {comma(thresholds[i36], 4)} wobec {comma(LITERATURE_36, 3)} "
        "u Richardsona i Urbankego (2001); siatka LLR co 0,05 do ±25, bisekcja do 2·10⁻⁴. "
        "Kody regularne tracą do pojemności tym więcej, im wyższa sprawność: przy R = 0,9 f ≈ "
        f"{comma((1 - cfg.de_rates[-1]) / h2(thresholds[-1]), 2)} nawet dla nieskończonej ramki.",
        "",
        "## Wodospad kodu (3,6): p-stwo błędu ramki",
        "",
        "| Q | " + " | ".join(f"n = {n}" for n in cfg.waterfall_lengths) + " |",
        "| --- | " + " | ".join("---" for _ in cfg.waterfall_lengths) + " |",
    ]
    for k, q in enumerate(cfg.waterfall_qbers):
        lines.append(
            f"| {comma(100 * q, 1)}% | "
            + " | ".join(comma(waterfall[n][k], 3) for n in cfg.waterfall_lengths)
            + " |"
        )
    lines += [
        "",
        "Liczba prób: "
        + ", ".join(
            f"{t} dla n = {n}"
            for n, t in zip(cfg.waterfall_lengths, cfg.waterfall_trials, strict=True)
        )
        + f"; do {cfg.max_iterations} iteracji BP. Z długością ramki krzywa ostrzeje i zbliża się "
        "do progu z ewolucji gęstości.",
        "",
        f"## Reżim QKD: ramka {cfg.frame} bitów, {cfg.trials} prób na punkt",
        "",
        "| Q | "
        + " | ".join(f"FER przy f = {comma(f, 2)}" for f in cfg.efficiencies)
        + f" | LDPC: f przy FER = {comma(100 * cfg.target_fer, 0)}% | granica DE | Cascade: f | "
        "Cascade: wiadomości | LDPC: wiadomości na ramkę |",
        "| --- | "
        + " | ".join("---" for _ in cfg.efficiencies)
        + " | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {comma(100 * r.qber, 0)}% | "
            + " | ".join(comma(fer, 2) for fer in r.fer)
            + f" | {comma(r.ldpc_efficiency, 2)} | {comma(r.asymptotic_efficiency, 2)} | "
            f"{comma(r.cascade_efficiency, 3)} | {comma(r.cascade_messages, 0)} | 1 |"
        )
    lines += [
        "",
        "LDPC wysyła jeden syndrom na ramkę w jedną stronę (plus ziarno macierzy raz na sesję); "
        "Cascade prowadzi setki wymian parzystości. Za to kody regularne potrzebują większego f: "
        "ramka skończona dokłada swoje do granicy z ewolucji gęstości. Praktyka używa kodów "
        "nieregularnych i adaptacji sprawności (przebijanie, skracanie), co daje f ≈ 1,1 "
        "(Elkouss i in. 2009).",
        "",
        f"## Pełny postprocessing: GYS {cfg.length_km:g} km, μ = {comma(cfg.mu, 2)}, "
        f"{cfg.pulses:.0e} impulsów".replace("e+0", "·10^"),
        "",
        "| Uzgadnianie | Klucz po próbce | QBER próbki | Wyciek EC | f | Wiadomości EC | "
        "Odrzucone bity | Błędy resztkowe | Weryfikacja | Klucz końcowy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method, b in balances.items():
        lines.append(
            f"| {method} | {b.reconciled + b.discarded} | {comma(100 * b.qber_sample, 2)}% | "
            f"{b.leak_ec} | {comma(b.efficiency, 3)} | {message_counts[method]} | {b.discarded} | "
            f"{b.residual_errors} | {'tak' if b.verified else 'nie'} | {b.final_length} |"
        )

    fig, (ax_w, ax_f, ax_e) = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for n in cfg.waterfall_lengths:
        ax_w.semilogy(
            [100 * q for q in cfg.waterfall_qbers],
            [max(f, 1e-3) for f in waterfall[n]],
            "o-",
            label=f"n = {n}",
        )
    ax_w.axvline(100 * thresholds[i36], color="k", ls="--", label="próg BP (DE)")
    ax_w.set(
        xlabel="Q [%]",
        ylabel="FER (0 rysowane jako 10⁻³)",
        title="Wodospad kodu (3,6)",
        ylim=(5e-4, 1.5),
    )
    ax_w.legend(fontsize=8)
    ax_w.grid(True, which="both", alpha=0.3)

    for r in rows:
        ax_f.plot(cfg.efficiencies, r.fer, "o-", label=f"Q = {comma(100 * r.qber, 0)}%")
    ax_f.axhline(cfg.target_fer, color="k", ls=":", lw=1)
    ax_f.set(
        xlabel="zakładana sprawność f",
        ylabel="FER",
        title=f"LDPC d_v = 3, ramka {cfg.frame}",
    )
    ax_f.legend(fontsize=8)
    ax_f.grid(True, alpha=0.3)

    qs = [100 * r.qber for r in rows]
    ax_e.plot(
        qs,
        [r.ldpc_efficiency for r in rows],
        "o-",
        label=f"LDPC, FER = {comma(100 * cfg.target_fer, 0)}%",
    )
    ax_e.plot(qs, [r.asymptotic_efficiency for r in rows], "s--", label="LDPC, ewolucja gęstości")
    ax_e.plot(qs, [r.cascade_efficiency for r in rows], "^-", label="Cascade (4 przebiegi)")
    ax_e.axhline(1.0, color="k", lw=1)
    ax_e.set(xlabel="Q [%]", ylabel="f = wyciek/(n h(Q))", title="Sprawność uzgadniania")
    ax_e.legend(fontsize=8)
    ax_e.grid(True, alpha=0.3)
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
                "thresholds": thresholds,
                "waterfall": {str(n): v for n, v in waterfall.items()},
                "qkd": [asdict(r) for r in rows],
                "postprocessing": {m: asdict(b) for m, b in balances.items()},
                "messages": message_counts,
            },
            indent=2,
        ),
        "utf-8",
    )
    largest = cfg.waterfall_lengths[-1]
    near = cfg.waterfall_qbers.index(0.075)
    checks = [
        abs(thresholds[i36] - LITERATURE_36) < 5e-4,
        all(a > b for a, b in pairwise(thresholds)),
        waterfall[largest][near] < waterfall[cfg.waterfall_lengths[0]][near],
        all(
            f == 0.0
            for q, f in zip(cfg.waterfall_qbers, waterfall[largest], strict=True)
            if q <= 0.07
        ),
        all(r.asymptotic_efficiency < r.ldpc_efficiency < math.inf for r in rows),
        all(r.cascade_efficiency < r.ldpc_efficiency for r in rows),
        all(r.cascade_messages > 50 for r in rows),
        all(
            b.verified and b.residual_errors == 0 and b.final_length > 0 for b in balances.values()
        ),
        balances["ldpc"].final_length < balances["cascade"].final_length,
    ]
    print("sprawdzenia:", checks)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
