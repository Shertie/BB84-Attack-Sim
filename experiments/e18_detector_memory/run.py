"""E18: pamięć detektora, czyli czas martwy i afterpulsing (F1.8, F7).

Wynik: figures/e18_detector_memory.png, figures/e18_detector_memory.md
i results/e18_detector_memory.json.
(a) nasycenie: p-stwo kliknięcia na bramkę przy czasie martwym τ i afterpulsach, Monte Carlo
    na tle wzoru odnowienia c = 1/(τ + p + (1 − p)/f);
(b) tor z afterpulsami na parametrach Lima et al. 2014: Monte Carlo na tle modelu dokładnego
    (afterpuls jako niezależne ciemne zliczenie, analysis.wcp.afterpulse_dark_count) i wzorów
    Lima et al. (R = D(1 + p_ap), e = p_dc + e_mis(1 − e^(−ημ)) + p_ap D/2). Afterpuls sam może
    wywołać afterpuls, więc p na lawinę daje łącznie p/(1 − p) afterpulsów na kliknięcie;
    to jest p_ap we wzorach Lima et al.;
(c) koszt afterpulsów: o ile rośnie QBER.
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

from qkdsim.analysis.finite_key_decoy import LimChannel
from qkdsim.analysis.wcp import afterpulse_dark_count, matched_basis_clicks
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.detector import DetectorPair, stationary_click_rate, with_memory
from qkdsim.metrics import gain, matched_qber
from qkdsim.pipeline import Link, run_counts_parallel
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
LIM = LimChannel(0.0)  # parametry detektora i kanału z oceny Lima et al. 2014


@dataclass(frozen=True)
class Config:
    seed: int
    fresh_rates: tuple[float, ...]
    dead_times: tuple[int, ...]
    afterpulse: float
    memory_gates: int
    lim_lengths_km: tuple[float, ...]
    lim_mu: float
    lim_pulses: int


@dataclass(frozen=True)
class LinkPoint:
    length_km: float
    gain_mc: float
    gain_exact: float
    gain_lim: float
    qber_mc: float
    qber_exact: float
    qber_lim: float
    qber_without_afterpulse: float
    z_gain: float
    z_qber: float


def comma(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def sci(value: float) -> str:
    mantissa, exponent = f"{value:.3e}".split("e")
    return f"{mantissa.replace('.', ',')}·10^{int(exponent)}"


def total_afterpulse(per_avalanche: float) -> float:
    """Łączna liczba afterpulsów na kliknięcie, gdy afterpulsy też wywołują afterpulsy."""
    return per_avalanche / (1.0 - per_avalanche)


def lim_link(cfg: Config, length_km: float, afterpulse: float) -> Link:
    return Link(
        WCPSource((cfg.lim_mu,)),
        FiberChannel(length_km, LIM.alpha_db_per_km, LIM.misalignment),
        DetectorPair(LIM.eta_bob, LIM.dark_count, afterpulse=afterpulse),
    )


def link_point(cfg: Config, length_km: float, index: int) -> LinkPoint:
    seed = np.random.SeedSequence([cfg.seed, index])
    with_ap = run_counts_parallel(lim_link(cfg, length_km, cfg.afterpulse), cfg.lim_pulses, seed)
    without = run_counts_parallel(lim_link(cfg, length_km, 0.0), cfg.lim_pulses, seed)
    eta = 10.0 ** (-LIM.alpha_db_per_km * length_km / 10.0) * LIM.eta_bob
    dark = afterpulse_dark_count(cfg.lim_mu, eta, LIM.dark_count, LIM.misalignment, cfg.afterpulse)
    exact = matched_basis_clicks(cfg.lim_mu, eta, dark, LIM.misalignment)
    lim = LimChannel(length_km, afterpulse=total_afterpulse(cfg.afterpulse))
    detection, error = lim(cfg.lim_mu)
    g, q = gain(with_ap), matched_qber(with_ap)
    return LinkPoint(
        length_km,
        g.value,
        exact.detected,
        detection,
        q.value,
        exact.qber,
        error / detection,
        matched_qber(without).value,
        g.z_score(exact.detected),
        q.z_score(exact.qber),
    )


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    rng = np.random.default_rng(cfg.seed)
    p_total = total_afterpulse(cfg.afterpulse)
    cases = [(tau, 0.0) for tau in cfg.dead_times] + [(0, cfg.afterpulse)]
    saturation = {
        case: [
            float(with_memory(rng.random(cfg.memory_gates) < f, *case, rng).mean())
            for f in cfg.fresh_rates
        ]
        for case in cases
    }
    points = [link_point(cfg, length, i) for i, length in enumerate(cfg.lim_lengths_km)]

    lines = [
        f"## Nasycenie detektora ({cfg.memory_gates:.0e} bramek na punkt)".replace("e+0", "·10^"),
        "",
        "| Świeże kliknięcia f | "
        + " | ".join(f"τ = {tau}, p = {comma(100 * prob, 0)}% (MC / wzór)" for tau, prob in cases)
        + " |",
        "| --- | " + " | ".join("---" for _ in cases) + " |",
    ]
    worst = 0.0
    for k, f in enumerate(cfg.fresh_rates):
        cells = []
        for case in cases:
            mc, model = saturation[case][k], stationary_click_rate(f, *case)
            worst = max(worst, abs(mc - model) / math.sqrt(model / cfg.memory_gates))
            cells.append(f"{mc:.3e} / {model:.3e}".replace(".", ","))
        lines.append(f"| {f:.0e} | ".replace("e-0", "·10^-") + " | ".join(cells) + " |")
    lines += [
        "",
        f"Największe |z| w tabeli: {comma(worst, 2)} (σ z liczby kliknięć). Czas martwy "
        "τ bramek obcina p-stwo kliknięcia do f/(1 + τf); przy f·τ ≥ 1 detektor jest martwy "
        "przez większość czasu. Afterpulsy dokładają p·f(1 − f) kliknięć, czyli w pierwszym "
        "rzędzie D·p_ap, jak w modelu Lima et al.",
        "",
        f"## Afterpulsy na parametrach Lima et al. 2014 (μ = {comma(cfg.lim_mu, 1)}, "
        f"{cfg.lim_pulses:.0e} impulsów na punkt)".replace("e+0", "·10^"),
        "",
        f"Detektor ma p = {comma(100 * cfg.afterpulse, 0)}% na lawinę, czyli łącznie "
        f"p_ap = p/(1 − p) = {comma(100 * p_total, 2)}% afterpulsów na kliknięcie. „Dokładny”: "
        "afterpuls jako niezależne ciemne zliczenie z p-stwem p·c; „Lim”: ich wzory z tym p_ap.",
        "",
        "| L | Zysk: MC / dokładny / Lim | z | QBER: MC / dokładny / Lim | z | "
        "QBER bez afterpulsów |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pt in points:
        lines.append(
            f"| {pt.length_km:g} km | {sci(pt.gain_mc)} / {sci(pt.gain_exact)} / "
            f"{sci(pt.gain_lim)} | {comma(pt.z_gain, 1)} | {comma(100 * pt.qber_mc, 2)}% / "
            f"{comma(100 * pt.qber_exact, 2)}% / {comma(100 * pt.qber_lim, 2)}% | "
            f"{comma(pt.z_qber, 1)} | {comma(100 * pt.qber_without_afterpulse, 2)}% |"
        )
    near = points[0]
    lines += [
        "",
        "Wzory Lima et al. są liniowe w p_ap: pomijają to, że afterpuls ginie, gdy ten sam "
        "detektor i tak klika, i że z fotonem w drugim detektorze daje podwójne kliknięcie. Przy "
        f"L = {near.length_km:g} km (D ≈ {comma(100 * near.gain_exact, 0)}%) zawyżają QBER o "
        f"{comma(100 * (near.qber_lim - near.qber_exact), 2)} pp, na dłuższych odcinkach różnica "
        "znika. Afterpulsy podnoszą QBER o ok. "
        f"{comma(100 * p_total / (2 * (1 + p_total)), 1)} pp (p_ap/(2(1 + p_ap))), "
        "czterokrotnie więcej niż cały błąd optyki 0,5% w tym modelu.",
    ]

    fig, (ax_s, ax_g, ax_q) = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    grid = np.logspace(-4.2, -0.4, 100)
    for (case, values), color in zip(saturation.items(), ("C0", "C1", "C2"), strict=True):
        tau, prob = case
        ax_s.loglog(
            grid,
            [stationary_click_rate(f, tau, prob) for f in grid],
            color=color,
            label=f"τ = {tau}, p = {comma(100 * prob, 0)}%",
        )
        ax_s.loglog(cfg.fresh_rates, values, "o", color=color)
    ax_s.loglog(grid, grid, "k:", label="bez pamięci")
    ax_s.set(
        xlabel="świeże kliknięcia f [na bramkę]",
        ylabel="kliknięcia [na bramkę]",
        title="Czas martwy i afterpulsy (punkty: MC)",
    )
    ax_s.legend(fontsize=8)
    ax_s.grid(True, which="both", alpha=0.3)

    lengths = np.linspace(0, 120, 61)
    lim = [LimChannel(float(x), afterpulse=p_total)(cfg.lim_mu) for x in lengths]
    exact = [
        matched_basis_clicks(
            cfg.lim_mu,
            eta,
            afterpulse_dark_count(
                cfg.lim_mu, eta, LIM.dark_count, LIM.misalignment, cfg.afterpulse
            ),
            LIM.misalignment,
        )
        for eta in 10.0 ** (-LIM.alpha_db_per_km * lengths / 10.0) * LIM.eta_bob
    ]
    ax_g.semilogy(lengths, [c.detected for c in exact], label="dokładny")
    ax_g.semilogy(lengths, [d for d, _ in lim], "--", label="Lim et al.: R = D(1 + p_ap)")
    ax_g.semilogy(
        [pt.length_km for pt in points], [pt.gain_mc for pt in points], "o", label="Monte Carlo"
    )
    ax_g.set(xlabel="L [km]", ylabel="zysk", title="Zysk z afterpulsami (model Lima et al.)")
    ax_g.legend(fontsize=8)
    ax_g.grid(True, which="both", alpha=0.3)

    ax_q.plot(lengths, [100 * c.qber for c in exact], label="dokładny")
    ax_q.plot(lengths, [100 * e / d for d, e in lim], "--", label="Lim et al.")
    ax_q.plot(
        [pt.length_km for pt in points],
        [100 * pt.qber_mc for pt in points],
        "o",
        label="MC, p = 4%",
    )
    ax_q.plot(
        [pt.length_km for pt in points],
        [100 * pt.qber_without_afterpulse for pt in points],
        "s",
        label="MC bez afterpulsów",
    )
    ax_q.set(
        xlabel="L [km]", ylabel="QBER [%]", title="QBER: afterpulsy kosztują ~2 pp", ylim=(0, 6)
    )
    ax_q.legend(fontsize=8)
    ax_q.grid(True, alpha=0.3)
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
                "saturation": {f"tau={t},p={prob}": v for (t, prob), v in saturation.items()},
                "lim": [asdict(pt) for pt in points],
            },
            indent=2,
        ),
        "utf-8",
    )
    checks = [
        worst < 4.0,
        all(abs(pt.z_qber) < 4.0 and abs(pt.z_gain) < 4.0 for pt in points),
        all(pt.qber_mc > pt.qber_without_afterpulse + 0.015 for pt in points),
    ]
    print("sprawdzenia:", checks)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
