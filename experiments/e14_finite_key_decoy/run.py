"""E14: skończony klucz dla WCP ze stanami decoy (Lim et al. 2014).

Wynik: figures/e14_finite_key_decoy.png, figures/e14_finite_key_decoy.md
i results/e14_finite_key_decoy.json.
(a) odtworzenie rys. 1 artykułu: optymalna stopa R = ℓ/N w funkcji L dla n_X = 10⁴…10⁹
    i granica asymptotyczna z trzema intensywnościami; liczby z tekstu: 135 km przy n_X = 10⁴
    i stosunek ≈ 1,75 stóp dla 10⁹ i 10⁷ przy 100 km;
(b) to samo dla łącza GYS z dokładnym modelem detektorów symulatora, na tle asymptotycznego
    decoy vacuum+weak z e06, oraz rachunek Lima na licznikach z Monte Carlo.
"""

import dataclasses
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qkdsim.analysis.finite_key_decoy import (
    LIM_EPS_COR,
    LIM_EPS_SEC,
    DetectionModel,
    LimChannel,
    SimulatorChannel,
    expected_counts,
    optimal_rate,
)
from qkdsim.analysis.key_rate import GYS, rate_vacuum_weak
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link, run_counts
from qkdsim.postprocessing.finite_key_decoy import counts_from_simulation, key_length_decoy
from qkdsim.rng import BiasedSource, RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
P_D = dark_count_per_detector(GYS.background)
# Lim et al. 2014, tekst pod rys. 1.
PAPER = {"max_distance_1e4_km": 135.0, "ratio_100km": 1.75}


@dataclass(frozen=True)
class Config:
    seed: int
    lim_lengths_km: tuple[float, ...]
    lim_blocks: tuple[float, ...]
    gys_lengths_km: tuple[float, ...]
    gys_blocks: tuple[float, ...]
    gys_f_ec: float
    mc_lengths_km: tuple[float, ...]
    mc_block: float


def channel(kind: str, length_km: float) -> DetectionModel:
    if kind == "lim":
        return LimChannel(length_km)
    return SimulatorChannel(GYS.eta(length_km), P_D, GYS.misalignment)


def f_ec(kind: str, cfg: Config) -> float:
    return 1.16 if kind == "lim" else cfg.gys_f_ec


def curve(
    kind: str, block: float | None, lengths: tuple[float, ...], cfg: Config
) -> tuple[list[float], float]:
    """Optymalne stopy na siatce L i zasięg (bisekcja); `block` = None: bez poprawek."""
    finite = block is not None
    n = block if block is not None else 1e6

    def rate(length_km: float) -> float:
        return optimal_rate(channel(kind, length_km), n, f_ec(kind, cfg), finite=finite)[0]

    rates = [rate(length) for length in lengths]
    positive = [length for length, r in zip(lengths, rates, strict=True) if r > 0.0]
    if not positive:
        return rates, 0.0
    low = max(positive)
    high = min((length for length in lengths if length > low), default=low + 40.0)
    for _ in range(7):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if rate(middle) > 0.0 else (low, middle)
    return rates, 0.5 * (low + high)


@dataclass(frozen=True)
class MonteCarloPoint:
    length_km: float
    pulses: int
    n_key: float
    model_length: float
    mc_length: float


def monte_carlo(cfg: Config, length_km: float, index: int) -> MonteCarloPoint:
    model = channel("gys", length_km)
    _, choice = optimal_rate(model, cfg.mc_block, cfg.gys_f_ec)
    assert choice is not None
    key, test, pulses = expected_counts(choice, model, cfg.mc_block)
    link = Link(
        WCPSource(choice.settings.intensities, choice.settings.probabilities),
        FiberChannel(length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
    )
    p_x = 1.0 - choice.key_basis  # baza klucza to Z (0)
    streams = dataclasses.replace(
        RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, index])),
        alice=BiasedSource(np.random.default_rng([cfg.seed, index, 1]), p_x=p_x),
        bob=BiasedSource(np.random.default_rng([cfg.seed, index, 2]), p_x=p_x),
    )
    mc_key, mc_test = counts_from_simulation(run_counts(link, round(pulses), streams))
    args = (choice.settings, cfg.gys_f_ec, LIM_EPS_SEC, LIM_EPS_COR)
    return MonteCarloPoint(
        length_km,
        round(pulses),
        mc_key.total,
        key_length_decoy(key, test, *args).value,
        key_length_decoy(mc_key, mc_test, *args).value,
    )


def comma(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def power(block: float) -> str:
    return f"10^{round(np.log10(block))}"


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    jobs = [("lim", b, cfg.lim_lengths_km) for b in (*cfg.lim_blocks, None)]
    jobs += [("gys", b, cfg.gys_lengths_km) for b in (*cfg.gys_blocks, None)]
    with ProcessPoolExecutor() as pool:
        curve_futures = [pool.submit(curve, kind, b, lengths, cfg) for kind, b, lengths in jobs]
        mc_futures = [
            pool.submit(monte_carlo, cfg, length, i) for i, length in enumerate(cfg.mc_lengths_km)
        ]
        ratio_futures = [pool.submit(optimal_rate, LimChannel(100.0), b) for b in (1e9, 1e7)]
        curves = {
            (kind, b): f.result() for (kind, b, _), f in zip(jobs, curve_futures, strict=True)
        }
        points = [f.result() for f in mc_futures]
        ratio = ratio_futures[0].result()[0] / ratio_futures[1].result()[0]

    lim_max = curves[("lim", 1e4)][1]
    lines = [
        "## Odtworzenie Lim et al. 2014 (rys. 1 i tekst)",
        "",
        "| Wielkość | Artykuł | Model |",
        "| --- | --- | --- |",
        f"| zasięg przy n_X = 10⁴ | {comma(PAPER['max_distance_1e4_km'], 0)} km | "
        f"{comma(lim_max, 1)} km |",
        f"| R(n_X = 10⁹) / R(n_X = 10⁷) przy 100 km | ≈ {comma(PAPER['ratio_100km'], 2)} | "
        f"{comma(ratio, 3)} |",
        "",
        "Parametry z artykułu: η_Bob = 10%, p_dc = 6·10⁻⁷, p_ap = 4%, e_mis = 0,5%, f = 1,16, "
        "α = 0,2 dB/km, μ₃ = 2·10⁻⁴, ε_cor = 10⁻¹⁵, optymalizacja po (q, p₁, p₂, μ₁, μ₂). "
        "Liczby z tekstu odtwarza ε_sec = 10⁻¹⁰ (zamiast κℓ z arXiv v1, które daje ok. 128 km) "
        "i η_sys zamiast wydrukowanego η_ch w modelu błędu.",
        "",
        "| n_X | Zasięg (Lim) | Zasięg (GYS, symulator) |",
        "| --- | --- | --- |",
    ]
    for block in sorted(set(cfg.lim_blocks) | set(cfg.gys_blocks)):
        lim = curves.get(("lim", block))
        gys = curves.get(("gys", block))
        lim_cell = f"{comma(lim[1], 1)} km" if lim else "—"
        gys_cell = f"{comma(gys[1], 1)} km" if gys else "—"
        lines.append(f"| {power(block)} | {lim_cell} | {gys_cell} |")
    vacuum_weak = [rate_vacuum_weak(GYS, length, 0.48, 0.05) for length in np.linspace(0, 160, 321)]
    vw_max = float(np.max(np.linspace(0, 160, 321)[np.array(vacuum_weak) > 0]))
    shortfall = curves[("gys", None)][1] - curves[("gys", min(cfg.gys_blocks))][1]
    lines += [
        f"| ∞ (trzy intensywności, bez poprawek) | {comma(curves[('lim', None)][1], 1)} km | "
        f"{comma(curves[('gys', None)][1], 1)} km |",
        "",
        f"Dla porównania asymptotyczny decoy vacuum+weak z e06 (bazy 50/50, μ = 0,48, ν = 0,05): "
        f"{comma(vw_max, 1)} km. Asymetryczny wybór baz i trzy intensywności dają w granicy "
        "asymptotycznej więcej, a skończony blok zabiera część zasięgu: przy "
        f"n_X = {power(min(cfg.gys_blocks))} zasięg GYS jest krótszy o {comma(shortfall, 0)} km.",
        "",
        f"## Rachunek Lima na licznikach z Monte Carlo (GYS, n_X ≈ {power(cfg.mc_block)})",
        "",
        "| L | Impulsy N | n_X (MC) | ℓ z modelu | ℓ z MC | Różnica |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for p in points:
        mantissa, exponent = f"{p.pulses:.2e}".split("e")
        lines.append(
            f"| {p.length_km:g} km | {mantissa.replace('.', ',')}·10^{int(exponent)} | "
            f"{p.n_key:.0f} | {p.model_length:.0f} | {p.mc_length:.0f} | "
            f"{comma(100 * (p.mc_length / p.model_length - 1), 1)}% |"
        )
    lines += [
        "",
        "Różnice są rzędu rozrzutu statystycznego: przy n_X = 10⁵ błędy w bazie testowej dla μ₂ "
        "to kilkaset "
        "zdarzeń, więc granica błędu fazowego waha się o kilka procent, a ℓ razem z nią.",
    ]
    lines += [
        "",
        "Zastrzeżenie: dokumentacja `qkd` 0.2.0 powołuje się na Tupkary et al. "
        "(arXiv:2502.10340), według których oszacowanie błędu fazowego (5) nie jest ścisłą "
        "granicą. Nie weryfikowaliśmy tego.",
    ]

    fig, (ax_lim, ax_gys) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(cfg.lim_blocks)))
    for ax, kind, blocks, lengths, title in (
        (ax_lim, "lim", cfg.lim_blocks, cfg.lim_lengths_km, "Odtworzenie rys. 1 Lim et al. 2014"),
        (ax_gys, "gys", cfg.gys_blocks, cfg.gys_lengths_km, "Łącze GYS (model symulatora)"),
    ):
        palette = colors if kind == "lim" else colors[:: max(1, len(colors) // len(blocks))]
        for block, color in zip(blocks, palette, strict=False):
            rates, _ = curves[(kind, block)]
            values = np.array(rates)
            ax.semilogy(
                lengths, np.where(values > 0, values, np.nan), "o-", color=color, markersize=3,
                label=f"n_X = {power(block)}",
            )  # fmt: skip
        rates, _ = curves[(kind, None)]
        values = np.array(rates)
        ax.semilogy(
            lengths, np.where(values > 0, values, np.nan), "k--", label="bez poprawek (n_X → ∞)"
        )
        ax.set(xlabel="długość światłowodu L [km]", ylabel="R = ℓ/N [bit na impuls]", title=title)
        ax.grid(True, which="both", alpha=0.3)
    ax_lim.axvline(PAPER["max_distance_1e4_km"], color="C3", linestyle=":", label="135 km (tekst)")
    ax_lim.set_ylim(1e-7, 3e-2)
    grid = np.linspace(0, 160, 321)
    vw = np.array(vacuum_weak)
    ax_gys.semilogy(grid, np.where(vw > 0, vw, np.nan), color="C1", label="vacuum+weak, e06")
    ax_gys.semilogy(
        [p.length_km for p in points],
        [p.mc_length / p.pulses for p in points],
        "*",
        color="C3",
        markersize=11,
        label=f"Monte Carlo, n_X ≈ {power(cfg.mc_block)}",
    )
    ax_gys.set_ylim(1e-7, 3e-2)
    for ax in (ax_lim, ax_gys):
        ax.legend(fontsize=7)
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
                "ratio_100km": ratio,
                "max_distance_km": {
                    f"{kind}:{'inf' if b is None else power(b)}": d
                    for (kind, b), (_, d) in curves.items()
                },
                "monte_carlo": [asdict(p) for p in points],
            },
            indent=2,
        ),
        "utf-8",
    )
    lim_distances = [curves[("lim", b)][1] for b in cfg.lim_blocks]
    checks = [
        abs(lim_max - PAPER["max_distance_1e4_km"]) <= 5.0,
        abs(ratio - PAPER["ratio_100km"]) <= 0.05,
        lim_distances == sorted(lim_distances),
        all(abs(p.mc_length / p.model_length - 1) < 0.1 for p in points),
    ]
    print("sprawdzenia:", checks)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
