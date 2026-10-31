"""E16: QKD ze splątanymi parami: BBM92, CHSH i device-independent.

Wynik: figures/e16_entangled.png, figures/e16_entangled.md i results/e16_entangled.json.
(a) BBM92 ze źródłem SPDC pośrodku i detektorami GYS: stopa w funkcji L przy optymalnej
    jasności źródła, na tle decoy BB84 z e06; Monte Carlo zysku i QBER;
(b) CHSH i stopa device-independent w funkcji sprawności detekcji: S na wszystkich rundach
    (brak kliknięcia jako +1) i na samych koincydencjach; progi 82,8% i ~92,4%;
(c) fałszywe naruszenie CHSH przez Ewę sterującą detektorami (Gerhardt i in. 2011).
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

from qkdsim.analysis.entangled import (
    bbm92_rate,
    chsh_value,
    device_independent_rate,
    device_independent_threshold,
    key_setting_qber,
    optimal_bbm92,
    pair_statistics,
    violation_threshold,
)
from qkdsim.analysis.key_rate import GYS, rate_vacuum_weak
from qkdsim.config import load_yaml
from qkdsim.entangled import ChshCounts, DetectorControl, PairCounts, simulate_chsh, simulate_pairs
from qkdsim.rng import RandomStreams

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
PIRONIO_THRESHOLD = 0.924  # Pironio i in. 2009, podpis rys. 3 (za dokumentacją qkd)


@dataclass(frozen=True)
class Config:
    seed: int
    eta_detector: float
    dark_count: float
    misalignment: float
    f_ec: float
    alpha_db_per_km: float
    lengths_km: tuple[float, ...]
    mc_lengths_km: tuple[float, ...]
    mc_pulses: int
    efficiencies: tuple[float, ...]
    visibilities: tuple[float, ...]
    chsh_rounds: int
    fake_efficiency: float


def arm(cfg: Config, length_km: float) -> float:
    """Sprawność jednego ramienia: połowa światłowodu i detektor."""
    return float(cfg.eta_detector * 10.0 ** (-cfg.alpha_db_per_km * length_km / 20.0))


def bbm92(cfg: Config, length_km: float) -> tuple[float, float]:
    eta = arm(cfg, length_km)
    return optimal_bbm92(eta, eta, cfg.dark_count, cfg.misalignment, cfg.f_ec)


def pairs_point(cfg: Config, length_km: float, index: int) -> tuple[float, PairCounts]:
    _, lam = bbm92(cfg, length_km)
    eta = arm(cfg, length_km)
    streams = RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, index]))
    return lam, simulate_pairs(
        lam, eta, eta, cfg.dark_count, cfg.misalignment, cfg.mc_pulses, streams
    )


def chsh_point(cfg: Config, efficiency: float, index: int, eve: bool = False) -> ChshCounts:
    streams = RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, 100 + index]))
    attack = DetectorControl() if eve else None
    return simulate_chsh(efficiency, 1.0, cfg.chsh_rounds, streams, attack)


def comma(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def scientific(value: float, digits: int = 2) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa.replace('.', ',')}·10^{int(exponent)}"


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    with ProcessPoolExecutor() as pool:
        rate_futures = [pool.submit(bbm92, cfg, length) for length in cfg.lengths_km]
        pair_futures = [
            pool.submit(pairs_point, cfg, length, i) for i, length in enumerate(cfg.mc_lengths_km)
        ]
        chsh_futures = [
            pool.submit(chsh_point, cfg, eta, i) for i, eta in enumerate(cfg.efficiencies)
        ]
        honest_future = pool.submit(chsh_point, cfg, cfg.fake_efficiency, 50)
        fake_future = pool.submit(chsh_point, cfg, cfg.fake_efficiency, 51, True)
        rates = [f.result() for f in rate_futures]
        pair_points = [f.result() for f in pair_futures]
        chsh_points = [f.result() for f in chsh_futures]
        honest, fake = honest_future.result(), fake_future.result()

    positive = [length for length, (r, _) in zip(cfg.lengths_km, rates, strict=True) if r > 0]
    low, high = max(positive), min(length for length in cfg.lengths_km if length > max(positive))
    for _ in range(20):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if bbm92(cfg, middle)[0] > 0 else (low, middle)
    bbm92_range = 0.5 * (low + high)
    grid = np.linspace(0.0, 160.0, 321)
    decoy = np.array([rate_vacuum_weak(GYS, float(x), 0.48, 0.05) for x in grid])
    decoy_range = float(np.max(grid[decoy > 0]))
    thresholds = {
        v: (violation_threshold(v), device_independent_threshold(v)) for v in cfg.visibilities
    }

    lines = [
        "## BBM92 ze źródłem SPDC pośrodku (detektory GYS po obu stronach)",
        "",
        f"Zasięg BBM92: {comma(bbm92_range, 1)} km; decoy BB84 (vacuum+weak, e06): "
        f"{comma(decoy_range, 1)} km. Przypadkowa koincydencja wymaga ciemnych zliczeń po obu "
        "stronach naraz, więc źródło w środku sięga dużo dalej niż decoy BB84; ceną jest niska "
        "stopa, bo wielokrotne pary ograniczają jasność źródła (λ ≈ 0,04 pary na impuls).",
        "",
        "| L | λ optymalne | Zysk (MC / model) | QBER (MC / model) |",
        "| --- | --- | --- | --- |",
    ]
    z_pairs = []
    for length, (lam, pairs) in zip(cfg.mc_lengths_km, pair_points, strict=True):
        eta = arm(cfg, length)
        model = pair_statistics(lam, eta, eta, cfg.dark_count, cfg.misalignment)
        expected = model.gain * pairs.pulses
        z_pairs.append((pairs.coincidences - expected) / math.sqrt(expected))
        wrong = model.qber * pairs.coincidences
        z_pairs.append((pairs.errors - wrong) / math.sqrt(wrong * (1 - model.qber)))
        lines.append(
            f"| {length:g} km | {scientific(lam)} | "
            f"{scientific(pairs.coincidences / pairs.pulses)} / {scientific(model.gain)} | "
            f"{comma(100 * pairs.errors / pairs.coincidences, 2)}% / "
            f"{comma(100 * model.qber, 2)}% |"
        )
    lines += [
        "",
        "## CHSH i device-independent (stan maksymalnie splątany)",
        "",
        "| V | S > 2 od η = | Stopa DI > 0 od η = |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {comma(v, 2)} | {comma(100 * a, 2)}% | {comma(100 * b, 2)}% |"
        for v, (a, b) in thresholds.items()
    ]
    lines += [
        "",
        f"Próg dla S > 2 to 2/(1 + √2) = 82,84%. Próg dla klucza device-independent "
        f"(Acín i in. 2007) wychodzi {comma(100 * thresholds[1.0][1], 2)}%, wobec "
        f"{comma(100 * PIRONIO_THRESHOLD, 1)}% w Pironio i in. 2009 (za dokumentacją `qkd`). "
        "",
        "| η | S, wszystkie rundy (MC / model) | S, koincydencje (MC) | QBER klucza (MC / model) |",
        "| --- | --- | --- | --- |",
    ]
    for eta, counts in zip(cfg.efficiencies, chsh_points, strict=True):
        lines.append(
            f"| {comma(eta, 2)} | {comma(counts.chsh(), 3)} / {comma(chsh_value(eta), 3)} | "
            f"{comma(counts.chsh(True), 3)} | {comma(100 * counts.key_qber(), 2)}% / "
            f"{comma(100 * key_setting_qber(eta), 2)}% |"
        )
    lines += [
        "",
        f"## Fałszywe naruszenie CHSH (η = {comma(cfg.fake_efficiency, 2)})",
        "",
        "| Źródło | S, koincydencje | S, wszystkie rundy |",
        "| --- | --- | --- |",
        f"| uczciwe pary | {comma(honest.chsh(True), 3)} | {comma(honest.chsh(), 3)} |",
        f"| Ewa steruje detektorami | {comma(fake.chsh(True), 3)} | {comma(fake.chsh(), 3)} |",
        "",
        "Na samych koincydencjach Ewa sterująca detektorami (Gerhardt i in. 2011) daje S = 2√2 "
        "jak uczciwe pary, choć jest lokalnym modelem. Po odrzuceniu rund bez kliknięcia test CHSH "
        "zakłada fair sampling i przestaje cokolwiek dowodzić. Licząc wszystkie rundy, Ewa ma "
        "S ≤ 2, ale przy tej sprawności uczciwe pary też, więc device-independent nie zadziała.",
    ]

    fig, (ax_r, ax_s, ax_f) = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    values = np.array([r for r, _ in rates])
    ax_r.semilogy(
        cfg.lengths_km, np.where(values > 0, values, np.nan), "o-", label="BBM92, źródło pośrodku"
    )
    ax_r.semilogy(grid, np.where(decoy > 0, decoy, np.nan), label="decoy BB84 (vacuum+weak, e06)")
    ax_r.set(
        xlabel="odległość Alice–Bob L [km]",
        ylabel="stopa klucza [bit na impuls]",
        title="BBM92 z SPDC a decoy BB84 (detektory GYS)",
        ylim=(1e-12, 1e-2),
    )
    ax_r.legend(fontsize=8)
    ax_r.grid(True, which="both", alpha=0.3)

    etas = np.linspace(0.5, 1.0, 201)
    for v, style in zip(cfg.visibilities, ("-", "--"), strict=True):
        ax_s.plot(
            etas, [chsh_value(e, v) for e in etas], "C0" + style, label=f"S, V = {comma(v, 2)}"
        )
        dirate = [
            max(device_independent_rate(key_setting_qber(e, v), chsh_value(e, v)), 0.0)
            for e in etas
        ]
        ax_s.plot(
            etas, 2.0 + np.array(dirate), "C3" + style, label=f"2 + stopa DI, V = {comma(v, 2)}"
        )
    ax_s.plot(
        cfg.efficiencies, [c.chsh() for c in chsh_points], "o", color="C0", label="Monte Carlo"
    )
    ax_s.axhline(2.0, color="k", linewidth=0.8)
    ax_s.axvline(PIRONIO_THRESHOLD, color="C3", linestyle=":", label="92,4% (Pironio i in.)")
    ax_s.set(
        xlabel="sprawność detekcji η",
        ylabel="S  /  2 + r_DI",
        title="CHSH bez fair sampling i klucz device-independent",
        ylim=(1.6, 3.05),
    )
    ax_s.legend(fontsize=7)
    ax_s.grid(True, alpha=0.3)

    labels = ["koincydencje", "wszystkie rundy"]
    x = np.arange(2)
    ax_f.bar(x - 0.2, [honest.chsh(True), honest.chsh()], 0.4, color="C2", label="uczciwe pary")
    ax_f.bar(
        x + 0.2, [fake.chsh(True), fake.chsh()], 0.4, color="C3", label="Ewa steruje detektorami"
    )
    ax_f.axhline(2.0, color="k", linewidth=0.8, label="granica lokalna S = 2")
    ax_f.axhline(2 * math.sqrt(2), color="0.5", linestyle=":", label="2√2")
    ax_f.set_xticks(x, labels)
    ax_f.set(
        ylabel="S",
        title=f"Fałszywe naruszenie (η = {comma(cfg.fake_efficiency, 2)})",
        ylim=(0, 3.2),
    )
    ax_f.legend(fontsize=8)
    ax_f.grid(True, axis="y", alpha=0.3)
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
                "bbm92_range_km": bbm92_range,
                "decoy_range_km": decoy_range,
                "thresholds": {str(v): t for v, t in thresholds.items()},
                "chsh": [
                    {"eta": e, "s": c.chsh(), "s_post": c.chsh(True)}
                    for e, c in zip(cfg.efficiencies, chsh_points, strict=True)
                ],
                "fake": {"s_post": fake.chsh(True), "s_all": fake.chsh()},
            },
            indent=2,
        ),
        "utf-8",
    )
    tolerance = 4.0 * 4.0 * math.sqrt(6.0 / cfg.chsh_rounds)
    checks = [
        max(abs(z) for z in z_pairs) < 4.0,
        all(
            abs(c.chsh() - chsh_value(e)) < tolerance
            for e, c in zip(cfg.efficiencies, chsh_points, strict=True)
        ),
        abs(thresholds[1.0][1] - PIRONIO_THRESHOLD) < 2e-3,
        abs(fake.chsh(True) - 2 * math.sqrt(2)) < 0.05 and fake.chsh() < 2.0,
        bbm92_rate(pair_statistics(1e-3, 0.045, 0.045, cfg.dark_count, cfg.misalignment), cfg.f_ec)
        > 0,
    ]
    print("sprawdzenia:", checks)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
