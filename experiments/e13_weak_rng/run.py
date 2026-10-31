"""E13: atak 8, słaby generator Alice (F6.8–F6.10).

Wynik: figures/e13_weak_rng.png, figures/e13_weak_rng.md i results/e13_weak_rng.json.
(a) intercept-resend z bazą przewidzianą z obciążonego generatora: QBER i wiedza Ewy w funkcji
    obciążenia baz, do znanego ziarna (p = 1) włącznie;
(b) deficyt entropii klucza przy obciążonych bitach: Shannon (asymptotycznie) i min-entropia;
(c) korelacje Markowa w bitach Alice przed i po siftingu: straty je rozrzedzają;
(d) moc testów losowości w funkcji długości ciągu; MT19937 o znanym ziarnie je przechodzi.
"""

import dataclasses
import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qkdsim.analysis.intercept_resend import intercept_resend_model
from qkdsim.analysis.key_rate import GYS, rate_vacuum_weak
from qkdsim.analysis.wcp import gain, matched_basis_clicks
from qkdsim.analysis.weak_rng import (
    autocorrelation_power,
    min_entropy_deficit,
    monobit_power,
    runs_power,
    shannon_deficit,
    sifted_stay,
)
from qkdsim.attacks.intercept_resend import InterceptResend
from qkdsim.attacks.weak_rng import (
    BiasedSource,
    MajorityPredictor,
    MarkovSource,
    PassivePrediction,
    ReplayPredictor,
    known_seed_source,
)
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.defenses.randomness_tests import TESTS
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.eve import ChannelAttack
from qkdsim.metrics import eve_known_fraction, matched_qber
from qkdsim.metrics import gain as measured_gain
from qkdsim.pipeline import Counts, Link, run_counts, run_sifted
from qkdsim.rng import RandomSource, RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
P_D = dark_count_per_detector(GYS.background)
POWER = {"monobit": monobit_power, "runs": runs_power, "autokorelacja": autocorrelation_power}


@dataclass(frozen=True)
class Config:
    seed: int
    length_km: float
    mu: float
    nu: float
    pulses: int
    basis_bias: tuple[float, ...]
    markov_stays: tuple[float, ...]
    markov_pulses: int
    test_lengths: tuple[int, ...]
    test_runs: int
    test_bias: float
    test_stay: float
    known_seed_runs: int
    known_seed_length: int
    alpha: float


def link(cfg: Config, eve: ChannelAttack | None) -> Link:
    return Link(
        WCPSource((cfg.mu,)),
        FiberChannel(cfg.length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
        eve=eve,
    )


def streams(cfg: Config, alice: RandomSource, index: int) -> RandomStreams:
    base = RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, index]))
    return dataclasses.replace(base, alice=alice)


def rejection_rates(source: RandomSource, length: int, runs: int, alpha: float) -> dict[str, float]:
    rejected = dict.fromkeys(TESTS, 0)
    for _ in range(runs):
        bits = source.bits(length)
        for name, test in TESTS.items():
            rejected[name] += test(bits) < alpha
    return {name: count / runs for name, count in rejected.items()}


def comma(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def biased(cfg: Config, i: int) -> RandomSource:
    return BiasedSource(np.random.default_rng([cfg.seed, 300 + i]), p_one=cfg.test_bias)


def markov(cfg: Config, i: int) -> RandomSource:
    return MarkovSource(np.random.default_rng([cfg.seed, 400 + i]), stay_bits=cfg.test_stay)


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    honest = matched_basis_clicks(cfg.mu, GYS.eta(cfg.length_km), P_D, GYS.misalignment)

    # (a) Obciążone bazy i znane ziarno: intercept-resend w przewidzianej bazie.
    bias_points = []
    for i, p_x in enumerate(cfg.basis_bias):
        alice = BiasedSource(np.random.default_rng([cfg.seed, i]), p_x=p_x)
        eve = InterceptResend(1.0, predictor=MajorityPredictor(basis=1))
        counts = run_counts(link(cfg, eve), cfg.pulses, streams(cfg, alice, i))
        bias_points.append((p_x, matched_qber(counts), eve_known_fraction(counts)))
    seed_ir = run_counts(
        link(cfg, InterceptResend(1.0, predictor=ReplayPredictor(known_seed_source(cfg.seed)))),
        cfg.pulses,
        streams(cfg, known_seed_source(cfg.seed), 100),
    )
    seed_passive: Counts = run_counts(
        link(cfg, PassivePrediction(ReplayPredictor(known_seed_source(cfg.seed)))),
        cfg.pulses,
        streams(cfg, known_seed_source(cfg.seed), 101),
    )

    def ir_model(right: float) -> tuple[float, float]:
        m = intercept_resend_model(GYS, cfg.length_km, cfg.mu, 1.0, 1.0, P_D, right)
        return m.clicks.qber, m.known_detected / m.clicks.detected

    # (c) Korelacje Markowa przed i po siftingu.
    markov_points = []
    for i, stay in enumerate(cfg.markov_stays):
        chain = MarkovSource(np.random.default_rng([cfg.seed, 200 + i]), stay_bits=stay)
        key = run_sifted(link(cfg, None), cfg.markov_pulses, streams(cfg, chain, 200 + i))
        equal = float(np.mean(key.alice[1:] == key.alice[:-1]))
        sigma = math.sqrt(equal * (1 - equal) / (len(key) - 1))
        markov_points.append((stay, equal, sigma, len(key) / cfg.markov_pulses))
    sifted_fraction = float(np.mean([r for *_, r in markov_points]))

    # (d) Testy losowości.
    alternatives: dict[str, tuple[Callable[[Config, int], RandomSource], dict[str, float]]] = {
        f"obciążenie p = {comma(cfg.test_bias, 2)}": (biased, {"p_one": cfg.test_bias}),
        f"Markow s = {comma(cfg.test_stay, 2)}": (markov, {"stay": cfg.test_stay}),
    }
    power_mc = {
        name: [
            rejection_rates(make(cfg, i), n, cfg.test_runs, cfg.alpha)
            for i, n in enumerate(cfg.test_lengths)
        ]
        for name, (make, _) in alternatives.items()
    }
    seed_rates = rejection_rates(
        known_seed_source(cfg.seed + 1), cfg.known_seed_length, cfg.known_seed_runs, cfg.alpha
    )

    eta = GYS.eta(cfg.length_km)
    key_per_sifted = rate_vacuum_weak(GYS, cfg.length_km, cfg.mu, cfg.nu) / (
        GYS.q_sift * gain(cfg.mu, eta, GYS.background)
    )
    lines = [
        f"## Znane ziarno (MT19937), GYS {cfg.length_km:g} km",
        "",
        "| Atak | QBER | Bity znane Ewie na pewno |",
        "| --- | --- | --- |",
        f"| bez ataku (model) | {comma(100 * honest.qber, 2)}% | 0% |",
        f"| pasywny: Ewa odtwarza bity i bazy | {comma(100 * matched_qber(seed_passive).value, 2)}%"
        f" | {comma(100 * eve_known_fraction(seed_passive).value, 2)}% |",
        f"| intercept-resend w odtworzonej bazie | "
        f"{comma(100 * matched_qber(seed_ir).value, 2)}% | "
        f"{comma(100 * eve_known_fraction(seed_ir).value, 2)}% |",
        "",
        "Zysk przy ataku pasywnym: z = "
        f"{comma(measured_gain(seed_passive).z_score(honest.detected), 2)} "
        "względem modelu bez ataku. Ewa nie musi nawet dotykać światłowodu.",
        "",
        f"Ten sam generator w {cfg.known_seed_runs} ciągach po "
        f"{thousands(cfg.known_seed_length)} bitów odrzucony na poziomie "
        f"α = {comma(cfg.alpha, 2)}: "
        + ", ".join(f"{name} {comma(100 * r, 0)}%" for name, r in seed_rates.items())
        + ". Testy statystyczne nie widzą przewidywalności.",
        "",
        "## Obciążone bazy: Ewa mierzy zawsze w częstszej bazie",
        "",
        "| p(X) | QBER (MC) | QBER (model) | Bity znane Ewie (MC) | Model |",
        "| --- | --- | --- | --- | --- |",
    ]
    for p_x, qber, known in bias_points:
        model_qber, model_known = ir_model(p_x)
        lines.append(
            f"| {comma(p_x, 2)} | {comma(100 * qber.value, 2)}% | {comma(100 * model_qber, 2)}% | "
            f"{comma(100 * known.value, 1)}% | {comma(100 * model_known, 1)}% |"
        )
    lines += [
        "",
        "## Obciążone bity: deficyt entropii klucza",
        "",
        f"Przy p = 0,55 brakuje {comma(shannon_deficit(0.55), 4)} bitu entropii na bit "
        "(asymptotycznie, Shannon), czyli "
        f"{comma(100 * shannon_deficit(0.55) / key_per_sifted, 1)}% "
        f"klucza przy {cfg.length_km:g} km ({comma(key_per_sifted, 3)} bitu klucza na bit po "
        f"siftingu). W rachunku min-entropii, istotnym dla skończonego bloku, deficyt to "
        f"{comma(min_entropy_deficit(0.55), 4)} bitu, czyli więcej niż połowa klucza.",
        "",
        f"## Korelacje Markowa w bitach Alice (sifting zostawia {comma(100 * sifted_fraction, 2)}% "
        "impulsów)",
        "",
        "| s (surowe bity) | Sąsiednie bity klucza równe (MC) | Model "
        "| Deficyt przed / po siftingu |",
        "| --- | --- | --- | --- |",
    ]
    for stay, equal, sigma, r in markov_points:
        model = sifted_stay(stay, r)
        lines.append(
            f"| {comma(stay, 3)} | {comma(equal, 4)} ± {comma(sigma, 4)} | {comma(model, 4)} | "
            f"{comma(shannon_deficit(stay), 3)} / {comma(shannon_deficit(model), 4)} |"
        )
    lines += [
        "",
        "Sifting zostawia bity w odstępach geometrycznych, więc korelacja λ = 2s − 1 spada do "
        "rλ/(1 − (1 − r)λ). Na dłuższym łączu zostaje tylko korelacja bliska pełnej "
        "powtarzalności.",
        "",
        f"## Moc testów losowości (α = {comma(cfg.alpha, 2)}, {cfg.test_runs} ciągów na punkt)",
        "",
        "| Alternatywa | n | " + " | ".join(TESTS) + " |",
        "| --- | --- | " + " | ".join("---" for _ in TESTS) + " |",
    ]
    for name, (_, kwargs) in alternatives.items():
        for n, rates in zip(cfg.test_lengths, power_mc[name], strict=True):
            cells = [
                f"{comma(100 * rates[t], 0)}% ({comma(100 * POWER[t](n, **kwargs), 0)}%)"
                for t in TESTS
            ]
            lines.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
    lines += ["", "W nawiasach: przybliżenie normalne z analysis/weak_rng.py."]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    ax_ir, ax_bias, ax_markov, ax_power = axes.ravel()
    grid = np.linspace(0.5, 1.0, 101)
    models = [ir_model(p) for p in grid]
    ax_ir.plot(grid, [100 * q for q, _ in models], color="C3", label="QBER (model)")
    ax_ir.plot(grid, [100 * k for _, k in models], color="C0", label="bity znane Ewie (model)")
    ax_ir.plot(
        [p for p, *_ in bias_points], [100 * q.value for _, q, _ in bias_points], "o", color="C3"
    )
    ax_ir.plot(
        [p for p, *_ in bias_points], [100 * k.value for *_, k in bias_points], "o", color="C0"
    )
    ax_ir.plot([1.0], [100 * matched_qber(seed_ir).value], "*", color="C3", markersize=12)
    ax_ir.plot(
        [1.0], [100 * eve_known_fraction(seed_ir).value], "*", color="C0", markersize=12,
        label="znane ziarno (MC)",
    )  # fmt: skip
    ax_ir.axhline(11.0, color="0.5", linestyle=":", label="próg QBER 11%")
    ax_ir.set(
        xlabel="p-stwo częstszej bazy u Alice",
        ylabel="[%]",
        title="Intercept-resend w przewidzianej bazie (punkty: MC)",
        ylim=(0, 105),
    )
    ax_ir.legend(fontsize=8)
    ax_ir.grid(True, alpha=0.3)

    p_grid = np.linspace(0.5, 0.95, 91)
    ax_bias.semilogy(p_grid[1:], [shannon_deficit(p) for p in p_grid[1:]], label="Shannon 1 − h(p)")
    ax_bias.semilogy(
        p_grid[1:], [min_entropy_deficit(p) for p in p_grid[1:]], label="min-entropia 1 + log₂ p"
    )
    ax_bias.axhline(key_per_sifted, color="k", linestyle="--", label="klucz na bit po siftingu")
    ax_bias.set(
        xlabel="p-stwo częstszego bitu u Alice",
        ylabel="deficyt entropii [bit na bit]",
        title=f"Obciążone bity a klucz ({cfg.length_km:g} km)",
    )
    ax_bias.legend(fontsize=8)
    ax_bias.grid(True, which="both", alpha=0.3)

    one_minus = np.logspace(-4, math.log10(0.5), 80)
    ax_markov.loglog(one_minus, [shannon_deficit(1 - x) for x in one_minus], label="surowe bity")
    ax_markov.loglog(
        one_minus,
        [shannon_deficit(sifted_stay(1 - x, sifted_fraction)) for x in one_minus],
        label=f"klucz po siftingu ({cfg.length_km:g} km)",
    )
    ax_markov.errorbar(
        [1 - s for s, *_ in markov_points],
        [shannon_deficit(e) for _, e, _, _ in markov_points],
        yerr=[abs(math.log2(e / (1 - e))) * sg for _, e, sg, _ in markov_points],
        fmt="o",
        color="C1",
        label="Monte Carlo",
    )
    ax_markov.set(
        xlabel="1 − s (s: p-stwo powtórzenia bitu)",
        ylabel="deficyt entropii [bit na bit]",
        title="Korelacje Markowa: sifting je rozrzedza",
    )
    ax_markov.legend(fontsize=8)
    ax_markov.grid(True, which="both", alpha=0.3)

    lengths = np.logspace(2, 5.3, 60).astype(int)
    for (name, (_, kwargs)), style in zip(alternatives.items(), ("-", "--"), strict=True):
        for test, color in zip(TESTS, ("C0", "C2", "C4"), strict=True):
            ax_power.semilogx(
                lengths,
                [POWER[test](int(n), **kwargs) for n in lengths],
                color=color,
                linestyle=style,
                label=f"{test}, {name}",
            )
            ax_power.semilogx(
                cfg.test_lengths, [r[test] for r in power_mc[name]], "o", color=color, markersize=4
            )
    ax_power.axhline(cfg.alpha, color="k", linestyle=":", label="znane ziarno (= α)")
    ax_power.set(
        xlabel="długość ciągu n [bit]",
        ylabel="p-stwo odrzucenia",
        title="Moc testów losowości (punkty: MC)",
    )
    ax_power.legend(fontsize=6.5, loc="center right")
    ax_power.grid(True, which="both", alpha=0.3)
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
                "basis_bias": [
                    {"p_x": p, "qber": q.value, "known": k.value} for p, q, k in bias_points
                ],
                "known_seed": {
                    "passive_known": eve_known_fraction(seed_passive).value,
                    "ir_qber": matched_qber(seed_ir).value,
                    "test_rejection": seed_rates,
                },
                "markov": [
                    {"stay": s, "sifted_equal": e, "sifted_fraction": r}
                    for s, e, _, r in markov_points
                ],
                "power": power_mc,
            },
            indent=2,
        ),
        "utf-8",
    )
    checks = [
        eve_known_fraction(seed_passive).value == 1.0,
        abs(matched_qber(seed_passive).z_score(honest.qber)) < 4.0,
        abs(matched_qber(seed_ir).z_score(ir_model(1.0)[0])) < 4.0,
        all(
            abs(q.z_score(ir_model(p)[0])) < 4.0 and abs(k.z_score(ir_model(p)[1])) < 4.0
            for p, q, k in bias_points
        ),
        all(abs(e - sifted_stay(s, r)) < 4.0 * sg for s, e, sg, r in markov_points),
        all(rate < 0.05 for rate in seed_rates.values()),
    ]
    print("sprawdzenia:", checks)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
