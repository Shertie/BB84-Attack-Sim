"""E15: MDI-QKD (Lo, Curty, Qi 2012), F7.

Wynik: figures/e15_mdi_qkd.png, figures/e15_mdi_qkd.md i results/e15_mdi_qkd.json.
(a) stopa klucza MDI-QKD w funkcji odległości Alice–Bob (przekaźnik pośrodku) na tle decoy BB84
    z tymi samymi detektorami; artykuł: ponad 40 dB, czyli ponad 200 km;
(b) zysk i QBER przekaźnika z Monte Carlo na tle modelu analitycznego;
(c) Ewa jako przekaźnik: zna cały klucz bez błędów w bazie klucza, ale baza testowa ją zdradza.
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

from qkdsim.analysis.key_rate import LinkParameters, best_over_mu, max_distance, rate_infinite_decoy
from qkdsim.analysis.mdi import coherent_gains, optimal_rate
from qkdsim.analysis.mdi import max_distance as mdi_max_distance
from qkdsim.config import load_yaml
from qkdsim.mdi import MDICounts, MDILink, MeasureAndAnnounce, Relay, run_mdi
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
PAPER_RANGE_KM = 200.0  # „more than 40dB (i.e., 200km)”


@dataclass(frozen=True)
class Config:
    seed: int
    relay_efficiency: float
    dark_count: float
    misalignment: float
    f_ec: float
    alpha_db_per_km: float
    lengths_km: tuple[float, ...]
    mc_lengths_km: tuple[float, ...]
    mc_pairs: int
    mc_mu: float
    attack_length_km: float
    attack_pairs: int


def relay(cfg: Config) -> Relay:
    return Relay.with_misalignment(cfg.relay_efficiency, cfg.dark_count, cfg.misalignment)


def bb84(cfg: Config) -> LinkParameters:
    """Decoy BB84 z tymi samymi detektorami: tło dwóch detektorów, bazy asymetryczne (q → 1)."""
    return LinkParameters(
        cfg.alpha_db_per_km,
        cfg.relay_efficiency,
        2.0 * cfg.dark_count,
        cfg.misalignment,
        cfg.f_ec,
        q_sift=1.0,
    )


def link(cfg: Config, length_km: float, mu: float, attack: bool = False) -> MDILink:
    source = WCPSource((mu,))
    return MDILink(
        source,
        source,
        length_km / 2,
        length_km / 2,
        relay(cfg),
        cfg.alpha_db_per_km,
        MeasureAndAnnounce() if attack else None,
    )


def simulate(cfg: Config, length_km: float, pairs: int, attack: bool, index: int) -> MDICounts:
    seed = np.random.SeedSequence([cfg.seed, index])
    return run_mdi(link(cfg, length_km, cfg.mc_mu, attack), pairs, RandomStreams.from_seed(seed))


def comma(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def scientific(value: float, digits: int = 2) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa.replace('.', ',')}·10^{int(exponent)}"


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    with ProcessPoolExecutor() as pool:
        mc = [
            pool.submit(simulate, cfg, length, cfg.mc_pairs, False, i)
            for i, length in enumerate(cfg.mc_lengths_km)
        ]
        honest_future = pool.submit(
            simulate, cfg, cfg.attack_length_km, cfg.attack_pairs, False, 90
        )
        attack_future = pool.submit(simulate, cfg, cfg.attack_length_km, cfg.attack_pairs, True, 91)
        rates = [
            optimal_rate(length, relay(cfg), cfg.f_ec, cfg.alpha_db_per_km)
            for length in cfg.lengths_km
        ]
        mc_counts = [f.result() for f in mc]
        honest, attacked = honest_future.result(), attack_future.result()

    mdi_range = mdi_max_distance(relay(cfg), cfg.f_ec, cfg.alpha_db_per_km)
    params = bb84(cfg)

    def bb84_rate(length: float) -> float:
        return best_over_mu(lambda mu: rate_infinite_decoy(params, length, mu))[0]

    bb84_range = max_distance(bb84_rate)
    bb84_grid = np.linspace(0.0, 220.0, 111)
    bb84_rates = [bb84_rate(float(length)) for length in bb84_grid]

    lines = [
        "## Zasięg MDI-QKD (parametry z rys. 2 Lo, Curty, Qi 2012)",
        "",
        "| Układ | Zasięg |",
        "| --- | --- |",
        f"| MDI-QKD, przekaźnik pośrodku (artykuł: ponad {comma(PAPER_RANGE_KM, 0)} km) | "
        f"{comma(mdi_range, 1)} km |",
        f"| decoy BB84, te same detektory, bazy asymetryczne | {comma(bb84_range, 1)} km |",
        "",
        f"Przekaźnik: sprawność {comma(100 * cfg.relay_efficiency, 1)}%, ciemne zliczenia "
        f"{scientific(cfg.dark_count)} na detektor (tło 6,02·10^-6 z artykułu na parę "
        "detektorów), "
        f"błąd optyki {comma(100 * cfg.misalignment, 1)}% jako dwa obroty po połowie, f = "
        f"{comma(cfg.f_ec, 2)}, {comma(cfg.alpha_db_per_km, 1)} dB/km; nieskończenie wiele stanów "
        "decoy, μ_A = μ_B optymalne w każdym punkcie. Przy tle 6,02·10^-6 na detektor zasięg "
        "spada poniżej 200 km. W tym modelu MDI sięga dalej niż decoy BB84, ale nie dwa razy: "
        "artykuł porównuje z innym układem.",
        "",
        f"## Monte Carlo przekaźnika (μ_A = μ_B = {comma(cfg.mc_mu, 2)}, "
        f"{cfg.mc_pairs:.0e} par impulsów)".replace("e+0", "·10^"),
        "",
        "| L | Q_Z (MC / model) | E_Z (MC / model) | E_X (MC / model) |",
        "| --- | --- | --- | --- |",
    ]
    z_scores = []
    for length, counts in zip(cfg.mc_lengths_km, mc_counts, strict=True):
        t = 10.0 ** (-cfg.alpha_db_per_km * length / 20.0)
        cells = []
        for basis in (0, 1):
            gain, error = coherent_gains(cfg.mc_mu * t, cfg.mc_mu * t, relay(cfg), basis)
            sent, hits = counts.sent[0, 0, basis], counts.detected[0, 0, basis]
            wrong = counts.errors[0, 0, basis]
            z_scores += [
                (hits - sent * gain) / math.sqrt(sent * gain),
                (wrong - hits * error) / math.sqrt(hits * error * (1 - error)),
            ]
            if basis == 0:
                cells.append(f"{scientific(hits / sent)} / {scientific(gain)}")
            cells.append(f"{comma(100 * wrong / hits, 2)}% / {comma(100 * error, 2)}%")
        lines.append(f"| {length:g} km | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Największe |z| w tabeli: {comma(max(abs(z) for z in z_scores), 2)}. Przy jednej "
        "intensywności QBER w bazie X wynosi ok. 25% także bez ataku: to impulsy wielofotonowe; "
        "błąd fazowy liczy się z par pojedynczych fotonów (stany decoy).",
        "",
        f"## Ewa jako przekaźnik ({cfg.attack_length_km:g} km, μ = {comma(cfg.mc_mu, 2)})",
        "",
        "| Przekaźnik | QBER w bazie klucza | QBER w bazie testowej | Bity klucza znane Ewie |",
        "| --- | --- | --- | --- |",
    ]
    summary = {}
    for name, counts in (("uczciwy", honest), ("Ewa: pomiar Z, ogłasza ψ⁻", attacked)):
        key_hits, test_hits = counts.detected[0, 0, 0], counts.detected[0, 0, 1]
        e_z, e_x = counts.errors[0, 0, 0] / key_hits, counts.errors[0, 0, 1] / test_hits
        known = counts.eve_correct[0, 0, 0] / key_hits
        summary[name] = {"e_z": e_z, "e_x": e_x, "known": known}
        lines.append(
            f"| {name} | {comma(100 * e_z, 2)}% | {comma(100 * e_x, 2)}% | "
            f"{comma(100 * known, 1)}% |"
        )
    lines += [
        "",
        "Ewa kontroluje cały przekaźnik, więc oślepianie czy time-shift nic jej nie dodają: "
        "detektory i tak są jej. Żeby znać klucz, musi mierzyć stany Alice i Boba, a pomiar "
        "w bazie Z daje w bazie X 50% błędów, wyraźnie ponad ~25% uczciwego przekaźnika przy "
        "WCP. Także dla par pojedynczych fotonów jej ogłoszenia nie zależą od bitów X, więc "
        "e¹¹_X = ½ i klucza nie ma. Dla porównania w BB84 oślepienie detektorów zostawia QBER "
        "bez zmian, a Ewa zna cały klucz (e05).",
    ]

    fig, (ax_r, ax_q, ax_a) = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    values = np.array([r for r, _ in rates])
    ax_r.semilogy(
        cfg.lengths_km,
        np.where(values > 0, values, np.nan),
        "o-",
        label="MDI-QKD, przekaźnik pośrodku",
    )
    bb = np.array(bb84_rates)
    ax_r.semilogy(bb84_grid, np.where(bb > 0, bb, np.nan), label="decoy BB84, te same detektory")
    ax_r.axvline(PAPER_RANGE_KM, color="k", linestyle=":", label="200 km (artykuł)")
    ax_r.set(
        xlabel="odległość Alice–Bob L [km]",
        ylabel="stopa klucza [bit na parę impulsów]",
        title="MDI-QKD a decoy BB84 (nieskończenie wiele decoy)",
        ylim=(1e-9, 5e-2),
    )
    ax_r.legend(fontsize=8)
    ax_r.grid(True, which="both", alpha=0.3)

    grid = np.linspace(0.0, 100.0, 51)
    model = {
        basis: [
            coherent_gains(
                cfg.mc_mu * 10 ** (-0.01 * length),
                cfg.mc_mu * 10 ** (-0.01 * length),
                relay(cfg),
                basis,
            )
            for length in grid
        ]
        for basis in (0, 1)
    }
    ax_q.plot(grid, [100 * e for _, e in model[0]], color="C0", label="E_Z (model)")
    ax_q.plot(grid, [100 * e for _, e in model[1]], color="C3", label="E_X (model)")
    for basis, color in ((0, "C0"), (1, "C3")):
        ax_q.plot(
            cfg.mc_lengths_km,
            [100 * c.errors[0, 0, basis] / c.detected[0, 0, basis] for c in mc_counts],
            "o",
            color=color,
        )
    ax_q.plot([], [], "ko", label="Monte Carlo")
    ax_q.set(
        xlabel="odległość Alice–Bob L [km]",
        ylabel="QBER [%]",
        title=f"Przekaźnik: QBER przy μ = {comma(cfg.mc_mu, 2)}",
        ylim=(0, 35),
    )
    ax_q.legend(fontsize=8)
    ax_q.grid(True, alpha=0.3)

    labels = list(summary)
    x = np.arange(3)
    for k, (name, color) in enumerate(zip(labels, ("C2", "C3"), strict=True)):
        s = summary[name]
        ax_a.bar(
            x + (k - 0.5) * 0.38,
            [100 * s["e_z"], 100 * s["e_x"], 100 * s["known"]],
            0.38,
            color=color,
            label=name,
        )
    ax_a.set_xticks(x, ["QBER, baza klucza", "QBER, baza testowa", "klucz znany Ewie"])
    ax_a.set(ylabel="[%]", title="Ewa jako przekaźnik")
    ax_a.legend(fontsize=8)
    ax_a.grid(True, axis="y", alpha=0.3)
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
                "mdi_range_km": mdi_range,
                "bb84_range_km": bb84_range,
                "rates": [
                    {"length_km": L, "rate": r, "mu": mu}
                    for L, (r, mu) in zip(cfg.lengths_km, rates, strict=True)
                ],
                "attack": summary,
            },
            indent=2,
        ),
        "utf-8",
    )
    attack = summary["Ewa: pomiar Z, ogłasza ψ⁻"]
    checks = [
        mdi_range > PAPER_RANGE_KM,
        mdi_range > bb84_range,
        max(abs(z) for z in z_scores) < 4.0,
        attack["e_z"] == 0.0 and attack["known"] == 1.0,
        abs(attack["e_x"] - 0.5) < 0.02,
    ]
    print("sprawdzenia:", checks)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
