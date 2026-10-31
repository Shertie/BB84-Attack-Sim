"""E12: atak 5, Trojan horse.

Wynik: figures/e12_trojan_horse.png, figures/e12_trojan_horse.md i results/e12_trojan_horse.json.
(a) informacja Ewy o bicie w funkcji μ_out, ze stałą i losową fazą: wzory Gisina et al. 2006,
    Monte Carlo w łączu GYS i punkty z artykułu;
(b) przewaga Ewy w odgadywaniu bazy (Helstrom dla stanów mieszanych) i próg, od którego
    intercept-resend z tą wiedzą mieści się pod progiem QBER;
(c) wymagana izolacja w funkcji dopuszczalnego wycieku dla odbić z Gisina et al. 2006;
(d) budżet izolacji z Tabeli I Lucamariniego et al. 2015 (sonda ograniczona progiem LIDT).
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
from scipy.optimize import brentq

from qkdsim.analysis.entropy import h2
from qkdsim.analysis.key_rate import GYS, rate_vacuum_weak
from qkdsim.analysis.trojan_horse import (
    GISIN_REFLECTION_DB,
    basis_guess_probability,
    bit_guess_probability,
    bit_information,
    lidt_probe_photons,
    required_isolation_db,
    round_trip_attenuation_db,
    tolerable_photons,
)
from qkdsim.analysis.wcp import gain, matched_basis_clicks
from qkdsim.attacks.trojan_horse import TrojanHorse
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.metrics import Estimate, eve_guess_accuracy, eve_known_fraction, matched_qber
from qkdsim.metrics import gain as measured_gain
from qkdsim.pipeline import Link, run_counts_parallel
from qkdsim.source import WCPSource

# Lucamarini et al. 2015, Tabela I: (f_A [Hz], |γ̇|, |Ṙ|, |Ȧ|, |İ|, n) w dB, filtr Ḟ = 0.
LUCAMARINI_TABLE_I = (
    (1e9, 170, 40, 35, 60, 1),
    (1e9, 170, 50, 0, 60, 2),
    (1e6, 200, 40, 30, 50, 2),
    (1e6, 200, 50, 0, 50, 3),
    (1e3, 230, 40, 35, 60, 2),
    (1e3, 230, 50, 0, 60, 3),
)
LUCAMARINI_TARGET = 1e-6  # μ_out z artykułu
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
P_D = dark_count_per_detector(GYS.background)
REFERENCE_PROBE = 1e9  # μ_in w punktach Monte Carlo; tłumienie dobrane do μ_out
# Gisin et al. 2006, podpis rys. 7: μ_out = 0,1 daje 0,135 i 0,095 bitu.
GISIN = {"mu": 0.1, "coherent": 0.135, "random": 0.095}


@dataclass(frozen=True)
class Config:
    seed: int
    length_km: float
    mu: float
    nu: float
    mc_pulses: int
    mc_returned: tuple[float, ...]
    probe_photons: tuple[float, ...]
    max_information: tuple[float, ...]
    attenuator_db: float
    qber_threshold: float


@dataclass(frozen=True)
class Point:
    returned: float
    random_phase: bool
    information: float
    sigma: float
    model: float
    qber_z: float
    gain_z: float


def information_estimate(estimate: Estimate, random_phase: bool) -> tuple[float, float]:
    """Informacja Ewy i jej niepewność z trafności (stała faza) albo ułamka bitów pewnych."""
    if random_phase:
        return estimate.value, estimate.sigma
    p = estimate.value
    slope = abs(math.log2((1.0 - p) / p))
    return 1.0 - h2(p), slope * estimate.sigma


def monte_carlo(cfg: Config, returned: float, random_phase: bool, index: int) -> Point:
    attenuation = 10.0 * math.log10(REFERENCE_PROBE / returned)
    link = Link(
        WCPSource((cfg.mu,)),
        FiberChannel(cfg.length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
        eve=TrojanHorse(REFERENCE_PROBE, attenuation, random_phase),
    )
    counts = run_counts_parallel(
        link, cfg.mc_pulses, np.random.SeedSequence([cfg.seed, index]), chunk_size=5_000_000
    )
    honest = matched_basis_clicks(cfg.mu, GYS.eta(cfg.length_km), P_D, GYS.misalignment)
    estimate = eve_known_fraction(counts) if random_phase else eve_guess_accuracy(counts)
    information, sigma = information_estimate(estimate, random_phase)
    return Point(
        returned,
        random_phase,
        information,
        sigma,
        bit_information(returned, random_phase),
        matched_qber(counts).z_score(honest.qber),
        measured_gain(counts).z_score(honest.detected),
    )


def hidden_ir_photons(misalignment: float, threshold: float, random_phase: bool) -> float:
    """μ_out, od którego intercept-resend z bazą z sondy daje QBER ≤ progu.

    Trafiona baza: Bob widzi tylko błąd optyki e_d; chybiona: błąd ½. QBER = P e_d + (1 − P)/2.
    """
    target = (0.5 - threshold) / (0.5 - misalignment)

    def gap(log_mu: float) -> float:
        return basis_guess_probability(math.exp(log_mu), random_phase) - target

    return math.exp(brentq(gap, math.log(1e-6), math.log(30.0)))


def comma(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def scientific(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa.replace('.', ',')}·10^{int(exponent)}"


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    points = [
        monte_carlo(cfg, returned, random_phase, i)
        for i, (returned, random_phase) in enumerate(
            (r, phase) for phase in (False, True) for r in cfg.mc_returned
        )
    ]
    gisin = (bit_information(GISIN["mu"]), bit_information(GISIN["mu"], True))
    thresholds = {
        phase: hidden_ir_photons(GYS.misalignment, cfg.qber_threshold, phase)
        for phase in (False, True)
    }
    eta = GYS.eta(cfg.length_km)
    key_per_sifted = rate_vacuum_weak(GYS, cfg.length_km, cfg.mu, cfg.nu) / (
        GYS.q_sift * gain(cfg.mu, eta, GYS.background)
    )

    lines = [
        "## Odtworzenie Gisina et al. 2006 (rys. 7, μ_out = 0,1)",
        "",
        "| Faza | Artykuł | Model |",
        "| --- | --- | --- |",
        f"| stała (pomiar Helstroma) | {comma(GISIN['coherent'], 3)} bitu | "
        f"{comma(gisin[0], 4)} bitu |",
        f"| losowa (licznik fotonów) | {comma(GISIN['random'], 3)} bitu | "
        f"{comma(gisin[1], 4)} bitu |",
        "",
        f"## Monte Carlo w łączu GYS ({cfg.length_km:g} km, μ = {comma(cfg.mu, 2)}, "
        f"{scientific(cfg.mc_pulses)} impulsów na punkt)",
        "",
        "| μ_out | Faza | Informacja Ewy (MC) | Model | z QBER | z zysku |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for p in points:
        lines.append(
            f"| {comma(p.returned, 3)} | {'losowa' if p.random_phase else 'stała'} | "
            f"{p.information:.4f} ± {p.sigma:.4f} | {p.model:.4f} | {p.qber_z:+.2f} | "
            f"{p.gain_z:+.2f} |".replace(".", ",")
        )
    lines += [
        "",
        "Sygnału Ewa nie dotyka, więc QBER i zysk są takie jak bez ataku (kolumny z).",
        "",
        "## Baza w czasie rzeczywistym",
        "",
        f"Intercept-resend z bazą odgadniętą z sondy mieści się pod progiem QBER "
        f"{comma(100 * cfg.qber_threshold, 0)}%, gdy Ewa trafia bazę z p-stwem ≥ "
        f"{comma((0.5 - cfg.qber_threshold) / (0.5 - GYS.misalignment), 3)}. Potrzebuje do tego "
        f"μ_out ≥ {comma(thresholds[False], 2)} ze stałą fazą i ≥ {comma(thresholds[True], 2)} "
        "z losową. Przy małym μ_out przewaga w odgadywaniu bazy rośnie jak 0,27·√μ ze stałą fazą, "
        "ale tylko jak μ²/8 z losową: bez odniesienia fazowego pojedynczy foton nic nie mówi "
        "o bazie.",
        "",
        f"## Wymagana izolacja (μ_in fotonów sondy, tłumik Alice {comma(cfg.attenuator_db, 0)} dB)",
        "",
        "| Odbicie (Gisin et al. 2006) | μ_in | Wyciek ≤ "
        + " | Wyciek ≤ ".join(f"{scientific(i)} bitu" for i in cfg.max_information)
        + " |",
        "| --- | --- | " + " | ".join("---" for _ in cfg.max_information) + " |",
    ]
    isolation: dict[str, dict[str, list[float]]] = {}
    for name, reflection in GISIN_REFLECTION_DB.items():
        isolation[name] = {}
        for probe in cfg.probe_photons:
            cells, values = [], []
            for leak in cfg.max_information:
                coherent = required_isolation_db(probe, leak, reflection, cfg.attenuator_db)
                random = required_isolation_db(probe, leak, reflection, cfg.attenuator_db, True)
                values += [coherent, random]
                cells.append(f"{comma(coherent, 0)} dB ({comma(random, 0)} dB z losową fazą)")
            isolation[name][f"{probe:g}"] = values
            lines.append(
                f"| {name} (−{reflection:g} dB) | {scientific(probe)} | " + " | ".join(cells) + " |"
            )
    lines += [
        "",
        "Każdy rząd wielkości μ_in to 10 dB izolacji więcej, a losowa faza oszczędza "
        f"{comma(10 * math.log10(1 / math.log(2)), 1)} dB przy małym wycieku (czynnik 1/ln 2).",
        "",
        f"Koszt w kluczu: przy {cfg.length_km:g} km decoy vacuum+weak daje "
        f"{comma(key_per_sifted, 3)} bitu klucza na bit po siftingu. Wyciek I na impuls trzeba "
        "usunąć dodatkowym wzmacnianiem prywatności: I = 10⁻³ zabiera "
        f"{comma(100 * 1e-3 / key_per_sifted, 2)}% klucza, I = 0,135 (przykład Gisina) "
        f"{comma(100 * 0.135 / key_per_sifted, 0)}%.",
        "",
    ]

    # (d) Lucamarini et al. 2015, Tabela I: μ_out = 10⁻⁶ przy N = 10²⁰ fotonów/s.
    lines += [
        "",
        "## Budżet izolacji według Lucamariniego et al. 2015 "
        f"(μ_out = {scientific(LUCAMARINI_TARGET)}, N = 10²⁰ fotonów/s)",
        "",
        "| f_A | μ_in = N/f_A | γ z wzoru (nasz rachunek) | γ w Tabeli I | |Ṙ| + 2|Ȧ| + n|İ| | "
        "Informacja Ewy przy μ_out (stała / losowa faza) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lucamarini = []
    for clock, gamma, reflection, attenuator, isolator, count in LUCAMARINI_TABLE_I:
        probe = lidt_probe_photons(clock)
        ours = 10.0 * math.log10(probe / LUCAMARINI_TARGET)
        components = round_trip_attenuation_db(count * isolator, reflection, attenuator)
        lucamarini.append((clock, ours, gamma, components))
        lines.append(
            f"| {scientific(clock)} Hz | {scientific(probe)} | {comma(ours, 1)} dB | "
            f"{gamma:g} dB | "
            f"{reflection:g} + 2·{attenuator:g} + {count}·{isolator:g} = {components:g} dB | "
            f"{scientific(bit_information(LUCAMARINI_TARGET))} / "
            f"{scientific(bit_information(LUCAMARINI_TARGET, True))} bitu |"
        )
    loose = required_isolation_db(lidt_probe_photons(1e9), cfg.max_information[0], 0.0)
    lines += [
        "",
        "Wzór (4) artykułu, μ_out = Nγ/f_A, to nasz μ_out = μ_in·10^(−A/10) z μ_in ograniczonym "
        "progiem uszkodzenia światłowodu; składowe izolacji sumują się jak w naszym "
        "`round_trip_attenuation_db` (odbicie raz, tłumik dwa razy). Kryterium artykułu "
        "μ_out = 10⁻⁶ daje Ewie ok. 10⁻⁶ bitu na impuls; przy naszym progu wycieku "
        f"{scientific(cfg.max_information[0])} bitu wystarcza μ_out ≤ "
        f"{scientific(tolerable_photons(cfg.max_information[0]))}, czyli przy 1 GHz "
        f"{comma(loose, 0)} dB zamiast 170 dB. Stopy klucza pod atakiem (wzór (6) artykułu) "
        "nie liczymy.",
    ]

    grid = np.logspace(-6, 1, 141)
    fig, (ax_b, ax_s, ax_i) = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for random_phase, color, label in ((False, "C3", "stała faza"), (True, "C0", "losowa faza")):
        ax_b.loglog(
            grid, [bit_information(m, random_phase) for m in grid], color=color, label=label
        )
        mc = [p for p in points if p.random_phase is random_phase]
        ax_b.errorbar(
            [p.returned for p in mc],
            [p.information for p in mc],
            yerr=[p.sigma for p in mc],
            fmt="o",
            color=color,
            label=f"Monte Carlo, {label}",
        )
        ax_s.loglog(
            grid,
            [basis_guess_probability(m, random_phase) - 0.5 for m in grid],
            color=color,
            label=label,
        )
        ax_s.axvline(thresholds[random_phase], color=color, linestyle=":")
    ax_b.plot(
        [GISIN["mu"]] * 2,
        [GISIN["coherent"], GISIN["random"]],
        "D",
        color="C1",
        label="Gisin et al. 2006",
    )
    ax_b.set(
        xlabel="fotony wracające do Ewy μ_out",
        ylabel="informacja Ewy o bicie [bit na impuls]",
        title="Bit przy znanej bazie (Ewa trzyma sondę do siftingu)",
        ylim=(1e-7, 1.5),
    )
    ax_b.legend(fontsize=8)
    ax_b.grid(True, which="both", alpha=0.3)
    ax_s.axhline(
        (0.5 - cfg.qber_threshold) / (0.5 - GYS.misalignment) - 0.5,
        color="k",
        linestyle="--",
        label=f"intercept-resend pod progiem {comma(100 * cfg.qber_threshold, 0)}%",
    )
    ax_s.set(
        xlabel="fotony wracające do Ewy μ_out",
        ylabel="P(trafiona baza) − ½",
        title="Baza w czasie rzeczywistym (Helstrom, stany mieszane)",
        ylim=(1e-14, 1.0),
    )
    ax_s.legend(fontsize=8)
    ax_s.grid(True, which="both", alpha=0.3)

    leaks = np.logspace(-9, -0.5, 60)
    for (name, reflection), color in zip(
        GISIN_REFLECTION_DB.items(), ("C2", "C4", "C5"), strict=True
    ):
        for random_phase, style in ((False, "-"), (True, "--")):
            ax_i.semilogx(
                leaks,
                [
                    required_isolation_db(
                        REFERENCE_PROBE, leak, reflection, cfg.attenuator_db, random_phase
                    )
                    for leak in leaks
                ],
                color=color,
                linestyle=style,
                label=f"{name}{', losowa faza' if random_phase else ''}",
            )
    ax_i.set(
        xlabel="dopuszczalna informacja Ewy [bit na impuls]",
        ylabel="wymagana izolacja [dB]",
        title=f"Izolacja przy sondzie μ_in = {scientific(REFERENCE_PROBE)} fotonów",
    )
    ax_i.legend(fontsize=7)
    ax_i.grid(True, which="both", alpha=0.3)
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
                "gisin": {"coherent": gisin[0], "random": gisin[1]},
                "points": [asdict(p) for p in points],
                "hidden_ir_returned_photons": {
                    "coherent": thresholds[False],
                    "random": thresholds[True],
                },
                "isolation_db": isolation,
                "key_per_sifted_bit": key_per_sifted,
                "lucamarini": lucamarini,
            },
            indent=2,
        ),
        "utf-8",
    )
    checks = [
        abs(gisin[0] - GISIN["coherent"]) < 5e-4,
        abs(gisin[1] - GISIN["random"]) < 5e-4,
        all(abs(p.information - p.model) < 4.0 * p.sigma for p in points),
        all(abs(p.qber_z) < 4.0 and abs(p.gain_z) < 4.0 for p in points),
        all(bit_guess_probability(p.returned, p.random_phase) > 0.5 for p in points),
        all(
            abs(ours - gamma) < 0.05 and components == gamma
            for _, ours, gamma, components in lucamarini
        ),
    ]
    print("sprawdzenia:", checks)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
