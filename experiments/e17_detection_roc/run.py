"""E17: obrony z progiem dla ataków 1, 4, 5 i 6 i ich krzywe ROC.

Wynik: figures/e17_detection_roc.png, figures/e17_detection_roc.md
i results/e17_detection_roc.json.
(a) test progu QBER na próbce k bitów (ataki 1 i 6): dokładne krzywe ROC dwumianowe;
(b) monitor czasu przyjścia (atak 4): ROC dla krzywych Zhao przy dwóch rozmyciach czasowych,
    z Monte Carlo na przesunięciach z prawdziwego przebiegu ataku;
(c) monitor światła wchodzącego do Alice (atak 5): p-stwo wykrycia sondy, największa
    niezauważona sonda i wynikająca z niej izolacja.
"""

import dataclasses
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from qkdsim.analysis.intercept_resend import intercept_resend_model
from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.time_shift import fit_zhao, gain_matching_transmission, time_shift_model
from qkdsim.analysis.trojan_horse import required_isolation_db
from qkdsim.attacks.time_shift import TimeShift
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.defenses.input_monitor import InputMonitor
from qkdsim.defenses.power_monitor import PowerMonitor
from qkdsim.defenses.qber_test import binomial_auc, binomial_roc, detection_probability
from qkdsim.defenses.timing_monitor import TimingMonitor
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link, simulate
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
P_D = dark_count_per_detector(GYS.background)


@dataclass(frozen=True)
class Config:
    seed: int
    false_alarm: float
    qber_sample: int
    noise_misalignment: float
    ir_fractions: tuple[float, ...]
    fuchs_errors: tuple[float, ...]
    timing_length_km: float
    timing_pulses: int
    timing_jitters_ns: tuple[float, ...]
    timing_window_ns: float
    timing_clicks: int
    timing_sessions: int
    monitor_tap: float
    monitor_noise_w: float
    monitor_windows: int
    repetition_hz: float
    missed: float
    max_information: float
    reflection_db: float


@dataclass(frozen=True)
class QberRow:
    attack: str
    qber: float
    detected: float
    auc: float


@dataclass(frozen=True)
class TimingRow:
    jitter_ns: float
    p_honest: float
    p_attacked: float
    model: float
    monte_carlo: float
    false_alarms: float
    auc: float


def comma(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def scientific(value: float, digits: int = 1) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa.replace('.', ',')}·10^{int(exponent)}"


def noise_qber(cfg: Config, fraction: float) -> float:
    """QBER przy 0 km z błędem optyki tła i atakiem 1 na ułamku impulsów (jak w e07)."""
    params = dataclasses.replace(GYS, misalignment=cfg.noise_misalignment)
    return intercept_resend_model(params, 0.0, 0.48, fraction, 1.0, P_D).clicks.qber


def time_shift_arrivals(cfg: Config) -> tuple[np.ndarray, float]:
    """Przesunięcia wykrytych impulsów z przebiegu ataku 4 (krzywe Zhao, zysk wyrównany)."""
    fit = fit_zhao()
    t = 10.0 ** (-GYS.alpha_db_per_km * cfg.timing_length_km / 10.0)
    transmission = gain_matching_transmission(
        0.48, t, GYS.eta_bob, P_D, GYS.misalignment, fit.mismatch, fit.shifts_ns
    )
    assert transmission is not None
    p_a = time_shift_model(
        0.48, transmission, GYS.eta_bob, P_D, GYS.misalignment, fit.mismatch, fit.shifts_ns
    ).p_a
    link = Link(
        WCPSource((0.48,)),
        FiberChannel(cfg.timing_length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D, timing=fit.mismatch),
        eve=TimeShift(*fit.shifts_ns, p_a=p_a, transmission=transmission),
    )
    _, bob, eve = simulate(link, cfg.timing_pulses, RandomStreams.from_seed(cfg.seed))
    return np.asarray(fit.shifts_ns)[eve.bit[bob.detected]], p_a


def qber_rows(cfg: Config, honest: float) -> list[QberRow]:
    attacks = [(f"atak 1, p = {comma(100 * p, 0)}%", noise_qber(cfg, p)) for p in cfg.ir_fractions]
    attacks += [(f"atak 6, D = {comma(100 * d, 0)}%", d) for d in cfg.fuchs_errors]
    return [
        QberRow(
            name,
            qber,
            detection_probability(cfg.qber_sample, honest, qber, cfg.false_alarm),
            binomial_auc(cfg.qber_sample, honest, qber),
        )
        for name, qber in attacks
    ]


def timing_rows(cfg: Config, rng: np.random.Generator) -> tuple[list[TimingRow], float]:
    arrivals, p_a = time_shift_arrivals(cfg)
    shifts = fit_zhao().shifts_ns
    rows = []
    for jitter in cfg.timing_jitters_ns:
        monitor = TimingMonitor(jitter, cfg.timing_window_ns)
        p0 = monitor.outside_probability(0.0)
        p1 = p_a * monitor.outside_probability(shifts[0])
        p1 += (1 - p_a) * monitor.outside_probability(shifts[1])
        limit = monitor.alarm_limit(cfg.timing_clicks, cfg.false_alarm)
        sessions = range(cfg.timing_sessions)
        attacked = [monitor.outside(rng.choice(arrivals, cfg.timing_clicks), rng) for _ in sessions]
        honest = [monitor.outside(np.zeros(cfg.timing_clicks), rng) for _ in sessions]
        rows.append(
            TimingRow(
                jitter,
                p0,
                p1,
                float(binom.sf(limit, cfg.timing_clicks, p1)),
                float(np.mean(np.array(attacked) > limit)),
                float(np.mean(np.array(honest) > limit)),
                binomial_auc(cfg.timing_clicks, p0, p1),
            )
        )
    return rows, p_a


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    rng = np.random.default_rng(cfg.seed)
    honest_qber = noise_qber(cfg, 0.0)
    qber = qber_rows(cfg, honest_qber)
    timing, p_a = timing_rows(cfg, rng)
    shifts = fit_zhao().shifts_ns
    monitor = InputMonitor(
        PowerMonitor(cfg.monitor_tap, cfg.monitor_noise_w), cfg.repetition_hz, cfg.monitor_windows
    )
    hidden = monitor.largest_hidden_probe(cfg.missed, cfg.false_alarm)
    isolation = required_isolation_db(hidden, cfg.max_information, cfg.reflection_db)

    lines = [
        f"## Test progu QBER (próbka {cfg.qber_sample} bitów, tło {comma(100 * honest_qber, 2)}%)",
        "",
        f"| Atak | QBER pod atakiem | Wykrycie przy {comma(100 * cfg.false_alarm, 0)}% "
        "fałszywych alarmów | AUC |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.attack} | {comma(100 * r.qber, 2)}% | {comma(100 * r.detected, 1)}% | "
        f"{comma(r.auc, 3)} |"
        for r in qber
    ]
    lines += [
        "",
        "Liczba błędów w próbce to Bin(k, Q), więc krzywe są dokładne. Atak 1 na 4% impulsów "
        "wykrywa się w 23% sesji, jak w Monte Carlo e07. Atak 6 daje QBER równy D, więc przy "
        "tle 2% test wykrywa go już przy D = 5%; poniżej tła nie odróżni go od szumu, ale wtedy "
        "rachunek klucza i tak przypisuje cały QBER Ewie (założenie Z7).",
        "",
        f"## Monitor czasu przyjścia (atak 4, {cfg.timing_clicks} kliknięć na test, okno "
        f"±{comma(1000 * cfg.timing_window_ns, 0)} ps)",
        "",
        "| Rozmycie | Poza oknem: uczciwie / atak | Wykrycie (model / MC) "
        "| Fałszywe alarmy (MC) | AUC |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {comma(1000 * r.jitter_ns, 0)} ps | {comma(100 * r.p_honest, 2)}% / "
        f"{comma(100 * r.p_attacked, 1)}% | {comma(100 * r.model, 1)}% / "
        f"{comma(100 * r.monte_carlo, 1)}% | {comma(100 * r.false_alarms, 1)}% | "
        f"{comma(r.auc, 3)} |"
        for r in timing
    ]
    lines += [
        "",
        f"Ewa przesuwa impulsy o {comma(1000 * shifts[0], 0)} i {comma(1000 * shifts[1], 0)} ps "
        f"(p_A = {comma(100 * p_a, 1)}%), więc nawet przy rozmyciu 150 ps większość kliknięć "
        "wypada poza okno. Losowe przypisanie bit ↔ detektor (e11) usuwa jej wiedzę, a monitor "
        "czasu wykrywa sam atak.",
        "",
        "## Monitor światła wchodzącego do Alice (atak 5)",
        "",
        f"Fotodioda jak w e05: odczep {comma(100 * cfg.monitor_tap, 0)}%, szum "
        f"{scientific(cfg.monitor_noise_w)} W na odczyt, średnia z {cfg.monitor_windows} odczytów, "
        f"{comma(cfg.repetition_hz / 1e6, 0)} MHz, {comma(100 * cfg.false_alarm, 0)}% fałszywych "
        f"alarmów. Sondę przeoczy z p-stwem ≥ {comma(100 * cfg.missed, 0)}% tylko do "
        f"{scientific(hidden)} fotonów na impuls. Żeby wyciek nie przekroczył "
        f"{scientific(cfg.max_information, 0)} bitu na impuls przy odbiciu od modulatora "
        f"(−{comma(cfg.reflection_db, 0)} dB), wystarczy wtedy {comma(isolation, 1)} dB izolacji: "
        "monitor ogranicza μ_in, izolacja μ_out (e12).",
    ]

    fig, (ax_q, ax_t, ax_m) = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for row, color in zip(qber, ("C0", "C1", "C2", "C3"), strict=True):
        fpr, tpr = binomial_roc(cfg.qber_sample, honest_qber, row.qber)
        ax_q.plot(fpr, tpr, color=color, label=row.attack)
    for ax in (ax_q, ax_t):
        ax.plot([0, 1], [0, 1], color="0.6", linestyle=":")
        ax.set(xlabel="fałszywe alarmy (FPR)", ylabel="wykrycie (TPR)")
        ax.grid(True, alpha=0.3)
    ax_q.axvline(cfg.false_alarm, color="0.4", linestyle="--", linewidth=0.8)
    ax_q.set_title(f"Test progu QBER, próbka {cfg.qber_sample} bitów (ataki 1 i 6)")
    ax_q.legend(fontsize=8)
    for row, color in zip(timing, ("C0", "C3"), strict=True):
        fpr, tpr = binomial_roc(cfg.timing_clicks, row.p_honest, row.p_attacked)
        ax_t.plot(fpr, tpr, color=color, label=f"rozmycie {comma(1000 * row.jitter_ns, 0)} ps")
        ax_t.plot([row.false_alarms], [row.monte_carlo], "o", color=color)
    ax_t.plot([], [], "ko", label="Monte Carlo przy α = 1%")
    ax_t.set_title(f"Monitor czasu, {cfg.timing_clicks} kliknięć (atak 4)")
    ax_t.legend(fontsize=8)
    probes = np.logspace(3, 8, 200)
    ax_m.semilogx(
        probes, [monitor.detection_probability(float(mu), cfg.false_alarm) for mu in probes]
    )
    ax_m.axvline(hidden, color="C3", linestyle="--", label=f"przeoczona do {scientific(hidden)}")
    ax_m.set(
        xlabel="fotony sondy na impuls μ_in",
        ylabel="p-stwo alarmu",
        title="Monitor światła wchodzącego do Alice (atak 5)",
    )
    ax_m.legend(fontsize=8)
    ax_m.grid(True, which="both", alpha=0.3)
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
                "qber_test": [asdict(r) for r in qber],
                "timing": [asdict(r) for r in timing],
                "monitor": {"hidden_probe": hidden, "isolation_db": isolation},
            },
            indent=2,
        ),
        "utf-8",
    )
    sigma = (0.25 / cfg.timing_sessions) ** 0.5
    checks = [
        abs(qber[0].detected - 0.23) < 0.03,  # e07: 23% (Monte Carlo)
        all(abs(r.monte_carlo - r.model) < 4 * sigma for r in timing),
        all(r.false_alarms <= cfg.false_alarm + 4 * sigma for r in timing),
        abs(monitor.detection_probability(hidden, cfg.false_alarm) - (1 - cfg.missed)) < 1e-9,
        isolation > 0.0,
    ]
    print("sprawdzenia:", checks)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
