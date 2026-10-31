"""E19: symulator QRNG na dzielniku wiązki (przedłużenie ataku 8).

Wynik: figures/e19_qrng.png, figures/e19_qrng.md i results/e19_qrng.json.
(a) surowe bity czterech generatorów (idealny, niedopasowane detektory, pamięć detektorów, oba):
    wydajność i obciążenie na tle wzoru dokładnego, korelacje, min-entropia w funkcji kontekstu,
    testy losowości;
(b) ekstrakcja Toeplitzem z min-entropią z kalibracji: testy i min-entropia wyjścia;
(c) atak 8 z QRNG: Alice bierze bazy z silnie niedopasowanego generatora; Ewa robi
    intercept-resend w częstszej bazie. Surowe bity dają jej przewagę, po ekstrakcji znika;
(d) testy zdrowia NIST SP 800-90B: brak fałszywych alarmów, oślepiony detektor (RCT) i dryf
    sprawności (APT, p-stwo wykrycia na okno na tle wzoru dwumianowego).
"""

import dataclasses
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qkdsim.analysis.intercept_resend import intercept_resend_model
from qkdsim.analysis.key_rate import GYS
from qkdsim.analysis.wcp import matched_basis_clicks
from qkdsim.attacks.intercept_resend import InterceptResend
from qkdsim.attacks.weak_rng import MajorityPredictor
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.defenses.health_tests import (
    WINDOW,
    adaptive_alarms,
    adaptive_cutoff,
    adaptive_detection_probability,
    first_repetition_alarm,
    repetition_cutoff,
)
from qkdsim.defenses.randomness_tests import TESTS
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.metrics import eve_known_fraction, matched_qber
from qkdsim.pipeline import Link, run_counts
from qkdsim.qrng import BeamSplitterQrng, QrngSource, min_entropy
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
P_D = dark_count_per_detector(GYS.background)
ALPHA = 0.01


@dataclass(frozen=True)
class Config:
    seed: int
    mean_photons: float
    efficiency: float
    mismatch: float
    strong_mismatch: float
    dead_time_gates: int
    afterpulse: float
    slots: int
    contexts: tuple[int, ...]
    calibration_context: int
    block: int
    epsilon: float
    test_bits: int
    batches: int
    length_km: float
    mu: float
    pulses: int
    drifts: tuple[float, ...]
    drift_windows: int


@dataclass(frozen=True)
class DeviceRow:
    name: str
    bit_rate_mc: float
    bit_rate_model: float
    bias_mc: float
    bias_model: float
    z_bias: float
    same_as_previous: float
    entropies: tuple[float, ...]
    raw_pvalues: tuple[float, ...]
    calibrated_entropy: float
    out_per_block: int
    extracted_pvalues: tuple[float, ...]
    extracted_entropies: tuple[float, ...]


@dataclass(frozen=True)
class AttackRow:
    name: str
    qber_mc: float
    qber_model: float
    z_qber: float
    known_mc: float
    known_model: float
    z_known: float


@dataclass(frozen=True)
class DriftRow:
    ratio: float
    bias: float
    detection_mc: float
    detection_model: float
    z: float


def comma(value: float, digits: int) -> str:
    return f"{round(value, digits) + 0.0:.{digits}f}".replace(".", ",")


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def pvalues(bits: np.ndarray) -> tuple[float, ...]:
    return tuple(test(bits) for test in TESTS.values())


def batch_sigma(bits: np.ndarray, batches: int) -> float:
    """σ średniej z rozrzutu średnich w paczkach (bity są skorelowane)."""
    means = bits[: len(bits) // batches * batches].reshape(batches, -1).mean(axis=1)
    return float(np.std(means, ddof=1) / math.sqrt(batches))


def generators(cfg: Config) -> dict[str, BeamSplitterQrng]:
    eta, tau, ap = cfg.efficiency, cfg.dead_time_gates, cfg.afterpulse
    mismatched = (eta, cfg.mismatch * eta)
    return {
        "idealny": BeamSplitterQrng(cfg.mean_photons, efficiency=(eta, eta)),
        f"η₁/η₀ = {comma(cfg.mismatch, 1)}": BeamSplitterQrng(
            cfg.mean_photons, efficiency=mismatched
        ),
        f"τ = {tau}, p = {comma(100 * ap, 0)}%": BeamSplitterQrng(
            cfg.mean_photons, efficiency=(eta, eta), dead_time_gates=tau, afterpulse=ap
        ),
        "oba": BeamSplitterQrng(
            cfg.mean_photons, efficiency=mismatched, dead_time_gates=tau, afterpulse=ap
        ),
    }


def source(cfg: Config, device: BeamSplitterQrng, entropy: float, index: int) -> QrngSource:
    return QrngSource(
        device,
        np.random.default_rng([cfg.seed, index, 1]),
        np.random.default_rng([cfg.seed, index, 2]),
        entropy,
        cfg.block,
        cfg.epsilon,
    )


def device_row(cfg: Config, name: str, device: BeamSplitterQrng, index: int) -> DeviceRow:
    raw = device.raw_bits(cfg.slots, np.random.default_rng([cfg.seed, index]))
    bias = float(raw.mean())
    entropy = min_entropy(raw, cfg.calibration_context)
    extractor = source(cfg, device, entropy, index)
    out = extractor.bits(cfg.test_bits)
    return DeviceRow(
        name,
        len(raw) / cfg.slots,
        device.bit_rate(),
        bias,
        device.bias(),
        (bias - device.bias()) / batch_sigma(raw, cfg.batches),
        float(np.mean(raw[1:] == raw[:-1])),
        tuple(min_entropy(raw[: cfg.test_bits], k) for k in cfg.contexts),
        pvalues(raw[: cfg.test_bits]),
        entropy,
        extractor.out_per_block,
        pvalues(out),
        tuple(min_entropy(out, k) for k in cfg.contexts),
    )


def attack_rows(cfg: Config) -> tuple[list[AttackRow], float, float]:
    eta = cfg.efficiency
    device = BeamSplitterQrng(cfg.mean_photons, efficiency=(eta, cfg.strong_mismatch * eta))
    entropy = min_entropy(
        device.raw_bits(cfg.slots, np.random.default_rng([cfg.seed, 50])), cfg.calibration_context
    )
    honest = matched_basis_clicks(cfg.mu, GYS.eta(cfg.length_km), P_D, GYS.misalignment)

    def ir_model(right: float) -> tuple[float, float]:
        m = intercept_resend_model(GYS, cfg.length_km, cfg.mu, 1.0, 1.0, P_D, right)
        return m.clicks.qber, m.known_detected / m.clicks.detected

    eve = InterceptResend(1.0, predictor=MajorityPredictor(basis=0))  # Z częstsza, bo P(1) < ½
    cases = [
        ("bez Ewy, QRNG po ekstrakcji", entropy, None, (honest.qber, 0.0)),
        ("intercept-resend w częstszej bazie, surowy QRNG", 1.0, eve, ir_model(1 - device.bias())),
        ("intercept-resend w częstszej bazie, QRNG po ekstrakcji", entropy, eve, ir_model(0.5)),
    ]
    rows = []
    for i, (name, alice_entropy, attack, (qber, known)) in enumerate(cases):
        link = Link(
            WCPSource((cfg.mu,)),
            FiberChannel(cfg.length_km, GYS.alpha_db_per_km, GYS.misalignment),
            DetectorPair(GYS.eta_bob, P_D),
            eve=attack,
        )
        base = RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, 60 + i]))
        alice = source(cfg, device, alice_entropy, 60 + i)
        counts = run_counts(link, cfg.pulses, dataclasses.replace(base, alice=alice))
        q, k = matched_qber(counts), eve_known_fraction(counts)
        rows.append(
            AttackRow(
                name,
                q.value,
                qber,
                q.z_score(qber),
                k.value,
                known,
                k.z_score(known) if known > 0 else 0.0,
            )
        )
    return rows, device.bias(), entropy


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    devices = generators(cfg)
    rows = [device_row(cfg, name, dev, i) for i, (name, dev) in enumerate(devices.items())]
    attacks, strong_bias, strong_entropy = attack_rows(cfg)

    # (d) Testy zdrowia.
    realistic, mismatched = rows[3], rows[1]
    rct, apt = (
        repetition_cutoff(realistic.calibrated_entropy),
        adaptive_cutoff(realistic.calibrated_entropy),
    )
    healthy = devices["oba"].raw_bits(cfg.slots, np.random.default_rng([cfg.seed, 70]))
    healthy_windows = adaptive_alarms(healthy, apt)
    healthy_rct = first_repetition_alarm(healthy, rct)
    blinded_device = dataclasses.replace(devices["oba"], efficiency=(cfg.efficiency, 0.0))
    blinded = blinded_device.raw_bits(1_000_000, np.random.default_rng([cfg.seed, 71]))
    blinded_alarm = first_repetition_alarm(blinded, rct)
    drift_apt = adaptive_cutoff(mismatched.calibrated_entropy)
    drift_rows = []
    for i, ratio in enumerate(cfg.drifts):
        dev = BeamSplitterQrng(
            cfg.mean_photons, efficiency=(cfg.efficiency, ratio * cfg.efficiency)
        )
        slots = math.ceil(1.02 * cfg.drift_windows * WINDOW / dev.bit_rate())
        alarms = adaptive_alarms(
            dev.raw_bits(slots, np.random.default_rng([cfg.seed, 80 + i])), drift_apt
        )
        model = adaptive_detection_probability(dev.bias(), drift_apt)
        mc = float(alarms[: cfg.drift_windows].mean())
        sigma = math.sqrt(max(model * (1 - model), 1e-12) / cfg.drift_windows)
        drift_rows.append(DriftRow(ratio, dev.bias(), mc, model, (mc - model) / sigma))

    names = list(TESTS)
    lines = [
        f"## Surowe bity ({cfg.slots:.0e} taktów, μ = {comma(cfg.mean_photons, 1)}, "
        f"η₀ = {comma(100 * cfg.efficiency, 0)}%)".replace("e+0", "·10^"),
        "",
        "| Generator | Bity na takt: MC / wzór | P(1): MC / wzór | z | P(bit = poprzedni) | "
        + " | ".join(f"H_min, k = {k}" for k in cfg.contexts)
        + " | "
        + " | ".join(f"p {n}" for n in names)
        + " |",
        "| --- | --- | --- | --- | --- | "
        + " | ".join("---" for _ in cfg.contexts)
        + " | "
        + " | ".join("---" for _ in names)
        + " |",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} | {comma(r.bit_rate_mc, 5)} / {comma(r.bit_rate_model, 5)} | "
            f"{comma(r.bias_mc, 4)} / {comma(r.bias_model, 4)} | {comma(r.z_bias, 1)} | "
            f"{comma(r.same_as_previous, 3)} | "
            + " | ".join(comma(h, 3) for h in r.entropies)
            + " | "
            + " | ".join(f"{p:.1e}".replace(".", ",") for p in r.raw_pvalues)
            + " |"
        )
    lines += [
        "",
        "Wzór: P(1) = c₁(1 − c₀)/(c₀(1 − c₁) + c₁(1 − c₀)) ze stacjonarnymi p-stwami kliknięcia "
        "c_i (e18); procesy kliknięć D0 i D1 są niezależne, więc wzór jest dokładny także "
        "z pamięcią. σ z rozrzutu średnich w "
        f"{cfg.batches} paczkach (bity są skorelowane). H_min: najczęstsza wartość z górną "
        "granicą ufności 99% (NIST SP 800-90B, 6.3.1) w najgorszym z 2ᵏ kontekstów k "
        f"poprzednich bitów; H_min i testy na pierwszych {thousands(cfg.test_bits)} bitach, "
        "kalibracja ekstraktora na całej próbce.",
        "",
        "Czas martwy odwraca bity (detektor, który kliknął, milknie na τ taktów), afterpuls je "
        "powtarza. Przy k = 0 korelacji nie widać; min-entropia z kontekstem spada do ok. ½.",
        "",
        f"## Ekstrakcja Toeplitzem (blok {cfg.block}, ε = {cfg.epsilon:.0e})".replace(
            "e-", "·10^-"
        ),
        "",
        "| Generator | H z kalibracji (k = "
        f"{cfg.calibration_context}) | Bity wyjścia na blok | "
        + " | ".join(f"p {n}" for n in names)
        + " | "
        + " | ".join(f"H_min, k = {k}" for k in cfg.contexts)
        + " |",
        "| --- | --- | --- | "
        + " | ".join("---" for _ in names)
        + " | "
        + " | ".join("---" for _ in cfg.contexts)
        + " |",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} | {comma(r.calibrated_entropy, 3)} | {r.out_per_block} | "
            + " | ".join(f"{p:.2f}".replace(".", ",") for p in r.extracted_pvalues)
            + " | "
            + " | ".join(comma(h, 3) for h in r.extracted_entropies)
            + " |"
        )
    lines += [
        "",
        "m = ⌊nH − 2 log₂(1/ε)⌋ (lemat o resztkowym haszowaniu). Wyjście przechodzi testy i ma "
        "min-entropię taką jak idealny generator (spadek z k to kara za ufność przy 2ᵏ "
        "kontekstach).",
        "",
        f"## Atak 8 z QRNG: bazy Alice z generatora η₁/η₀ = {comma(cfg.strong_mismatch, 1)} "
        f"(GYS {cfg.length_km:g} km, μ = {comma(cfg.mu, 2)})",
        "",
        "| Przebieg | QBER: MC / model | z | Bity znane Ewie: MC / model | z |",
        "| --- | --- | --- | --- | --- |",
    ]
    for a in attacks:
        lines.append(
            f"| {a.name} | {comma(100 * a.qber_mc, 2)}% / {comma(100 * a.qber_model, 2)}% | "
            f"{comma(a.z_qber, 1)} | {comma(100 * a.known_mc, 2)}% / "
            f"{comma(100 * a.known_model, 2)}% | {comma(a.z_known, 1)} |"
        )
    lines += [
        "",
        f"Surowy generator daje P(X) = {comma(strong_bias, 3)}, więc Ewa mierzy w bazie Alice "
        f"z p-stwem {comma(1 - strong_bias, 3)} zamiast ½ (model z e13). Po ekstrakcji "
        f"(H = {comma(strong_entropy, 3)}) przewaga znika, a QBER wraca do poziomu ataku "
        "w losowej bazie.",
        "",
        "## Testy zdrowia NIST SP 800-90B (α = 2⁻²⁰)",
        "",
        f"- Generator „oba” (H = {comma(realistic.calibrated_entropy, 3)}): progi RCT C = {rct}, "
        f"APT C = {apt} na {WINDOW}. Na {thousands(len(healthy))} nowych bitach: "
        + f"alarmy APT w {int(healthy_windows.sum())} z {len(healthy_windows)} okien, RCT "
        + ("brak alarmu." if healthy_rct is None else f"alarm przy bicie {healthy_rct}."),
        f"- Oślepiony D1 (η₁ = 0): wszystkie bity równe 0; RCT alarmuje przy bicie "
        f"{blinded_alarm} (C − 1 = {rct - 1}).",
        f"- Dryf sprawności D1 przy H = {comma(mismatched.calibrated_entropy, 3)} z kalibracji "
        f"generatora η₁/η₀ = {comma(cfg.mismatch, 1)} (APT C = {drift_apt}), "
        f"{cfg.drift_windows} okien na punkt:",
        "",
        "| η₁/η₀ | P(1) | P-stwo alarmu APT na okno: MC / wzór | z |",
        "| --- | --- | --- | --- |",
    ]
    for d in drift_rows:
        lines.append(
            f"| {comma(d.ratio, 1)} | {comma(d.bias, 3)} | {comma(d.detection_mc, 4)} / "
            f"{comma(d.detection_model, 4)} | {comma(d.z, 1)} |"
        )
    lines += [
        "",
        "Wzór: p·P(Bin(W − 1, p) ≥ C − 1) + (1 − p)·P(Bin(W − 1, 1 − p) ≥ C − 1). Test w wersji "
        "z normy patrzy tylko na wartość pierwszej próbki okna, więc przy dużym dryfie wykrywa "
        "ok. P(częstszej wartości) okien; po kilku oknach alarm jest pewny.",
    ]

    fig, (ax_h, ax_d, ax_a) = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for r, color in zip(rows, ("C0", "C1", "C2", "C3"), strict=True):
        ax_h.plot(cfg.contexts, r.entropies, "o-", color=color, label=f"{r.name} (surowe)")
        ax_h.plot(cfg.contexts, r.extracted_entropies, "s--", color=color, alpha=0.6)
    ax_h.set(
        xlabel="kontekst k [poprzednie bity]",
        ylabel="H_min na bit (MCV, 99%)",
        title="Min-entropia: surowe (linia) i po ekstrakcji (przerywana)",
        ylim=(0.4, 1.02),
    )
    ax_h.legend(fontsize=8)
    ax_h.grid(True, alpha=0.3)

    grid = np.linspace(0.45, 0.95, 101)
    curve = [
        adaptive_detection_probability(
            BeamSplitterQrng(
                cfg.mean_photons, efficiency=(cfg.efficiency, g * cfg.efficiency)
            ).bias(),
            drift_apt,
        )
        for g in grid
    ]
    ax_d.plot(grid, curve, label="wzór dwumianowy")
    ax_d.plot([d.ratio for d in drift_rows], [d.detection_mc for d in drift_rows], "o", label="MC")
    ax_d.set(
        xlabel="η₁/η₀ po dryfie",
        ylabel="p-stwo alarmu APT na okno",
        title=f"Test APT (C = {drift_apt}) przy dryfie sprawności",
    )
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3)

    labels = ["bez Ewy\n(ekstrakcja)", "IR\n(surowy)", "IR\n(ekstrakcja)"]
    x = np.arange(len(attacks))
    ax_a.bar(x - 0.2, [100 * a.qber_mc for a in attacks], 0.4, label="QBER (MC)")
    ax_a.bar(x + 0.2, [100 * a.known_mc for a in attacks], 0.4, label="bity znane Ewie (MC)")
    ax_a.plot(x - 0.2, [100 * a.qber_model for a in attacks], "k_", markersize=20, label="model")
    ax_a.plot(x + 0.2, [100 * a.known_model for a in attacks], "k_", markersize=20)
    ax_a.set_xticks(x, labels)
    ax_a.set(ylabel="%", title="Bazy Alice z QRNG: przewaga Ewy znika po ekstrakcji")
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
                "devices": [asdict(r) for r in rows],
                "attack": [asdict(a) for a in attacks],
                "health": {
                    "rct_cutoff": rct,
                    "apt_cutoff": apt,
                    "healthy_apt_alarms": int(healthy_windows.sum()),
                    "healthy_rct_alarm": healthy_rct,
                    "blinded_rct_alarm": blinded_alarm,
                    "drift": [asdict(d) for d in drift_rows],
                },
            },
            indent=2,
        ),
        "utf-8",
    )
    bonferroni = ALPHA / (len(rows) * len(names))
    checks = [
        all(abs(r.z_bias) < 4.0 for r in rows),
        all(abs(r.bit_rate_mc / r.bit_rate_model - 1.0) < 3e-3 for r in rows),
        min(rows[0].raw_pvalues) > bonferroni,
        all(min(r.raw_pvalues) < ALPHA for r in rows[1:]),
        all(min(r.extracted_pvalues) > bonferroni for r in rows),
        all(min(r.extracted_entropies) > min(rows[0].entropies) - 0.015 for r in rows),
        all(abs(a.z_qber) < 4.0 and abs(a.z_known) < 4.0 for a in attacks),
        int(healthy_windows.sum()) == 0 and healthy_rct is None,
        blinded_alarm is not None and blinded_alarm <= rct,
        all(abs(d.z) < 4.0 for d in drift_rows),
    ]
    print("sprawdzenia:", checks)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
