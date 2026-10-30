"""E09: atak MITM i uwierzytelnianie Wegmana–Cartera (F4.11–F4.14).

Wynik: figures/e09_mitm.png, figures/e09_mitm.md i results/e09_mitm.json.
(a) skuteczność podrobienia tagu w funkcji p: Monte Carlo vs 1/p;
(b) bilans klucza na rundę w funkcji L: klucz wytworzony vs klucz zużyty na uwierzytelnianie
    (tag na cały transkrypt vs tag na każdą wiadomość, gdzie Cascade wysyła wiadomość na błąd);
(c) tabela przebiegu MITM bez uwierzytelniania i z nim.
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

from qkdsim.analysis.key_rate import GYS, rate_vacuum_weak
from qkdsim.analysis.wcp import gain, qber
from qkdsim.attacks.mitm import MITMOutcome, run_mitm
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link, run_sifted
from qkdsim.postprocessing.auth import MERSENNE_61, KeyPool, WegmanCarter
from qkdsim.postprocessing.run import postprocess
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
P_D = dark_count_per_detector(GYS.background)
TAG_BITS = MERSENNE_61.bit_length()
FIXED_MESSAGES = 13  # próbka (3), permutacje i parzystości Cascade (7), weryfikacja (2), PA (1)


@dataclass(frozen=True)
class Config:
    seed: int
    mitm_length_km: float
    mitm_pulses: int
    forgery_primes: tuple[int, ...]
    forgery_trials: int
    balance_round_pulses: tuple[int, ...]
    check_length_km: float
    check_pulses: int


def link(length_km: float) -> Link:
    return Link(
        WCPSource((0.48,)),
        FiberChannel(length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
    )


def forgery_rate(cfg: Config, prime: int) -> float:
    rng = np.random.default_rng([cfg.seed, prime])
    bits = rng.integers(0, 2, 200, dtype=np.uint8)
    successes = 0
    for t in range(cfg.forgery_trials):
        key = np.random.default_rng([cfg.seed, prime, t]).integers(0, 2, 128, dtype=np.uint8)
        successes += WegmanCarter(KeyPool(key), prime).verify(bits, int(rng.integers(0, prime)))
    return successes / cfg.forgery_trials


def messages_per_round(length_km: float, pulses: int) -> float:
    """Model: wiadomości stałe + jedna wiadomość BINARY na każdy poprawiony błąd."""
    eta = GYS.eta(length_km)
    sifted = 0.5 * gain(0.48, eta, GYS.background) * pulses
    errors = 0.9 * sifted * qber(0.48, eta, GYS.background, GYS.misalignment)
    return FIXED_MESSAGES + errors


def thousands(value: float) -> str:
    """Liczba całkowita z odstępem jako separatorem tysięcy."""
    return f"{value:,.0f}".replace(",", " ")


def describe(outcome: MITMOutcome) -> list[str]:
    rows = []
    for name, side in (("Alice ↔ Ewa", outcome.alice_side), ("Ewa ↔ Bob", outcome.bob_side)):
        if side.result is not None:
            b = side.result.balance
            rows.append(
                f"{name}: klucz {b.final_length} bitów, QBER {100 * b.qber_sample:.2f}%".replace(
                    ".", ","
                )
            )
        else:
            assert side.failure is not None
            failure = side.failure
            rows.append(
                f"{name}: przerwanie przy wiadomości nr {failure.message_index} ({failure.kind})"
            )
    return rows


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    mitm_link = link(cfg.mitm_length_km / 2)  # Ewa w połowie światłowodu
    scenarios = {
        "bez uwierzytelniania": run_mitm(mitm_link, mitm_link, cfg.mitm_pulses, cfg.seed),
        "Wegman–Carter, tag na wiadomość": run_mitm(
            mitm_link, mitm_link, cfg.mitm_pulses, cfg.seed, 10**6, "message"
        ),
        "Wegman–Carter, tag na transkrypt": run_mitm(
            mitm_link, mitm_link, cfg.mitm_pulses, cfg.seed, 10**6, "transcript"
        ),
    }
    keys = scenarios["bez uwierzytelniania"].keys()
    assert keys is not None
    lines = [
        "## Atak MITM (GYS, Ewa w połowie 25 km)",
        "",
        "| Scenariusz | Sesja Alice ↔ Ewa | Sesja Ewa ↔ Bob |",
        "| --- | --- | --- |",
    ]
    for name, outcome in scenarios.items():
        left, right = describe(outcome)
        lines.append(f"| {name} | {left.split(': ', 1)[1]} | {right.split(': ', 1)[1]} |")
    n = min(map(len, keys))
    agreement = 100 * float(np.mean(keys[0][:n] == keys[1][:n]))
    lines += [
        "",
        f"Bez uwierzytelniania klucze Alice i Boba są różne (zgodność {agreement:.0f}% bitów, jak "
        "dla losowych), a Ewa zna oba: w sesji z Alice była Bobem, w sesji z Bobem była Alice. "
        "QBER na obu odcinkach jest taki jak bez ataku.",
    ]

    streams = RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, 3]))
    check = postprocess(
        run_sifted(link(cfg.check_length_km), cfg.check_pulses, streams), streams.public
    )
    counted, modelled = (
        len(check.channel.transcript),
        messages_per_round(cfg.check_length_km, cfg.check_pulses),
    )
    lines += [
        "",
        f"Model liczby wiadomości na rundę przy {cfg.check_length_km:g} km i "
        f"{thousands(cfg.check_pulses)} impulsach: {thousands(modelled)}; w symulacji: "
        f"{thousands(counted)}.",
    ]

    rates = {p: forgery_rate(cfg, p) for p in cfg.forgery_primes}
    lengths = np.linspace(0.0, 145.0, 291)
    fig, (ax_f, ax_b) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    primes = np.array(cfg.forgery_primes, dtype=float)
    ax_f.loglog(primes, 1 / primes, "k--", label="1/p")
    ax_f.errorbar(
        primes,
        list(rates.values()),
        yerr=[math.sqrt(r * (1 - r) / cfg.forgery_trials) for r in rates.values()],
        fmt="o",
        color="C3",
        label=f"Monte Carlo, {thousands(cfg.forgery_trials)} prób",
    )
    ax_f.set(
        xlabel="liczba pierwsza p (tag ma ⌈log₂ p⌉ bitów)",
        ylabel="p-stwo podrobienia tagu",
        title="Fałszerstwo tagu Wegmana–Cartera bez znajomości klucza",
    )
    ax_f.legend(fontsize=8)
    ax_f.grid(True, which="both", alpha=0.3)

    for color, pulses in zip(("C0", "C1", "C2"), cfg.balance_round_pulses, strict=True):
        produced = [max(rate_vacuum_weak(GYS, x, 0.48, 0.05), 0.0) * pulses for x in lengths]
        label = f"{pulses:.0e}".replace("e+0", "·10^")
        ax_b.semilogy(
            lengths,
            np.where(np.array(produced) > 0, produced, np.nan),
            color=color,
            label=f"klucz wytworzony, N = {label}",
        )
        ax_b.semilogy(
            lengths,
            [TAG_BITS * messages_per_round(x, pulses) for x in lengths],
            color=color,
            linestyle=":",
            label=f"zużyty, tag na wiadomość, N = {label}",
        )
    ax_b.axhline(
        2 * TAG_BITS, color="k", linestyle="--", label="zużyty, tag na transkrypt (2 × 61 bitów)"
    )
    ax_b.set(
        xlabel="długość światłowodu L [km]",
        ylabel="bity na rundę",
        title="Bilans klucza: QKD rozszerza klucz, a nie tworzy go z niczego",
        ylim=(10, 1e6),
    )
    ax_b.legend(fontsize=7, loc="upper right")
    ax_b.grid(True, which="both", alpha=0.3)
    fig.savefig(ROOT / "figures" / f"{NAME}.png", dpi=150)
    plt.close(fig)

    break_even = {}
    for pulses in cfg.balance_round_pulses:
        positive = [
            x for x in lengths if rate_vacuum_weak(GYS, x, 0.48, 0.05) * pulses > 2 * TAG_BITS
        ]
        break_even[pulses] = max(positive) if positive else 0.0
    lines += [
        "",
        "| Impulsy na rundę N | Zasięg z dodatnim bilansem (tag na transkrypt) |",
        "| --- | --- |",
    ]
    lines += [f"| {thousands(p)} | {d:.1f} km |".replace(".", ",") for p, d in break_even.items()]
    text = "\n".join(lines) + "\n"
    (ROOT / "figures" / f"{NAME}.md").write_text(text, "utf-8")
    print(text)
    for p, r in rates.items():
        print(f"p = {p}: fałszerstwo {r:.5f} (1/p = {1 / p:.5f})")
    out = ROOT / "results" / f"{NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "forgery": rates,
                "break_even_km": break_even,
                "messages": {"counted": counted, "model": modelled},
            },
            indent=2,
        ),
        "utf-8",
    )
    worst = max(
        abs(r - 1 / p) / math.sqrt((1 / p) * (1 - 1 / p) / cfg.forgery_trials)
        for p, r in rates.items()
    )
    detected = all(o.aborted for name, o in scenarios.items() if "Wegman" in name)
    return 0 if worst < 4.0 and detected and not scenarios["bez uwierzytelniania"].aborted else 1


if __name__ == "__main__":
    sys.exit(main())
