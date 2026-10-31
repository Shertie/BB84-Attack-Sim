"""E03: bilans klucza od impulsów do klucza końcowego dla GYS i test liniowej Ewy.

Wynik: figures/e03_key_balance.md (tabela do README) i results/e03_key_balance.json.
"""

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from qkdsim.analysis.key_rate import (
    GYS,
    GYS_REPETITION_RATE_HZ,
    infinite_decoy_singles,
    rate_infinite_decoy,
)
from qkdsim.analysis.wcp import gain
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link, run_sifted
from qkdsim.postprocessing.leakage import linear_eve_knowledge
from qkdsim.postprocessing.run import PostprocessingConfig, postprocess
from qkdsim.protocol import SiftedKey
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name


@dataclass(frozen=True)
class Config:
    seed: int
    mu: float
    lengths_km: tuple[float, ...]
    pulses: tuple[int, ...]
    sample_fraction: float
    linear_eve_bits: int


def balance_at(cfg: Config, index: int) -> dict[str, float]:
    length, n_pulses = cfg.lengths_km[index], cfg.pulses[index]
    link = Link(
        WCPSource((cfg.mu,)),
        FiberChannel(length, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, dark_count_per_detector(GYS.background)),
    )
    streams = RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, index]))
    key = run_sifted(link, n_pulses, streams)

    # GLLP: udział detekcji z impulsów jednofotonowych i ich błąd (granica decoy, model).
    eta = GYS.eta(length)
    y1, e1 = infinite_decoy_singles(eta, GYS.background, GYS.misalignment)
    single_fraction = y1 * cfg.mu * math.exp(-cfg.mu) / gain(cfg.mu, eta, GYS.background)
    config = PostprocessingConfig(
        sample_fraction=cfg.sample_fraction,
        single_photon_fraction=single_fraction,
        phase_error=e1,
    )
    result = postprocess(key, streams.public, config)
    b = result.balance
    return {
        "length_km": length,
        "pulses": n_pulses,
        "sifted": b.sifted,
        "sample": b.sample,
        "qber_sample": b.qber_sample,
        "reconciled": b.reconciled,
        "errors_before_ec": b.errors_before_ec,
        "leak_ec": b.leak_ec,
        "f": b.efficiency,
        "verified": b.verified,
        "leak_verification": b.leak_verification,
        "single_photon_fraction": single_fraction,
        "e1": e1,
        "final_length": b.final_length,
        "rate_per_pulse": b.final_length / n_pulses,
        "rate_analytic": rate_infinite_decoy(GYS, length, cfg.mu),
    }


def linear_eve_demo(cfg: Config) -> dict[str, int]:
    """Liniowa Ewa na fragmencie prawdziwego klucza z 25 km: wiedza Ewy z i bez odjęcia wycieku."""
    link = Link(
        WCPSource((cfg.mu,)),
        FiberChannel(25.0, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, dark_count_per_detector(GYS.background)),
    )
    streams = RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, 99]))
    full = run_sifted(link, 2_000_000, streams)
    n = cfg.linear_eve_bits
    key = SiftedKey(full.index[:n], full.alice[:n], full.bob[:n])
    out: dict[str, int] = {}
    for subtract in (True, False):
        config = PostprocessingConfig(record_rows=True, subtract_leakage=subtract)
        result = postprocess(key, np.random.default_rng([cfg.seed, 99, subtract]), config)
        eve = linear_eve_knowledge(result.channel, result.balance.reconciled, result.pa_seed)
        suffix = "with_leak" if subtract else "without_leak"
        out[f"length_{suffix}"] = result.balance.final_length
        out[f"eve_bits_{suffix}"] = eve.known_final_bits
        out["reconciled"] = result.balance.reconciled
        out["disclosed_rank"] = eve.disclosed_rank
    return out


def _pl(text: str) -> str:
    """Zapis polski: spacja jako separator tysięcy, przecinek dziesiętny."""
    return text.replace(",", " ").replace(".", ",")


def markdown(points: list[dict[str, float]], eve: dict[str, int]) -> str:
    rows = [
        ("Wysłane impulsy N", "pulses", "{:,.0f}"),
        ("Bity po siftingu", "sifted", "{:,.0f}"),
        ("Próbka do estymacji (ujawniona)", "sample", "{:,.0f}"),
        ("QBER z próbki", "qber_sample", "{:.2%}"),
        ("Klucz do korekcji n", "reconciled", "{:,.0f}"),
        ("Błędy przed korekcją", "errors_before_ec", "{:,.0f}"),
        ("Wyciek Cascade leak_EC [bit]", "leak_ec", "{:,.0f}"),
        ("Sprawność Cascade f", "f", "{:.3f}"),
        ("Wyciek weryfikacji [bit]", "leak_verification", "{:,.0f}"),
        ("Udział detekcji jednofotonowych Q₁/Q_μ", "single_photon_fraction", "{:.3f}"),
        ("Błąd jednofotonowy e₁", "e1", "{:.2%}"),
        ("Klucz końcowy ℓ [bit]", "final_length", "{:,.0f}"),
        ("ℓ / N [bit na impuls]", "rate_per_pulse", "{:.2e}"),
        ("Wzór asymptotyczny R(L) [bit na impuls]", "rate_analytic", "{:.2e}"),
    ]
    header = "| Krok | " + " | ".join(f"{p['length_km']:.0f} km" for p in points) + " |"
    lines = [header, "| --- |" + " --- |" * len(points)]
    for label, key, fmt in rows:
        cells = " | ".join(_pl(fmt.format(p[key])) for p in points)
        lines.append(f"| {label} | {cells} |")
    rate_hz = [p["rate_per_pulse"] * GYS_REPETITION_RATE_HZ for p in points]
    lines.append("| Przy 2 MHz [bit/s] | " + " | ".join(_pl(f"{r:,.0f}") for r in rate_hz) + " |")
    return (
        "## Bilans klucza (GYS, μ = 0,48)\n\n"
        + "\n".join(lines)
        + "\n\nℓ/N jest o kilka procent niższe od wzoru asymptotycznego. 10% bitów po siftingu "
        "idzie na próbkę (strata ×0,9), co tylko częściowo kompensuje Cascade z f ≈ 1,16 zamiast "
        "1,22 przyjętego we wzorze. Udział jednofotonowych i e₁ pochodzą z modelu "
        "(granica decoy); estymatę decoy z danych Monte Carlo pokazuje e06.\n\n"
        "## Pasywna Ewa i zapomniany wyciek\n\n"
        f"Klucz {eve['reconciled']} bitów z 25 km. Ewa zna wszystkie parzystości Cascade i hasz "
        f"weryfikacji: rząd {eve['disclosed_rank']} niezależnych funkcji liniowych klucza.\n\n"
        "| Długość klucza końcowego | ℓ [bit] | Bity znane Ewie z pewnością |\n"
        "| --- | --- | --- |\n"
        f"| z odjęciem wycieku | {eve['length_with_leak']} | {eve['eve_bits_with_leak']} |\n"
        f"| bez odjęcia wycieku | {eve['length_without_leak']} | "
        f"{eve['eve_bits_without_leak']} |\n"
    )


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    points = [balance_at(cfg, i) for i in range(len(cfg.lengths_km))]
    eve = linear_eve_demo(cfg)
    text = markdown(points, eve)
    print(text)
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "figures" / f"{NAME}.md").write_text(text, "utf-8")
    out = ROOT / "results" / f"{NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"config": asdict(cfg), "points": points, "linear_eve": eve}, indent=2), "utf-8"
    )
    ok = all(p["verified"] for p in points) and eve["eve_bits_with_leak"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
