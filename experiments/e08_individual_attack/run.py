"""E08: granice informacyjne, atak indywidualny Fuchsa vs ataki ogólne.

Wynik: figures/e08_individual_attack.png i results/e08_individual_attack.json.
(a) I_AB i I_AE w funkcji błędu D: wzory (linie) i Monte Carlo klonera (punkty),
    intercept-resend jako odniesienie, przecięcie D* ≈ 14,64% i próg Shora–Preskilla 11,00%;
(b) stopa klucza: jednokierunkowa przeciw atakowi indywidualnemu, I_AB − I_AE, vs 1 − 2h(D)
    przeciw atakom ogólnym. Różnica obszarów to cena za bezpieczeństwo wobec ataków koherentnych.
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from qkdsim.analysis.entropy import h2
from qkdsim.analysis.individual import (
    FUCHS_CROSSING,
    bob_information,
    eve_information_individual,
    eve_information_intercept_resend,
)
from qkdsim.analysis.key_rate import shor_preskill_rate
from qkdsim.attacks.optimal_individual import OptimalIndividualAttack, mutual_information
from qkdsim.config import load_yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name


@dataclass(frozen=True)
class Config:
    seed: int
    errors: tuple[float, ...]
    photons: int


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    points = []
    for i, error in enumerate(cfg.errors):
        rng = np.random.default_rng([cfg.seed, i])
        alice, _bases, bob, eve = OptimalIndividualAttack(error).sample(cfg.photons, rng)
        points.append(
            {
                "error": error,
                "qber": float(np.mean(alice != bob)),
                "i_ab": mutual_information(alice, bob),
                "i_ae": mutual_information(alice, eve),
                "i_ab_formula": bob_information(error),
                "i_ae_formula": eve_information_individual(error),
            }
        )
    shor_preskill = brentq(shor_preskill_rate, 0.01, 0.3)

    grid = np.linspace(0.0, 0.3, 601)
    fig, (ax_i, ax_r) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    ax_i.plot(
        100 * grid, [bob_information(d) for d in grid], color="C0", label="I$_{AB}$ = 1 − h(D)"
    )
    ax_i.plot(
        100 * grid,
        [eve_information_individual(d) for d in grid],
        color="C3",
        label="I$_{AE}$, atak Fuchsa (optymalny indywidualny)",
    )
    ir = grid[grid <= 0.25]
    ax_i.plot(
        100 * ir,
        [eve_information_intercept_resend(d) for d in ir],
        color="C3",
        linestyle="--",
        label="I$_{AE}$, intercept-resend",
    )
    ax_i.plot(
        [100 * p["qber"] for p in points],
        [p["i_ab"] for p in points],
        "o",
        color="C0",
        markersize=4,
    )
    ax_i.plot(
        [100 * p["qber"] for p in points],
        [p["i_ae"] for p in points],
        "o",
        color="C3",
        markersize=4,
        label="Monte Carlo klonera",
    )
    labels = (
        (shor_preskill, "Shor–Preskill", "right", -0.3),
        (FUCHS_CROSSING, "I$_{AB}$ = I$_{AE}$", "left", 0.3),
    )
    for x, label, align, shift in labels:
        ax_i.axvline(100 * x, color="0.4", linestyle=":", linewidth=1)
        ax_i.text(100 * x + shift, 0.04, f"{label}\n{100 * x:.2f}%", fontsize=8, ha=align)
    ax_i.set(
        xlabel="błąd u Boba D [%]",
        ylabel="informacja [bit na bit po siftingu]",
        title="Informacja Boba i Ewy",
        ylim=(0, 1.02),
    )
    ax_i.legend(fontsize=8, loc="upper right")

    individual = [bob_information(d) - eve_information_individual(d) for d in grid]
    general = [1 - 2 * h2(d) for d in grid]
    ax_r.plot(100 * grid, individual, color="C3", label="atak indywidualny: I$_{AB}$ − I$_{AE}$")
    ax_r.plot(100 * grid, general, color="C0", label="ataki ogólne (Shor–Preskill): 1 − 2h(D)")
    ax_r.fill_between(
        100 * grid,
        np.maximum(general, 0),
        np.maximum(individual, 0),
        color="C3",
        alpha=0.15,
        label="tylko wobec ataku indywidualnego",
    )
    ax_r.axhline(0, color="k", linewidth=0.8)
    ax_r.set(
        xlabel="błąd u Boba D [%]",
        ylabel="stopa klucza na bit po siftingu (f = 1)",
        title="Ataki koherentne są silniejsze niż indywidualne",
        ylim=(-0.2, 1.0),
    )
    ax_r.legend(fontsize=8)
    for ax in (ax_i, ax_r):
        ax.grid(True, alpha=0.3)
    photons = f"{cfg.photons:,}".replace(",", " ")
    fig.suptitle(
        f"Optymalny atak indywidualny (Fuchs et al. 1997); punkty: {photons} fotonów na punkt"
    )
    path = ROOT / "figures" / f"{NAME}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    worst = 0.0
    for p in points:
        n, d = cfg.photons, p["error"]
        if 0 < d < 0.5:
            worst = max(worst, abs(p["qber"] - d) / np.sqrt(d * (1 - d) / n))
        print(
            f"D = {100 * d:5.1f}%: I_AB {p['i_ab']:.4f} ({p['i_ab_formula']:.4f})  "
            f"I_AE {p['i_ae']:.4f} ({p['i_ae_formula']:.4f})"
        )
    print(
        f"przecięcie D* = {100 * FUCHS_CROSSING:.4f}%, Shor–Preskill = {100 * shor_preskill:.4f}%"
    )
    print(f"max |z| QBER = {worst:.2f}")
    out = ROOT / "results" / f"{NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "points": points,
                "shor_preskill": shor_preskill,
                "crossing": FUCHS_CROSSING,
            },
            indent=2,
        ),
        "utf-8",
    )
    return 0 if worst < 4.0 else 1


if __name__ == "__main__":
    sys.exit(main())
