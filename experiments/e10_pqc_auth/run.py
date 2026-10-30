"""E10: wariant E, czyli uwierzytelnianie postkwantowe i modele zagrożeń (F4.15–F4.21).

Wynik: figures/e10_pqc_auth.png, figures/e10_pqc_auth.md i results/e10_pqc_auth.json.
(a) scenariusze MITM (a)–(g) z F4.18 na tle przewidywań, w tym złamanie podpisu po sesji
    i kontrast z ML-KEM (harvest now, decrypt later) oraz hybrydą KDF(QKD ‖ ML-KEM);
(b) narzut uwierzytelniania w uczciwej rundzie: bity klucza, bajty łącza klasycznego, czas CPU;
    tag/podpis na transkrypt vs na wiadomość (decyzja z F4.16);
(c) czas podpisu i weryfikacji ML-DSA-44/65/87 na transkrypcie rundy;
(d) bilans klucza netto w funkcji L: Wegman–Carter zużywa klucz, ML-DSA nie;
(e) tabela modeli zagrożeń QKD + ML-DSA vs ML-KEM vs hybryda (F4.20).
"""

import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import mlkem

from qkdsim.analysis.key_rate import GYS, rate_vacuum_weak
from qkdsim.analysis.wcp import gain, qber
from qkdsim.attacks.mitm import MITMOutcome, SessionOutcome, run_mitm
from qkdsim.channel import FiberChannel
from qkdsim.config import load_yaml
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link, run_sifted
from qkdsim.postprocessing.auth import (
    MERSENNE_61,
    AuthenticatedChannel,
    Authenticator,
    KeyPool,
    WegmanCarter,
    message_bits,
)
from qkdsim.postprocessing.hybrid_key import hybrid_key
from qkdsim.postprocessing.leakage import linear_eve_knowledge
from qkdsim.postprocessing.pqc_auth import (
    CONTEXT,
    FIPS_204_SIZES,
    HybridAuth,
    generate,
    mldsa_auth,
    transcript_bytes,
)
from qkdsim.postprocessing.run import PostprocessingConfig, postprocess
from qkdsim.rng import Bits, RandomStreams
from qkdsim.source import WCPSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = HERE.name
P_D = dark_count_per_detector(GYS.background)
TAG_BITS = MERSENNE_61.bit_length()
TAG_BYTES = -(-TAG_BITS // 8)
FIXED_MESSAGES = 13  # jak w e09: próbka (3), Cascade (7), weryfikacja (2), PA (1)
Mode = Literal["message", "transcript"]


@dataclass(frozen=True)
class Config:
    seed: int
    mitm_length_km: float
    mitm_pulses: int
    levels: tuple[int, ...]
    cpu_repeats: int
    balance_round_pulses: tuple[int, ...]
    check_length_km: float
    check_pulses: int


@dataclass
class Timed:
    """`Authenticator` z pomiarem czasu uczciwych tagów i podpisów."""

    inner: Authenticator
    seconds: float = 0.0

    @property
    def tags_used(self) -> int:
        return self.inner.tags_used

    @property
    def key_bits_used(self) -> int:
        return self.inner.key_bits_used

    @property
    def bytes_sent(self) -> int:
        return self.inner.bytes_sent

    def genuine(self, sender: str, bits: Bits) -> None:
        start = time.perf_counter()
        self.inner.genuine(sender, bits)
        self.seconds += time.perf_counter() - start

    def impostor_passes(self, sender: str, bits: Bits, rng: np.random.Generator) -> bool:
        return self.inner.impostor_passes(sender, bits, rng)


def link(length_km: float) -> Link:
    return Link(
        WCPSource((0.48,)),
        FiberChannel(length_km, GYS.alpha_db_per_km, GYS.misalignment),
        DetectorPair(GYS.eta_bob, P_D),
    )


def messages_per_round(length_km: float, pulses: int) -> float:
    """Model z e09: wiadomości stałe + jedna wiadomość BINARY na każdy poprawiony błąd."""
    eta = GYS.eta(length_km)
    sifted = 0.5 * gain(0.48, eta, GYS.background) * pulses
    errors = 0.9 * sifted * qber(0.48, eta, GYS.background, GYS.misalignment)
    return FIXED_MESSAGES + errors


def thousands(value: float) -> str:
    """Liczba całkowita z odstępem jako separatorem tysięcy."""
    return f"{value:,.0f}".replace(",", " ")


def wegman_carter(seed: int) -> WegmanCarter:
    pool = np.random.default_rng([seed, 7]).integers(0, 2, 10**6, dtype=np.uint8)
    return WegmanCarter(KeyPool(pool))


def side_text(side: SessionOutcome) -> str:
    if side.result is not None:
        b = side.result.balance
        return f"klucz {b.final_length} bitów, QBER {100 * b.qber_sample:.2f}%".replace(".", ",")
    assert side.failure is not None
    return f"przerwanie przy wiadomości nr {side.failure.message_index} ({side.failure.kind})"


def mitm_scenarios(cfg: Config) -> list[tuple[str, str, bool, MITMOutcome]]:
    """(nazwa, przewidywanie, czy przewidywane przerwanie, wynik)."""
    mitm_link = link(cfg.mitm_length_km / 2)  # Ewa w połowie światłowodu
    run = partial(run_mitm, mitm_link, mitm_link, cfg.mitm_pulses, cfg.seed)
    pool = np.random.default_rng([cfg.seed, 7]).integers(0, 2, 10**6, dtype=np.uint8)

    def hybrid_broken() -> Authenticator:
        return HybridAuth(WegmanCarter(KeyPool(pool.copy())), mldsa_auth(65, "forgery_oracle"))

    return [
        ("(a) bez uwierzytelniania", "MITM udany", False, run()),
        ("(b) Wegman–Carter", "wykryty", True, run(pre_shared_bits=10**6)),
        ("(c) ML-DSA-65", "wykryty", True, run(authenticator=lambda: mldsa_auth(65))),
        (
            "(d) ML-DSA-65 złamane w trakcie sesji",
            "MITM udany",
            False,
            run(authenticator=lambda: mldsa_auth(65, "forgery_oracle")),
        ),
        (
            "(f) klucz Ewy w katalogu ofiary",
            "MITM udany",
            False,
            run(authenticator=lambda: mldsa_auth(65, "poisoned_directory")),
        ),
        (
            "(g) hybryda WC + ML-DSA-65, ML-DSA złamane w trakcie",
            "wykryty",
            True,
            run(authenticator=hybrid_broken),
        ),
    ]


@dataclass(frozen=True)
class AfterTheFact:
    final_bits: int
    disclosed_rank: int
    eve_known_bits: int  # (e): wiedza Ewy po wycieku klucza podpisu
    mlkem_recovered_bits: int  # HNDL: bity sekretu ML-KEM odzyskane po złamaniu
    mlkem_secret_bits: int
    hybrid_key_guessed: bool  # czy Ewa z sekretem ML-KEM odtwarza klucz hybrydowy


def after_the_fact(cfg: Config) -> AfterTheFact:
    """(e) ML-DSA złamane po sesji vs ML-KEM złamane po sesji vs hybryda KDF."""
    streams = RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, 3]))
    key = run_sifted(link(cfg.mitm_length_km / 2), cfg.mitm_pulses, streams)
    auth = mldsa_auth(65)
    result = postprocess(
        key, streams.public, PostprocessingConfig(record_rows=True), AuthenticatedChannel(auth)
    )
    # Po sesji Ewa dostaje klucze prywatne ML-DSA. Podpisy przyszłych sesji umie podrobić,
    # ale klucz tej sesji zależy tylko od tego, co podsłuchała w kanale klasycznym.
    eve = linear_eve_knowledge(result.channel, result.balance.reconciled, result.pa_seed)

    decapsulation_key = mlkem.MLKEM768PrivateKey.generate()
    secret, recorded = decapsulation_key.public_key().encapsulate()
    stolen = decapsulation_key.decapsulate(recorded)  # kraty złamane po latach
    agree = np.unpackbits(np.frombuffer(secret, np.uint8)) == np.unpackbits(
        np.frombuffer(stolen, np.uint8)
    )
    qkd_key = result.alice_key
    guess = hybrid_key(np.zeros_like(qkd_key), stolen)  # Ewa nie zna składowej QKD
    return AfterTheFact(
        result.balance.final_length,
        eve.disclosed_rank,
        eve.known_final_bits,
        int(agree.sum()),
        8 * len(secret),
        guess == hybrid_key(qkd_key, secret),
    )


@dataclass(frozen=True)
class Overhead:
    method: str
    mode: Mode
    tags: int
    key_bits: int
    bytes_sent: int
    cpu_ms: float
    final_bits: int


def overhead(cfg: Config) -> tuple[list[Overhead], Bits]:
    """Uczciwa runda z każdą metodą; zwraca też transkrypt Alice do pomiaru czasu."""
    streams = RandomStreams.from_seed(np.random.SeedSequence([cfg.seed, 4]))
    key = run_sifted(link(cfg.check_length_km), cfg.check_pulses, streams)
    methods: list[tuple[str, Mode, Callable[[], Authenticator]]] = [
        ("Wegman–Carter", "transcript", lambda: wegman_carter(cfg.seed)),
        ("Wegman–Carter", "message", lambda: wegman_carter(cfg.seed)),
        ("ML-DSA-65", "transcript", lambda: mldsa_auth(65)),
        ("ML-DSA-65", "message", lambda: mldsa_auth(65)),
        (
            "hybryda WC + ML-DSA-65",
            "transcript",
            lambda: HybridAuth(wegman_carter(cfg.seed), mldsa_auth(65)),
        ),
    ]
    rows = []
    transcript: Bits = np.zeros(0, dtype=np.uint8)
    mldsa_auth(65).genuine("alice", np.zeros(8, dtype=np.uint8))  # rozgrzewka OpenSSL
    for name, mode, factory in methods:
        auth = Timed(factory())
        public = np.random.default_rng([cfg.seed, 5])  # ta sama runda dla każdej metody
        result = postprocess(key, public, channel=AuthenticatedChannel(auth, mode=mode))
        rows.append(
            Overhead(
                name,
                mode,
                auth.tags_used,
                auth.key_bits_used,
                auth.bytes_sent,
                1e3 * auth.seconds,
                result.balance.final_length,
            )
        )
        sent = [message_bits(m) for m in result.channel.transcript if m.sender == "alice"]
        transcript = np.concatenate(sent)
    return rows, transcript


@dataclass(frozen=True)
class Timing:
    method: str
    public_key_bytes: int
    signature_bytes: int
    sign_us: float
    verify_us: float


def timings(cfg: Config, transcript: Bits) -> list[Timing]:
    """Mediana czasu na transkrypcie Alice z uczciwej rundy (na tej maszynie)."""
    data = transcript_bytes(transcript)
    rows = []
    wc = wegman_carter(cfg.seed)
    samples = []
    for _ in range(cfg.cpu_repeats):
        start = time.perf_counter()
        wc.polynomial_hash(transcript)
        samples.append(time.perf_counter() - start)
    hash_us = 1e6 * statistics.median(samples)
    rows.append(Timing("Wegman–Carter", 0, TAG_BYTES, hash_us, hash_us))
    for level in cfg.levels:
        private, public_bytes = generate(level)
        public = private.public_key()
        sign, verify = [], []
        for _ in range(cfg.cpu_repeats):
            start = time.perf_counter()
            signature = private.sign(data, CONTEXT)
            middle = time.perf_counter()
            public.verify(signature, data, CONTEXT)
            sign.append(middle - start)
            verify.append(time.perf_counter() - middle)
        rows.append(
            Timing(
                f"ML-DSA-{level}",
                len(public_bytes),
                len(signature),
                1e6 * statistics.median(sign),
                1e6 * statistics.median(verify),
            )
        )
    return rows


def threat_table(max_range_km: float, after: AfterTheFact) -> list[str]:
    ek = mlkem.MLKEM768PrivateKey.generate().public_key()
    ek_bytes = len(ek.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))
    ct_bytes = len(ek.encapsulate()[1])
    rows = [
        (
            "Harvest now, decrypt later (Mosca: x + y > z)",
            "odporne: podpis musi wytrzymać tylko czas sesji, a klucz nie zależy od założeń "
            f"obliczeniowych; po wycieku klucza podpisu Ewa zna {after.eve_known_bits} bitów (e)",
            f"podatne: nagrany szyfrogram ({ct_bytes} B) i późniejsze złamanie dają "
            f"{after.mlkem_recovered_bits}/{after.mlkem_secret_bits} bitów sekretu",
            "odporne, dopóki tajny jest klucz QKD",
        ),
        (
            "Przyszłe złamanie problemów kratowych",
            "przeszłe klucze bezpieczne (e); przyszłe sesje można uwierzytelniać "
            "Wegmanem–Carterem kluczem z poprzednich rund",
            "wszystkie nagrane sesje odkryte",
            "przeszłe klucze bezpieczne, zostaje składowa QKD",
        ),
        (
            "Złamanie podpisu w czasie rzeczywistym",
            "MITM udany (d); z hybrydą WC + ML-DSA wykryty (g)",
            "MITM udany: ML-KEM też wymaga uwierzytelnienia (np. ML-DSA w TLS)",
            "bezpieczna, jeśli część QKD uwierzytelnia hybryda WC + ML-DSA (g)",
        ),
        (
            "Ataki na implementację",
            "sprzęt: oślepianie (atak 2, e05), Trojan horse i time-shift (ataki 4, 5); "
            "obroną jest monitoring, a nie dowód",
            "oprogramowanie: kanały boczne, np. KyberSlash (czas dzielenia zależny od sekretu "
            "w implementacjach Kybera, 2024)",
            "trzeba złamać obie implementacje naraz",
        ),
        (
            "Zasięg i infrastruktura",
            f"R(L) > 0 do ok. {max_range_km:.0f} km (GYS, decoy vacuum+weak, e06); "
            "osobny światłowód i detektory",
            f"bez limitu zasięgu, zwykła sieć; {ek_bytes} B klucza publicznego i {ct_bytes} B "
            "szyfrogramu",
            "ograniczenia QKD plus koszt ML-KEM",
        ),
        (
            "Zaufane węzły pośrednie",
            "dalej niż zasięg: łańcuch węzłów zaufanych, z których każdy zna klucz",
            "end-to-end, bez węzłów zaufanych",
            "węzeł zaufany zna tylko składową QKD; sekret ML-KEM jest end-to-end",
        ),
    ]
    lines = [
        "| Zagrożenie | QKD + ML-DSA | ML-KEM-768 | Hybryda KDF(QKD ‖ ML-KEM) |",
        "| --- | --- | --- | --- |",
    ]
    return lines + [f"| {' | '.join(row)} |" for row in rows]


def main() -> int:
    cfg = load_yaml(HERE / "config.yaml", Config)
    scenarios = mitm_scenarios(cfg)
    after = after_the_fact(cfg)
    costs, transcript = overhead(cfg)
    times = timings(cfg, transcript)

    lines = [
        "## Scenariusze MITM (GYS, Ewa w połowie 25 km)",
        "",
        "| Scenariusz | Przewidywanie | Sesja Alice ↔ Ewa | Sesja Ewa ↔ Bob | Zgodne |",
        "| --- | --- | --- | --- | --- |",
    ]
    agreed = []
    for name, prediction, expect_abort, outcome in scenarios:
        ok = outcome.aborted is expect_abort
        agreed.append(ok)
        lines.append(
            f"| {name} | {prediction} | {side_text(outcome.alice_side)} | "
            f"{side_text(outcome.bob_side)} | {'tak' if ok else '**nie**'} |"
        )
    lines += [
        "",
        "## Złamanie po sesji: QKD a ML-KEM",
        "",
        "| Scenariusz | Co Ewa ma | Co Ewa wie o kluczu sesji |",
        "| --- | --- | --- |",
        f"| (e) ML-DSA-65 złamane po sesji | transkrypt i klucze prywatne ML-DSA | "
        f"{after.eve_known_bits} z {after.final_bits} bitów (ujawniony transkrypt to "
        f"{after.disclosed_rank} niezależnych funkcji liniowych klucza przed PA) |",
        f"| ML-KEM-768 złamane po sesji (HNDL) | szyfrogram i klucz dekapsulacji | "
        f"{after.mlkem_recovered_bits} z {after.mlkem_secret_bits} bitów sekretu |",
        f"| hybryda KDF(QKD ‖ ML-KEM), HNDL | sekret ML-KEM, bez klucza QKD | "
        f"{'cały klucz' if after.hybrid_key_guessed else 'nic: klucz KDF różny od zgadywania'} |",
        "",
        f"## Narzut uwierzytelniania w uczciwej rundzie ({cfg.check_length_km:g} km, "
        f"{thousands(cfg.check_pulses)} impulsów)",
        "",
        "| Metoda | Tag | Tagi i podpisy | Klucz zużyty [bit] | Łącze klasyczne [B] | CPU [ms] |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in costs:
        mode = "na transkrypt" if c.mode == "transcript" else "na wiadomość"
        lines.append(
            f"| {c.method} | {mode} | {thousands(c.tags)} | {thousands(c.key_bits)} | "
            f"{thousands(c.bytes_sent)} | {c.cpu_ms:.1f} |".replace(".", ",")
        )
    produced = costs[0].final_bits
    per_round, per_message = (
        next(c for c in costs if c.method == "ML-DSA-65" and c.mode == mode)
        for mode in ("transcript", "message")
    )
    lines += [
        "",
        f"Klucz wytworzony w tej rundzie: {thousands(produced)} bitów. Wegman–Carter liczy "
        "w kolumnie CPU jedno wyliczenie tagu (odbiorca powtarza je przy weryfikacji); ML-DSA "
        "podpis i weryfikację. Pierwsza runda Wegmana–Cartera zużywa dodatkowo 61 bitów na klucz "
        "haszu, potem tylko klucz jednorazowy na tag.",
        "",
        f"Decyzja z F4.16: podpis na transkrypt. Podpis na wiadomość kosztuje "
        f"{per_message.bytes_sent / per_round.bytes_sent:.0f} razy więcej bajtów "
        f"({thousands(per_message.bytes_sent)} B, czyli więcej niż sam klucz) i "
        f"{per_message.cpu_ms / per_round.cpu_ms:.0f} razy więcej czasu, a bezpieczeństwo "
        "jest takie samo: "
        "Ewa i tak zostaje wykryta przed użyciem klucza, bo klucz powstaje dopiero po "
        "sprawdzeniu transkryptu.",
        "",
        "## Czas na transkrypcie Alice z tej rundy (mediana, ta maszyna)",
        "",
        f"Transkrypt: {thousands(len(transcript))} bitów.",
        "",
        "| Metoda | Klucz publiczny [B] | Tag/podpis [B] | Tag/podpis [µs] | Weryfikacja [µs] |",
        "| --- | --- | --- | --- | --- |",
    ]
    for t in times:
        lines.append(
            f"| {t.method} | {t.public_key_bytes or '—'} | {t.signature_bytes} | "
            f"{t.sign_us:.0f} | {t.verify_us:.0f} |"
        )
    lines += [
        "",
        "Hasz Wegmana–Cartera jest tu pętlą Pythona po 60-bitowych kawałkach (schemat Hornera), "
        "więc jego czas mówi o implementacji, a nie o metodzie: w C albo w sprzęcie taki hasz "
        "liczy się z przepustowością rzędu GB/s. Czas ML-DSA pochodzi z OpenSSL przez "
        "`cryptography` i obejmuje skrót SHAKE256 całego transkryptu; podpis ma zmienny czas, bo "
        "losuje, aż przejdzie test odrzucenia (rejection sampling).",
    ]

    lengths = np.linspace(0.0, 200.0, 401)
    rates = np.array([max(rate_vacuum_weak(GYS, x, 0.48, 0.05), 0.0) for x in lengths])
    max_range = float(np.max(lengths[rates > 0]))
    wc_per_round = 2 * TAG_BITS
    lines += [
        "",
        "## Zasięg z dodatnim bilansem klucza na rundę",
        "",
        "| Impulsy na rundę N | Wegman–Carter (122 bity na rundę) | ML-DSA (0 bitów) |",
        "| --- | --- | --- |",
    ]
    break_even = {}
    for pulses in cfg.balance_round_pulses:
        positive = lengths[rates * pulses > wc_per_round]
        break_even[pulses] = float(np.max(positive)) if len(positive) else 0.0
        lines.append(
            f"| {thousands(pulses)} | {break_even[pulses]:.1f} km | {max_range:.1f} km |".replace(
                ".", ","
            )
        )
    lines += [
        "",
        "ML-DSA nie zużywa klucza, więc bilans jest dodatni wszędzie, gdzie R(L) > 0. Ceną jest "
        "założenie obliczeniowe, które musi stać w czasie sesji (scenariusz d), i kilobajty "
        "na rundę zamiast 16 B.",
        "",
        "## Modele zagrożeń (F4.20)",
        "",
        *threat_table(max_range, after),
    ]

    fig, (ax_k, ax_b, ax_t) = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for color, pulses in zip(("C0", "C1", "C2"), cfg.balance_round_pulses, strict=True):
        label = f"{pulses:.0e}".replace("e+0", "·10^")
        net_wc = rates * pulses - wc_per_round
        ax_k.semilogy(
            lengths,
            np.where(rates > 0, rates * pulses, np.nan),
            color=color,
            label=f"ML-DSA, N = {label}",
        )
        ax_k.semilogy(
            lengths,
            np.where(net_wc > 0, net_wc, np.nan),
            color=color,
            linestyle="--",
            label=f"Wegman–Carter, N = {label}",
        )
    ax_k.set(
        xlabel="długość światłowodu L [km]",
        ylabel="klucz netto na rundę [bit]",
        title="Klucz netto: WC zużywa 2 × 61 bitów na rundę, ML-DSA nic",
        ylim=(1, 1e6),
    )
    ax_k.legend(fontsize=7, loc="upper right")
    ax_k.grid(True, which="both", alpha=0.3)

    pulses = cfg.check_pulses
    messages = np.array([messages_per_round(x, pulses) for x in lengths])
    signature = FIPS_204_SIZES[65][1]
    mask = rates > 0
    curves = [
        ("Wegman–Carter, na transkrypt", np.full_like(lengths, 2 * TAG_BYTES), "C0", "-"),
        ("Wegman–Carter, na wiadomość", TAG_BYTES * messages, "C0", ":"),
        ("ML-DSA-65, na transkrypt", np.full_like(lengths, 2 * signature), "C3", "-"),
        ("ML-DSA-65, na wiadomość", signature * messages, "C3", ":"),
    ]
    for label, values, color, style in curves:
        ax_b.semilogy(lengths[mask], values[mask], color=color, linestyle=style, label=label)
    for c in costs:
        if c.method != "hybryda WC + ML-DSA-65":
            color = "C0" if c.method == "Wegman–Carter" else "C3"
            ax_b.plot(cfg.check_length_km, c.bytes_sent, "o", color=color)
    ax_b.plot([], [], "ko", label=f"zmierzone w rundzie ({cfg.check_length_km:g} km)")
    ax_b.set(
        xlabel="długość światłowodu L [km]",
        ylabel="bajty uwierzytelniania na rundę",
        title=f"Narzut łącza klasycznego, N = {thousands(pulses)}",
        ylim=(5, 1e8),
    )
    ax_b.legend(fontsize=7, loc="upper right")
    ax_b.grid(True, which="both", alpha=0.3)

    names = [t.method for t in times]
    x = np.arange(len(names))
    ax_t.bar(x - 0.2, [t.sign_us for t in times], 0.4, label="tag / podpis", color="C2")
    ax_t.bar(x + 0.2, [t.verify_us for t in times], 0.4, label="weryfikacja", color="C4")
    ax_t.set_xticks(x, names, fontsize=8)
    ax_t.set_yscale("log")
    ax_t.set(
        ylabel="czas [µs], mediana",
        title=f"CPU na transkrypcie {thousands(len(transcript))} bitów",
    )
    ax_t.legend(fontsize=8)
    ax_t.grid(True, which="both", axis="y", alpha=0.3)
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
                "mitm": {name: o.aborted for name, _, _, o in scenarios},
                "after_the_fact": asdict(after),
                "overhead": [asdict(c) for c in costs],
                "timings": [asdict(t) for t in times],
                "break_even_km": break_even,
                "max_range_km": max_range,
            },
            indent=2,
        ),
        "utf-8",
    )
    sizes_ok = all(
        (t.public_key_bytes, t.signature_bytes) == FIPS_204_SIZES[int(t.method[-2:])]
        for t in times[1:]
    )
    mldsa_transcript = next(c for c in costs if c.method == "ML-DSA-65" and c.mode == "transcript")
    passed = (
        all(agreed)
        and after.eve_known_bits == 0
        and after.mlkem_recovered_bits == after.mlkem_secret_bits
        and not after.hybrid_key_guessed
        and sizes_ok
        and mldsa_transcript.key_bits == 0
        and mldsa_transcript.bytes_sent == 2 * FIPS_204_SIZES[65][1]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
