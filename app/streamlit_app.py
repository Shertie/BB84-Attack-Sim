"""Interaktywny symulator BB84 z atakami (Streamlit).

Uruchomienie: `make app` albo `streamlit run app/streamlit_app.py`. Logika scenariuszy jest
w `qkdsim.scenarios`; ten plik zawiera tylko interfejs.
"""

import math
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from qkdsim.scenarios import (
    ATTACKS,
    BLINDING,
    INTERCEPT_RESEND,
    NONE,
    PNS,
    TIME_SHIFT,
    TROJAN,
    WEAK_RNG,
    Attack,
    Outcome,
    Settings,
    final_key,
    key_rates,
    simulate,
    time_shift_min_length,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
QBER_LIMIT = 0.11  # próg Shora–Preskilla

ABOUT = {
    NONE: "Uczciwe łącze GYS: zysk i QBER zgodne z modelem Ma et al. 2005.",
    INTERCEPT_RESEND: "Ewa mierzy ułamek impulsów w losowej bazie i odsyła wynik. Mieści się "
    "w modelu dowodu: zdradza ją QBER (¼ + ½·e_d przy pełnym ataku).",
    BLINDING: "Ewa oślepia detektory Boba ciągłym światłem i steruje nimi jasnymi impulsami "
    "(progi Clavis2, Lydersen et al. 2010). Łamie założenie o detektorze jednofotonowym: "
    "QBER i zysk bez zmian, Ewa zna cały klucz. Obrona: losowe tłumienie przed detektorami.",
    PNS: "Ewa blokuje impulsy jednofotonowe i zabiera po fotonie z wielofotonowych, dobierając "
    "zysk tak, by był jak bez ataku. Łamie założenie o znanym rozkładzie liczby fotonów; "
    "obroną są stany decoy (e06).",
    TIME_SHIFT: "Ewa przesuwa impulsy w czasie tak, by klikał głównie jeden detektor (krzywe "
    "sprawności z Zhao et al. 2008). QBER bez zmian. Obrona: Bob losowo zamienia przypisanie "
    "bit ↔ detektor.",
    TROJAN: "Ewa sonduje modulator Alice jasnym światłem i odczytuje ustawienie z odbicia "
    "(Gisin et al. 2006). Sygnału nie dotyka. Obrona: izolacja i losowa faza.",
    WEAK_RNG: "Generator Alice wybiera bazę X z p-stwem p ≠ ½, a Ewa mierzy zawsze w częstszej "
    "bazie. Im większe obciążenie, tym mniejszy QBER przy pełnym intercept-resend.",
}


def percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%".replace(".", ",")


def number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def scientific(value: float) -> str:
    return f"{value:.3e}".replace(".", ",")


@st.cache_data(show_spinner=False)
def cached_simulation(settings: dict[str, object], attack: dict[str, object]) -> Outcome:
    return simulate(Settings(**settings), Attack(**attack))  # type: ignore[arg-type]


@st.cache_data(show_spinner=False)
def cached_key(
    settings: dict[str, object], attack: dict[str, object], method: str
) -> dict[str, object]:
    balance = final_key(Settings(**settings), Attack(**attack), method)  # type: ignore[arg-type]
    return {**asdict(balance), "efficiency": balance.efficiency}


@st.cache_data(show_spinner=False)
def cached_rates(mu: float, nu: float) -> dict[str, list[float]]:
    return key_rates(mu, nu, [float(x) for x in np.linspace(0.0, 160.0, 161)])


def sidebar() -> tuple[Settings, Attack]:
    st.sidebar.header("Łącze (parametry GYS)")
    attack_name = st.sidebar.selectbox("Atak", ATTACKS)
    minimum = time_shift_min_length(0.48) if attack_name == TIME_SHIFT else 0.0
    length = st.sidebar.slider(
        "Długość światłowodu L [km]",
        0.0,
        150.0,
        75.0 if attack_name == TIME_SHIFT else 25.0,
        1.0,
        format="%.0f",
    )
    mu = st.sidebar.slider("Średnia liczba fotonów μ", 0.05, 1.0, 0.48, 0.01, format="%.2f")
    pulses = st.sidebar.select_slider(
        "Liczba impulsów",
        options=[100_000, 300_000, 1_000_000, 3_000_000, 10_000_000],
        value=3_000_000 if attack_name == TIME_SHIFT else 1_000_000,
        format_func=lambda n: f"{n:,}".replace(",", " "),
    )
    seed = int(st.sidebar.number_input("Ziarno", min_value=0, value=1, step=1))

    params: dict[str, object] = {"name": attack_name}
    if attack_name != NONE:
        st.sidebar.header("Parametry ataku")
    if attack_name == INTERCEPT_RESEND:
        params["fraction"] = st.sidebar.slider(
            "Ułamek przechwytywanych impulsów p", 0.0, 1.0, 1.0, 0.01
        )
        params["position"] = st.sidebar.slider(
            "Położenie Ewy (0 = przy Alice, 1 = przy Bobie)", 0.0, 1.0, 1.0, 0.05
        )
    elif attack_name == BLINDING:
        params["trigger_uw"] = st.sidebar.slider(
            "Moc impulsu wyzwalającego [µW]", 300.0, 2200.0, 1100.0, 10.0,
            help="Doskonały atak w oknie 932–1294 µW (Clavis2).",
        )  # fmt: skip
        params["calibrated"] = st.sidebar.checkbox("Ewa kompensuje obrót polaryzacji", True)
        params["attenuator"] = st.sidebar.checkbox("Obrona: losowe tłumienie 3 dB w 10% bramek")
    elif attack_name == TIME_SHIFT:
        st.sidebar.caption(f"Atak wymaga łącza ≥ {number(minimum, 1)} km przy μ = 0,48.")
        params["random_assignment"] = st.sidebar.checkbox(
            "Obrona: losowe przypisanie bit ↔ detektor"
        )
    elif attack_name == TROJAN:
        params["returned_photons"] = st.sidebar.select_slider(
            "Fotony wracające do Ewy μ_out",
            options=[0.001, 0.01, 0.1, 0.3, 1.0, 3.0],
            value=0.1,
        )
        params["random_phase"] = st.sidebar.checkbox("Obrona: Alice losuje fazę")
    elif attack_name == WEAK_RNG:
        params["basis_bias"] = st.sidebar.slider(
            "P-stwo bazy X w generatorze Alice", 0.5, 1.0, 0.8, 0.01
        )
    return Settings(length, mu, int(pulses), seed), Attack(**params)  # type: ignore[arg-type]


def attack_tab(settings: Settings, attack: Attack) -> None:
    st.markdown(ABOUT[attack.name])
    try:
        with st.spinner("Monte Carlo..."):
            outcome = cached_simulation(asdict(settings), asdict(attack))
    except ValueError as error:
        st.error(str(error))
        return

    gain_col, qber_col, eve_col = st.columns(3)
    gain_col.metric(
        "Zysk (detekcje na impuls)",
        scientific(outcome.gain),
        f"{100 * (outcome.gain / outcome.honest_gain - 1):+.1f}% wobec braku ataku".replace(
            ".", ","
        ),
        delta_color="off",
    )
    qber_col.metric(
        "QBER",
        percent(outcome.qber),
        f"{100 * (outcome.qber - outcome.honest_qber):+.2f} pp wobec braku ataku".replace(".", ","),
        delta_color="inverse",
    )
    eve_col.metric("Informacja Ewy [bit na bit po siftingu]", number(outcome.eve))

    if outcome.defense_alarm:
        st.success(f"Obrona wykrywa atak. {outcome.defense}")
    elif outcome.qber > QBER_LIMIT:
        st.warning(
            f"QBER {percent(outcome.qber, 1)} przekracza próg {percent(QBER_LIMIT, 0)}: "
            "Alice i Bob przerywają protokół."
        )
    elif outcome.eve > 0.05:
        st.error(
            f"QBER {percent(outcome.qber, 1)} jest poniżej progu, a Ewa wie "
            f"{number(outcome.eve, 2)} bitu na bit: atak przechodzi niezauważony."
        )
    if outcome.defense and not outcome.defense_alarm:
        st.info(outcome.defense)

    def sigma(value: float, digits: int) -> str:
        return "—" if math.isnan(value) else f"± {value:.{digits}f}".replace(".", ",")

    rows = {
        "Wielkość": ["Zysk", "QBER", "Informacja Ewy"],
        "Monte Carlo": [
            f"{scientific(outcome.gain)} {sigma(outcome.gain_sigma, 6)}",
            f"{percent(outcome.qber)} {sigma(100 * outcome.qber_sigma, 2)} pp",
            f"{number(outcome.eve)} {sigma(outcome.eve_sigma, 3)}",
        ],
        "Model analityczny": [
            "—" if outcome.model_gain is None else scientific(outcome.model_gain),
            percent(outcome.model_qber),
            number(outcome.model_eve),
        ],
        "Bez ataku (model)": [scientific(outcome.honest_gain), percent(outcome.honest_qber), "0"],
    }
    st.table(rows, hide_index=True)
    st.caption(
        "Monte Carlo w warstwie liczników; model to dokładna wartość oczekiwana dla tego samego "
        "ataku. Różnice rzędu σ są statystyczne. Zysk z tłumikiem Boba nie ma modelu (—)."
    )


def key_tab(settings: Settings, attack: Attack) -> None:
    st.markdown(
        "Pełny postprocessing na kluczu z tego samego łącza i ataku: próbka 10% do estymacji "
        "QBER, korekcja błędów, weryfikacja haszem i wzmacnianie prywatności (GLLP z granicą "
        "decoy dla uczciwego łącza)."
    )
    method = st.radio("Korekcja błędów", ["cascade", "ldpc"], horizontal=True,
                      format_func=lambda m: "Cascade" if m == "cascade" else "LDPC")  # fmt: skip
    try:
        with st.spinner("Klucz..."):
            balance = cached_key(asdict(settings), asdict(attack), method)
    except ValueError as error:
        st.error(str(error))
        return
    left, middle, right = st.columns(3)
    left.metric("Bity po siftingu", f"{balance['sifted']:,}".replace(",", " "))
    middle.metric("QBER z próbki", percent(float(balance["qber_sample"])))  # type: ignore[arg-type]
    right.metric("Klucz końcowy [bity]", f"{balance['final_length']:,}".replace(",", " "))
    st.table(
        {
            "Krok": [
                "Próbka do estymacji",
                "Klucz do korekcji",
                "Błędy przed korekcją",
                "Wyciek w korekcji błędów",
                "Sprawność korekcji f",
                "Odrzucone ramki LDPC (bity)",
                "Błędy po korekcji",
                "Weryfikacja",
                "Klucz końcowy",
            ],
            "Wartość": [
                str(balance["sample"]),
                str(balance["reconciled"]),
                str(balance["errors_before_ec"]),
                str(balance["leak_ec"]),
                number(float(balance["efficiency"]), 3),  # type: ignore[arg-type]
                str(balance["discarded"]),
                str(balance["residual_errors"]),
                "zgodne" if balance["verified"] else "niezgodne: klucz odrzucony",
                str(balance["final_length"]),
            ],
        },
        hide_index=True,
    )
    undefended = (
        (attack.name == BLINDING and not attack.attenuator)
        or attack.name in (PNS, TROJAN)
        or (attack.name == TIME_SHIFT and not attack.random_assignment)
    )
    if attack.name == BLINDING and attack.attenuator:
        st.info("Z testem losowego tłumienia protokół zostałby przerwany przed użyciem klucza.")
    elif undefended and balance["final_length"]:
        st.error(
            "Klucz powstał, choć Ewa zna jego część: ten atak nie podnosi QBER, więc rachunek "
            "długości klucza go nie widzi. Potrzebna jest osobna obrona (zakładka „Atak i obrona”)."
        )


def range_tab() -> None:
    st.markdown(
        "Stopa klucza asymptotycznie dla łącza GYS (bity na wysłany impuls). GLLP bez decoy "
        "liczy się z optymalnym μ dla każdej długości, pozostałe krzywe z μ i ν z suwaków."
    )
    left, right = st.columns(2)
    mu = left.slider("μ sygnału", 0.05, 1.0, 0.48, 0.01, key="range_mu")
    nu = right.slider("ν słabego decoy", 0.01, 0.3, 0.05, 0.01, key="range_nu")
    lengths = [float(x) for x in np.linspace(0.0, 160.0, 161)]
    rates = cached_rates(mu, nu)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, values in rates.items():
        positive = [(x, r) for x, r in zip(lengths, values, strict=True) if r > 0]
        if positive:
            ax.semilogy(*zip(*positive, strict=True), label=name)
    ax.set(xlabel="L [km]", ylabel="R [bit/impuls]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    reach = {
        name: max((x for x, r in zip(lengths, values, strict=True) if r > 0), default=0.0)
        for name, values in rates.items()
    }
    st.caption(
        "Zasięg (ostatni punkt siatki co 1 km z R > 0): "
        + ", ".join(f"{name} {x:.0f} km" for name, x in reach.items())
    )


def results_tab() -> None:
    st.markdown("Wykresy i tabele z eksperymentów (`make figures`).")
    pngs = sorted(FIGURES.glob("e*.png"))
    if not pngs:
        st.warning("Brak wykresów w figures/: uruchom `make figures`.")
        return
    choice = st.selectbox("Eksperyment", pngs, format_func=lambda p: p.stem)
    st.image(str(choice), width="stretch")
    table = FIGURES / f"{choice.stem.removesuffix('_roc')}.md"
    if table.exists():
        st.markdown(table.read_text("utf-8"))


def main() -> None:
    st.set_page_config(page_title="Symulator BB84 z atakami", layout="wide")
    st.title("Symulator BB84 z atakami")
    st.caption(
        "Monte Carlo na poziomie impulsów: ataki są wykonywane, a nie wyceniane wzorem. "
        "Parametry łącza z eksperymentu GYS (Ma, Qi, Zhao, Lo 2005)."
    )
    settings, attack = sidebar()
    tabs = st.tabs(["Atak i obrona", "Klucz końcowy", "Zasięg", "Wyniki eksperymentów"])
    with tabs[0]:
        attack_tab(settings, attack)
    with tabs[1]:
        key_tab(settings, attack)
    with tabs[2]:
        range_tab()
    with tabs[3]:
        results_tab()


main()
