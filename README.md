# bb84-attacks

Symulator BB84 z atakami. Każdy wynik Monte Carlo jest porównywany ze wzorem analitycznym albo z wynikiem z publikacji.

Model: polaryzacyjny BB84, światłowód 1550 nm, źródło słabych impulsów koherentnych (WCP), dwa detektory SPAD po stronie Boba. Parametry startowe z eksperymentu GYS (Ma, Qi, Zhao, Lo 2005).

Pomost do kryptografii postkwantowej: kanał klasyczny można uwierzytelniać kluczem współdzielonym (Wegman–Carter), podpisem ML-DSA (FIPS 204) albo hybrydowo. Symulator pokazuje, kiedy takie uwierzytelnianie chroni klucz QKD, i porównuje QKD z ML-KEM w różnych modelach zagrożeń.

> **Status:** fazy F1 i F2 zakończone (bramki G1 i G2 zaliczone), w fazie F3 gotowe ataki 1 (intercept-resend) i 2 (oślepianie detektorów z obronami). Fizyka bez ataków zgadza się ze wzorami na 0–150 km, postprocessing daje identyczne klucze z poprawnie policzonym wyciekiem. Kolejne ataki: [ROADMAP.md](ROADMAP.md).

## Wyniki

![Zysk i QBER w funkcji długości światłowodu](figures/e01_gain_qber_vs_L.png)

Monte Carlo (punkty) na tle wzorów Ma et al. 2005 (linie) dla parametrów eksperymentu GYS: do 10⁹ impulsów na punkt, wszystkie punkty w granicach |z| < 1,8.

![Sprawność Cascade](figures/e02_cascade_efficiency.png)

Cascade (4 przebiegi) ujawnia 1,11–1,19 razy więcej bitów niż granica Shannona dla QBER 0,5–8%, w przedziale zakładanym w planie.

![Atak intercept-resend](figures/e04_intercept_resend.png)

Atak intercept-resend przy 25 km: QBER rośnie do ~26,5% (fizyka wykrywa podsłuch), a Ewa tuż za Alice zdradza się dodatkowo spadkiem zysku o 20%. Przy pełnym ataku Ewa zna połowę bitów po siftingu na pewno, a zgaduje 75%. QBER(p) dla źródła WCP leży poniżej prostej z planu, bo impulsy przechwycone są wykrywane rzadziej.

![Oślepianie detektorów](figures/e05_blinding.png)

Oślepianie detektorów (progi Clavis2 z Lydersena et al.): skalibrowana Ewa przy mocy impulsu 950–1300 µW zostawia QBER i zysk bez zmian, a zna 100% klucza. Losowe tłumienie 3 dB w 10% bramek wykrywa ją z z ≈ −35. Żadna moc impulsu nie przechodzi jednocześnie testu QBER i testu tłumienia. Krzywe ROC obu obron: [figures/e05_blinding_roc.png](figures/e05_blinding_roc.png).

[Bilans klucza dla GYS przy 25 i 50 km](figures/e03_key_balance.md): od 10⁸ impulsów do klucza końcowego, oraz pomiar wiedzy pasywnej Ewy, gdy wyciek z korekcji błędów zostanie (albo nie) odjęty.

Wszystkie wyniki odtwarza `make figures` (około 2,5 min na 12 rdzeniach).

## Szybki start

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -e '.[dev,crosscheck]'
```

```bash
make check
```

`make check` uruchamia lintera (ruff), sprawdzanie typów (mypy) i szybkie testy. Pozostałe cele: `make test-slow` (testy statystyczne), `make figures` (wszystkie wykresy z `experiments/`), `make report` (raport LaTeX).

## Dokumentacja

- [ROADMAP.md](ROADMAP.md): fazy, bramki jakości, harmonogram
- [Plan projektu](<assets/Symulator BB84 z atakami – plan projektu.md>): założenia i teoria
- [docs/model.md](docs/model.md): założenia fizyczne, konwencje, parametry
- [docs/architecture.md](docs/architecture.md): tor impulsów i punkty ataku P1–P8
- [docs/adr/](docs/adr/): decyzje architektoniczne
- [docs/related.md](docs/related.md): istniejące symulatory i czym ten projekt się od nich różni

## Licencja

MIT, patrz [LICENSE](LICENSE). Artykuły w `assets/` mają własne licencje i nie są częścią repozytorium.
