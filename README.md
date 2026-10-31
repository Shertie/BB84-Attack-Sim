# bb84-attacks

Symulator Monte Carlo BB84 na poziomie impulsów, w którym ataki są wykonywane, a nie wyceniane wzorem. Każdy wynik jest porównany ze wzorem lub publikacją, od fotonu do klucza końcowego, a obrony są oceniane krzywymi ROC.

Model: polaryzacyjny BB84, światłowód 1550 nm, źródło słabych impulsów koherentnych (WCP), dwa detektory SPAD po stronie Boba, Cascade, hasz Toeplitza i skończony klucz. Parametry z eksperymentu GYS (Ma, Qi, Zhao, Lo 2005). Wzory sprawdzamy też niezależną implementacją [`qkd`](https://pypi.org/project/qkd/), która wycenia część tych samych ataków w formie zamkniętej ([porównanie](docs/related.md)).

Pomost do kryptografii postkwantowej (wariant E): kanał klasyczny uwierzytelniany Wegmanem–Carterem, podpisem ML-DSA (FIPS 204) albo hybrydowo, z tabelą modeli zagrożeń QKD wobec ML-KEM.

> **Status: v1.0, w drodze do v1.1.** Ataki 1 (intercept-resend), 2 (oślepianie detektorów), 3 (PNS), 6 (optymalny atak indywidualny) i 7 (MITM) z obronami, skończony klucz i uwierzytelnianie postkwantowe; po v1.0 doszły ataki 4 (time-shift), 5 (Trojan horse) i 8 (słaby generator). Do v1.1 zostało domknięcie mapy, raportu i release: [ROADMAP.md](ROADMAP.md). Raport: [report/main.tex](report/main.tex) (`make report`).

## Wyniki kontrolne

![Zysk i QBER w funkcji długości światłowodu](figures/e01_gain_qber_vs_L.png)

Zysk i QBER z Monte Carlo (do 10⁹ impulsów na punkt) zgadzają się ze wzorami Ma et al. 2005 na 0–150 km, wszystkie punkty w granicach |z| < 1,8.

![Atak intercept-resend](figures/e04_intercept_resend.png)

Pełny intercept-resend przy 25 km daje QBER 26,5–26,8%: 25% z samego ataku plus błąd optyki e_d (wzór ¼ + ½·e_d daje 26,65%).

![PNS i stany decoy](figures/e06_decoy_pns.png)

Stopa klucza R(L) jak u Ma et al. 2005: GLLP bez decoy kończy się przy ~40 km, decoy vacuum+weak przy 140,6 km (punkty: estymata decoy z Monte Carlo).

![Atak indywidualny i ataki ogólne](figures/e08_individual_attack.png)

I_AB = I_AE przy D\* = 14,64% dla optymalnego ataku indywidualnego (Fuchs et al. 1997, zweryfikowany w QuTiP), a próg Shora–Preskilla dla ataków ogólnych to 11,00%.

Wszystkie wykresy odtwarza `make figures` (około 5 min na 12 rdzeniach). Każdy skrypt w `experiments/` kończy się błędem, jeśli jego wynik wyjdzie poza tolerancję.

## Mapa ataków

Każdy atak, który nie podnosi QBER, łamie któreś założenie dowodu bezpieczeństwa, a nie wzór na długość klucza. Założenia Z1–Z8 opisuje [docs/model.md](docs/model.md), rozdz. 5.

| Atak | Status | Złamane założenie | Obrona | Metryka (wynik) |
| --- | --- | --- | --- | --- |
| 1. Intercept-resend | v1.0, [e04](figures/e04_intercept_resend.png) | żadne: atak mieści się w modelu dowodu; przy małej próbce Z8 | estymacja QBER z progiem 11%; skończony klucz (Tomamichel et al. 2012) | QBER 26,5% przy pełnym ataku; przechwycenie 4% impulsów przy próbce 1000 bitów wykryte w 23% przypadków ([e07](figures/e07_finite_key.png)) |
| 2. Oślepianie detektorów | v1.0, [e05](figures/e05_blinding.png) | Z3: detektor w trybie jednofotonowym | losowe tłumienie 3 dB w 10% bramek; monitor mocy | QBER i zysk bez zmian, Ewa zna 100% klucza; test tłumienia z ≈ −35; [ROC](figures/e05_blinding_roc.png) |
| 3. Photon-number splitting | v1.0, [e06](figures/e06_decoy_pns.png) | Z1: rozkład liczby fotonów uwzględniony w długości klucza | stany decoy (vacuum+weak) | od 32,8 km Ewa zna 98–100% klucza przy niezmienionym QBER i zysku; decoy przerywa protokół |
| 6. Optymalny atak indywidualny | v1.0, [e08](figures/e08_individual_attack.png) | Z7: granica dla ataków indywidualnych nie jest granicą ogólną | próg Shora–Preskilla 11,00% zamiast 14,64% | I_AB = I_AE przy D\* = 14,64%; numpy i QuTiP zgodne do 10⁻¹² |
| 7. Man-in-the-middle | v1.0, [e09](figures/e09_mitm.md) | Z5a: kanał klasyczny uwierzytelniony | Wegman–Carter z kluczem współdzielonym | przerwanie przy pierwszym tagu Ewy; fałszerstwo z p-stwem 1/p ≈ 2⁻⁶¹; 122 bity klucza na rundę |
| 7′. MITM przy podpisie ML-DSA | v1.0, [e10](figures/e10_pqc_auth.md) | Z5a w trakcie sesji; Z5b: autentyczne klucze publiczne | hybryda Wegman–Carter + ML-DSA; weryfikacja odcisku klucza | ML-DSA złamane na żywo lub podmieniony klucz: MITM udany; hybryda: wykryty; złamanie po sesji: 0 bitów klucza dla Ewy |
| 4. Time-shift | v1.1, [e11](figures/e11_time_shift.md) | Z3: sprawność niezależna od czasu przyjścia | losowe przypisanie bit ↔ detektor | QBER i zysk bez zmian; Ewa zna 0,31 bitu na wykryty bit przy krzywych Zhao et al. 2008; obrona: 0,000 ± 0,000 |
| 5. Trojan horse | v1.1, [e12](figures/e12_trojan_horse.md) | Z2: brak kanałów bocznych u Alice | izolacja, losowa faza odbitego światła, monitor mocy | QBER i zysk bez zmian; przy 0,1 fotonu w odbitej sondzie Ewa zna 0,135 bitu (0,095 z losową fazą); wyciek ≤ 10⁻³ przy sondzie 10⁹ fotonów i modulatorze (−20 dB) wymaga 102 dB izolacji |
| 8. Słaby generator | v1.1, [e13](figures/e13_weak_rng.md) | Z6: losowe i prywatne wybory | QRNG; testy losowości łapią obciążenie i korelacje, nie przewidywalność | znane ziarno: Ewa zna 100% klucza bez dotykania światłowodu, a testy odrzucają generator w 1% przypadków (= α); bazy obciążone 0,9: QBER 8%, Ewa zna 90% bitów |

## Pozostałe wyniki

![Sprawność Cascade](figures/e02_cascade_efficiency.png)

Cascade (4 przebiegi) ujawnia 1,11–1,19 razy więcej bitów niż granica Shannona dla QBER 0,5–8%. [Bilans klucza dla GYS przy 25 i 50 km](figures/e03_key_balance.md): od 10⁸ impulsów do klucza końcowego, z pomiarem wiedzy pasywnej Ewy, gdy wyciek z korekcji błędów zostanie (albo nie) odjęty.

Oślepianie detektorów (progi Clavis2 z Lydersena et al. 2010): skalibrowana Ewa przy mocy impulsu 950–1300 µW zostawia QBER i zysk bez zmian. Żadna moc impulsu nie przechodzi jednocześnie testu QBER i testu tłumienia.

![Skończony klucz](figures/e07_finite_key.png)

Skończony klucz (Tomamichel et al. 2012): stopa rośnie z blokiem n w stronę granicy 1 − 2h(Q). Uproszczenie z planu (Hoeffding na samej próbce) zawyża długość klucza o 95% przy n = 3000 i o 15% przy 10⁵.

![MITM i uwierzytelnianie](figures/e09_mitm.png)

Atak MITM: bez uwierzytelniania Ewa zna klucze Alice i Boba przy QBER jak bez ataku. Tag Wegmana–Cartera na każdą wiadomość zjada więcej klucza, niż QKD wytwarza; tag na transkrypt rundy kosztuje 122 bity.

![Uwierzytelnianie postkwantowe](figures/e10_pqc_auth.png)

Wariant E ([scenariusze i tabela modeli zagrożeń](figures/e10_pqc_auth.md)): ML-DSA nie zużywa klucza, więc przy 10⁶ impulsów na rundę bilans jest dodatni do ~140 km zamiast 60 km. Ceną są 2 × 3309 B na rundę i założenie obliczeniowe, które musi stać w trakcie sesji. Nagrany szyfrogram ML-KEM-768 oddaje po złamaniu krat cały sekret, a złamanie podpisu po sesji QKD nic Ewie nie daje.

![Atak time-shift](figures/e11_time_shift.png)

Atak time-shift ([tabele](figures/e11_time_shift.md)): z krzywymi sprawności dopasowanymi do Zhao et al. 2008 odtwarzamy ich Tabelę I(c) (K_U = 1130,9 i K_L = 1296,8 bitu wobec 1131 i 1297). W łączu GYS przy 75 km Ewa zna 0,31 bitu na wykryty bit przy niezmienionym QBER i zysku. Atak łamie bezpieczeństwo (K_L > K_U) od rozsunięcia krzywych 89 ps przy μ = 0,1, ale dopiero od 140 ps przy μ = 0,48, bo GLLP i tak usuwa część wielofotonową. Losowe przypisanie bit ↔ detektor zeruje wiedzę Ewy.

![Atak Trojan horse](figures/e12_trojan_horse.png)

Atak Trojan horse ([tabele](figures/e12_trojan_horse.md)): odtwarzamy liczby Gisina et al. 2006 (0,135 i 0,095 bitu przy 0,1 fotonu wracającym do Ewy), a Monte Carlo w łączu GYS potwierdza, że QBER i zysk się nie zmieniają. Bazę w czasie rzeczywistym (do intercept-resend bez śladu w QBER) Ewa zgaduje dużo gorzej, gdy Alice losuje fazę: bez odniesienia fazowego pojedynczy foton nic o bazie nie mówi. Każdy rząd wielkości mocy sondy to 10 dB izolacji więcej, a losowa faza oszczędza 1,6 dB.

![Słaby generator losowy](figures/e13_weak_rng.png)

Słaby generator Alice ([tabele](figures/e13_weak_rng.md)): MT19937 o znanym ziarnie przechodzi testy monobit, runs i autokorelacji jak prawdziwy generator, a Ewa, odtwarzając go, zna cały klucz przy niezmienionym QBER i zysku. Obciążenie baz pozwala na intercept-resend z mniejszym QBER (pod progiem 11% od p ≈ 0,84). Obciążenie bitów 0,55 kosztuje 3,3% klucza asymptotycznie, ale w rachunku min-entropii ponad połowę. Korelacje Markowa w bitach sifting prawie wymazuje: przy 25 km zostaje 0,3% impulsów.

## Szybki start

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -e '.[dev,crosscheck,verify]'
```

```bash
make check
```

```bash
make figures
```

`make check` uruchamia lintera (ruff), sprawdzanie typów (mypy --strict) i szybkie testy. Pozostałe cele: `make test-slow` (testy statystyczne i krzyżowe), `make coverage` (pokrycie rdzenia), `make report` (raport LaTeX).

## Dokumentacja

- [ROADMAP.md](ROADMAP.md): fazy, bramki jakości, harmonogram
- [Plan projektu](<assets/Symulator BB84 z atakami – plan projektu.md>): założenia i teoria
- [docs/model.md](docs/model.md): założenia fizyczne i dowodu, konwencje, parametry
- [docs/architecture.md](docs/architecture.md): tor impulsów i punkty ataku P1–P8
- [docs/adr/](docs/adr/): decyzje architektoniczne
- [docs/related.md](docs/related.md): istniejące symulatory i czym ten projekt się od nich różni
- [CHANGELOG.md](CHANGELOG.md): historia wersji

## Cytowanie i licencja

Metadane do cytowania: [CITATION.cff](CITATION.cff). Licencja MIT, patrz [LICENSE](LICENSE). Artykuły w `assets/` mają własne licencje i nie są częścią repozytorium.
