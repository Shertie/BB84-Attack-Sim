# Historia zmian

Format na wzór [Keep a Changelog](https://keepachangelog.com/pl/1.1.0/), wersje według [SemVer](https://semver.org/lang/pl/).

## [Unreleased]

## [1.2.0] – 2026-10-31

Wszystkie rozszerzenia z fazy F7 i audyt definicji ukończenia dla ośmiu ataków.

### Dodane

- Obrony z progiem i krzywe ROC dla ataków 1, 4, 5 i 6 (eksperyment e17): test progu QBER z dokładną ROC dwumianową (`defenses/qber_test.py`), monitor czasu przyjścia kliknięć (`defenses/timing_monitor.py`), fotodioda na wejściu Alice z największą niezauważoną sondą i wymaganą izolacją (`defenses/input_monitor.py`).

- Czas martwy i afterpulsy detektorów (F1.8, F7): `DetectorPair(dead_time_gates, afterpulse)`, wzór odnowienia na stacjonarne p-stwo kliknięcia, model dokładny toru z afterpulsami (`analysis/wcp.afterpulse_dark_count`), porównanie ze wzorami Lima et al. 2014, eksperyment e18.

- Symulator QRNG na dzielniku wiązki (F7, przedłużenie ataku 8): `qrng.BeamSplitterQrng` z niedopasowaniem i pamięcią detektorów, dokładne obciążenie bitów, min-entropia metodą najczęstszej wartości z NIST SP 800-90B z kontekstem, ekstrakcja Toeplitzem, `QrngSource` jako generator Alice; testy zdrowia RCT i APT z progami zgodnymi z Tabelą 2 normy (`defenses/health_tests.py`), eksperyment e19.

- Uzgadnianie LDPC z syndromem (F7): kody o stałym stopniu zmiennych, dekoder BP, ramki równej długości ze skracaniem i odrzucaniem niezdekodowanych ramek, `PostprocessingConfig(reconciliation="ldpc")`, wiersze syndromu w analizie liniowej Ewy, ewolucja gęstości z progiem BP kodu (3,6) 0,0841 (literatura: 0,084), eksperyment e20.

- Skończony klucz dla WCP ze stanami decoy (Lim et al. 2014, F7): wzory (1)–(5) z liczników (`postprocessing/finite_key_decoy.py`), oczekiwane liczniki dla modelu z artykułu i dla detektorów symulatora, optymalizacja parametrów protokołu, test krzyżowy z `qkd` (zgodność do 10⁻¹²), eksperyment e14. Odtworzone 135 km przy n_X = 10⁴ i stosunek ≈ 1,75 stóp przy 100 km.

- MDI-QKD (Lo, Curty, Qi 2012, F7): przekaźnik z pomiarem Bella i dokładną fizyką stanów koherentnych z losową fazą (`mdi.py`), model analityczny zysków, QBER i pary pojedynczych fotonów (`analysis/mdi.py`, zgodny ze wzorami Ma i Razaviego w `qkd`), stopa z wzoru (1), Ewa jako przekaźnik (`MeasureAndAnnounce`), eksperyment e15. Zasięg 207,5 km przy parametrach z artykułu.

- QKD ze splątanymi parami (F7): BBM92 ze źródłem SPDC (dokładny model wielu par, `entangled.py`, `analysis/entangled.py`, zgodny z Ma, Fungiem i Lo w `qkd`), test CHSH bez fair sampling, stopa device-independent (Acín i in. 2007) i progi sprawności 82,8% oraz 92,3%, fałszywe naruszenie CHSH przez sterowanie detektorami (Gerhardt i in. 2011), eksperyment e16.

### Zmienione

- `BiasedSource` przeniesiony do `rng.py` (uczciwy protokół asymetryczny); `attacks.weak_rng` nadal go udostępnia.
- `KeyBalance` ma pole `discarded` (bity z niezdekodowanych ramek LDPC); przy Cascade zawsze 0.
- e07 korzysta z progu z `defenses/qber_test.py`; wyniki identyczne co do bajtu.
- ROADMAP: audyt definicji ukończenia (rozdz. 7.1) dla ataków 1–8 zamiast szablonu; README i raport (mapa ataków, podrozdziały F7); `docs/model.md` rozdz. 14–17.
- Bez nowych opcji (czas martwy, afterpulsy, QRNG, LDPC) wyniki są identyczne jak w 1.1.0.

## [1.1.0] – 2026-10-31

Komplet ośmiu ataków z planu (faza F6).

### Dodane

- Atak 4, time-shift (Qi et al. 2007, Zhao et al. 2008): krzywe sprawności detektorów w czasie (`EfficiencyMismatch`) dopasowane do Tabeli I(a) artykułu, atak `TimeShift`, model analityczny z granicami K_U i K_L (odtworzone 1131 i 1297 bitów z danych artykułu), obrona losowym przypisaniem bit ↔ detektor (`Link.random_assignment`), eksperyment e11.
- Atak 5, Trojan horse (Gisin et al. 2006): model wycieku sondy przez izolację i odbicie, wiedza Ewy o bicie (Helstrom, licznik fotonów przy losowej fazie) i o bazie (Helstrom dla stanów mieszanych, weryfikacja w QuTiP), atak `TrojanHorse` w torze impulsów, wymagana izolacja w funkcji wycieku, eksperyment e12.
- Atak 8, słaby generator Alice: MT19937 o znanym ziarnie, obciążenie i korelacje Markowa jako `RandomSource`; Ewa odtwarza generator (atak pasywny lub intercept-resend w odtworzonej bazie) albo mierzy w częstszej bazie; deficyt entropii i rozrzedzanie korelacji przez sifting; testy losowości (monobit, runs, autokorelacja) z analizą mocy, eksperyment e13.

### Zmienione

- Raport (12 stron): podrozdziały o atakach 4, 5 i 8, pełna mapa ataków, wartości kontrolne z Zhao et al. 2008, Gisina et al. 2006 i NIST SP 800-22.
- README: mapa ataków ze wszystkimi ośmioma atakami.
- `make figures` z czystego klonu: 13 eksperymentów w 464 s na 12 rdzeniach; poza czasami CPU w e10 wykresy identyczne z repozytorium.
- `InterceptResend` przyjmuje opcjonalny predyktor baz, `intercept_resend_model` p-stwo trafienia bazy; detektory mają opcjonalne krzywe sprawności w czasie, a Bob losowe przypisanie bit ↔ detektor. Bez tych opcji wyniki są identyczne jak w 1.0.0.

## [1.0.0] – 2026-10-30

Pierwsza wersja publiczna (MVP z ROADMAP.md, fazy F0–F5).

### Model i walidacja

- Tor Monte Carlo BB84 na poziomie impulsów: źródło WCP z decoy, światłowód, para detektorów SPAD z ciemnymi zliczeniami i trybem liniowym (Clavis2), sifting. Dwie warstwy symulacji: bitów (pełne zapisy) i liczników (do 10⁹ impulsów na punkt, równolegle).
- Postprocessing: estymacja QBER z próbki, Cascade, weryfikacja haszem, wzmacnianie prywatności macierzą Toeplitza, liczenie wycieku i dokładna wiedza liniowej Ewy nad GF(2).
- Decoy vacuum+weak z danych Monte Carlo, stopa GLLP, skończony klucz według Tomamichela et al. 2012.
- Walidacja względem Ma et al. 2005 (GYS) na 0–150 km i krzyżowo względem pakietu `qkd` 0.2.0 (ADR 0007).

### Ataki i obrony

- Atak 1, intercept-resend, z Ewą w dowolnym punkcie światłowodu (e04).
- Atak 2, oślepianie detektorów, z obronami: losowe tłumienie i monitor mocy z krzywymi ROC (e05).
- Atak 3, photon-number splitting, i test spójności decoy (e06).
- Atak 6, optymalny atak indywidualny (Fuchs et al. 1997), zweryfikowany w QuTiP (e08).
- Atak 7, man-in-the-middle, i uwierzytelnianie Wegmana–Cartera (e09).
- Mapa „atak → złamane założenie dowodu → obrona → metryka” w README i raporcie.

### Wariant E: kryptografia postkwantowa

- Wspólny interfejs `Authenticator`: Wegman–Carter, podpis ML-DSA (FIPS 204, `cryptography`, zgodność z `dilithium-py`), hybryda obu.
- Scenariusze MITM (a)–(g): złamanie podpisu w trakcie i po sesji, zatruty katalog kluczy, hybryda.
- Hybrydowy klucz KDF(klucz QKD ‖ sekret ML-KEM) i kontrast „harvest now, decrypt later” z ML-KEM-768 (e10).
- Tabela modeli zagrożeń QKD + ML-DSA vs ML-KEM vs hybryda.

### Infrastruktura

- Dziesięć eksperymentów `experiments/e01`–`e10`; każdy kończy się błędem, gdy wynik wyjdzie poza tolerancję. `make figures` z czystego klonu: 365 s na 12 rdzeniach.
- 375 szybkich i 55 wolnych testów (statystyczne z-score, krzyżowe z `qkd` i QuTiP), pokrycie gałęzi rdzenia 95%, `mypy --strict`, ruff.
- CI: lint, typy i pokrycie na Pythonie 3.11–3.13; nocne testy wolne; `make figures` z czystego klonu przy tagu.
- Raport LaTeX (10 stron) w `report/`.

### Znane ograniczenia

- Skończony klucz tylko dla źródła jednofotonowego; WCP z decoy (Lim et al. 2014) w v1.1.
- Twierdzenia Tomamichela et al. dają stopy o 8–14% niższe niż Tabela 1 artykułu (ROADMAP, K18).
- Ataki 4 (time-shift), 5 (Trojan horse) i 8 (słaby generator) w v1.1.
