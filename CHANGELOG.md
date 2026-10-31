# Historia zmian

Format na wzór [Keep a Changelog](https://keepachangelog.com/pl/1.1.0/), wersje według [SemVer](https://semver.org/lang/pl/).

## [Unreleased]

### Dodane

- Atak 4, time-shift (Qi et al. 2007, Zhao et al. 2008): krzywe sprawności detektorów w czasie (`EfficiencyMismatch`) dopasowane do Tabeli I(a) artykułu, atak `TimeShift`, model analityczny z granicami K_U i K_L (odtworzone 1131 i 1297 bitów z danych artykułu), obrona losowym przypisaniem bit ↔ detektor (`Link.random_assignment`), eksperyment e11.

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
