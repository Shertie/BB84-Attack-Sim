# Model fizyczny, konwencje i założenia

Dokument referencyjny dla kodu w `src/qkdsim/`. Zmiana modelu = zmiana tego pliku w tym samym commicie.

## 1. Zakres

- Protokół: BB84 z kodowaniem polaryzacyjnym, bazy Z (H/V) i X (D/A).
- Kanał: światłowód 1550 nm, straty wykładnicze w długości, stały błąd optyki.
- Źródło: słabe impulsy koherentne (WCP), opcjonalnie z kilkoma intensywnościami (decoy).
- Warianty protokołu: MDI-QKD z niezaufanym przekaźnikiem (rozdz. 12) i BBM92 ze splątanymi parami (rozdz. 13).
- Bob: **aktywny** wybór bazy (modulator przed jednym PBS), dwa detektory SPAD w trybie bramkowanym. Przy biernym wyborze bazy (dzielnik wiązki i dwa PBS) zmienia się m.in. moc potrzebna do ataku oślepiania (Lydersen et al. 2010). Taki wariant to osobna decyzja.

## 2. Konwencje

| Wielkość | Konwencja |
| --- | --- |
| Stany polaryzacji | kąt θ: H = 0°, V = 90°, D = 45°, A = 135°; bit 0 = H lub D, bit 1 = V lub A |
| Pomiar | PBS ustawiony pod kątem φ bazy; foton trafia do detektora 0 z p-stwem cos²(θ − φ) |
| Bazy | 0 = Z, 1 = X |
| Długość | km |
| Tłumienie | dB/km (α), dB (straty skupione, izolacja) |
| Moc | W |
| Prawdopodobieństwa | nazwa parametru mówi, czego dotyczą: `*_per_pulse`, `*_per_gate`, `*_per_detector` |
| Losowość | wybory Alice i Boba przez wstrzykiwany `RandomSource`; przyroda (liczba fotonów, wynik pomiaru) i Ewa mają osobne generatory. Wszystkie cztery strumienie pochodzą z jednego ziarna (`RandomStreams.from_seed`); żadnych globalnych `np.random.*` |
| Wyniki | każdy eksperyment zapisuje konfigurację, ziarno, hash commita i wersje bibliotek |

## 3. Parametry startowe: eksperyment GYS

Źródło: Ma, Qi, Zhao, Lo 2005, Tabela I, oraz tekst artykułu; eksperyment: Gobby, Yuan, Shields 2004.

| Parametr | Symbol | Wartość | Uwagi |
| --- | --- | --- | --- |
| Długość fali | λ | 1550 nm | |
| Tłumienie | α | 0,21 dB/km | |
| Sprawność po stronie Boba | η_Bob | 0,045 | **zawiera** sprawność detektora; nie mnożyć drugi raz przez η_d |
| Tło (ciemne zliczenia) | Y₀ | 1,7 × 10⁻⁶ na impuls | **łączne dla obu detektorów**; na detektor p_d = 1 − √(1 − Y₀) ≈ Y₀/2 |
| Błąd optyki | e_d | 0,033 | stały w funkcji L |
| Błąd tła | e₀ | 0,5 | |
| Sprawność korekcji błędów | f | 1,22 | z tekstu artykułu, **nie** z Tabeli I |
| Częstotliwość powtarzania | — | 2 MHz | to jest kolumna „f” w Tabeli I; do przeliczenia R na bit/s |
| Wydajność siftingu | q | 0,5 | |
| Intensywność sygnału | μ | 0,48 | optimum dla f = 1,22 (0,54 dla f = 1) |
| Intensywność decoy | ν | 0,05 | wariant vacuum+weak |

Wartości kontrolne wynikające z tych parametrów: `tests/golden.py` i Tabela 1 raportu.

## 4. Model kroków toru

1. **Źródło:** liczba fotonów n ~ Poisson(μ_i) dla intensywności wybranej w danym impulsie. n jest zapisywane, bo potrzebuje go atak PNS, ale Alice i Bob go nie znają.
2. **Kanał:** liczba ocalałych fotonów ~ Binomial(n, 10^(−αL/10)). Błąd optyki to obrót polaryzacji o kąt θ_e z sin²θ_e = e_d.
3. **Bob:** wybór bazy, PBS (φ = 0 dla Z, π/4 dla X), rozkład fotonów na detektory ~ Binomial(n, cos²(θ − φ)). Prawdopodobieństwo liczy się jako (1 + cos 2(θ − φ))/2, co daje dokładnie 0 i 1 dla stanów z bazy analizatora; wprost liczone cos² zostawiłoby resztę rzędu 10⁻³³. Dalej: każdy foton wykryty z prawdopodobieństwem η, a do tego niezależne ciemne zliczenia p_d na detektor i bramkę.
4. **Klasyfikacja zdarzeń:** brak kliknięcia, pojedyncze, podwójne. Podwójne kliknięcie dostaje losowy bit i nie jest odrzucane (squashing, Beaudry, Moroder, Lütkenhaus 2008).
5. **Rozszerzenia detektora:** tryb liniowy (oślepianie), krzywe sprawności w czasie η_i(t) (time-shift), czas martwy i afterpulsing (rozdz. 14).

Własność do testów: dla stanu koherentnego liczby fotonów w dwóch wyjściach PBS są niezależnymi rozkładami Poissona o średnich μηcos² i μηsin². Próbkowanie przez n i rozkład dwumianowy musi dawać ten sam rozkład.

### Implementacja i model analityczny

- Detektor z k fotonami po PBS klika z p-stwem 1 − (1 − p_d)(1 − η)^k: jedno losowanie na detektor i impuls, dokładnie równoważne dwumianowi i ciemnemu zliczeniu.
- Ciemne zliczenia na detektor: p_d = 1 − √(1 − Y₀), żeby łączne tło obu detektorów wynosiło Y₀.
- `analysis/wcp.py` ma dwa modele:
  - wzory Ma et al. (sumy (7) i (8)), na których stoją stopy klucza;
  - dokładną wartość oczekiwaną dla detektorów symulatora (niezależne rozkłady Poissona w obu detektorach, podwójne kliknięcie z losowym bitem).

  Zysk jest w obu modelach identyczny. QBER różni się o mniej niż 3·10⁻⁵ względnie dla GYS na 0–200 km. MC jest walidowane względem modelu dokładnego, ale przechodzi też test względem wzoru z artykułu.
- Wzór Ma et al. na E_μ może przekroczyć ½ dla e_d > (1 − Y₀)/2, bo licznik błędów nie odejmuje nakładania sygnału i tła, a zysk je odejmuje. Model dokładny nigdy.

## 5. Założenia dowodu bezpieczeństwa

Mapa „atak → złamane założenie → obrona → metryka” z wynikami jest w README. Kolumna „Atak” poniżej mówi, które założenie łamie który atak; potwierdzają to eksperymenty e04–e13.

| # | Założenie | Atak, który je łamie | Obrona |
| --- | --- | --- | --- |
| Z1 | Rozkład liczby fotonów źródła jest znany i uwzględniony w długości klucza (GLLP / decoy) | 3, PNS (gdy liczymy klucz tak, jakby impulsy były jednofotonowe) | stany decoy |
| Z2 | Moduł Alice nie ujawnia ustawień poza wysłanym stanem (brak kanałów bocznych) | 5, Trojan horse | izolatory, monitor mocy, filtr |
| Z3 | Detektory Boba działają w trybie jednofotonowym, a ich sprawność nie zależy od bitu, bazy ani czasu przyjścia | 2, oślepianie; 4, time-shift | monitor mocy/fotoprądu, losowa sprawność, losowe przypisanie bit ↔ detektor, MDI |
| Z4 | Pomiar Boba jest opisany modelem squashing (podwójne kliknięcia dostają losowy bit) | odrzucanie podwójnych kliknięć jako „optymalizacja” implementacji | losowy bit zamiast odrzucenia |
| Z5a | Kanał klasyczny jest uwierzytelniony, a uwierzytelnienie nie zostaje złamane **w trakcie** sesji (później już nie musi wytrzymać) | 7, MITM; MITM z fałszerstwem podpisu w czasie rzeczywistym | Wegman–Carter z kluczem współdzielonym, ML-DSA lub hybryda |
| Z5b | Klucze publiczne stron są autentyczne (tylko przy uwierzytelnianiu podpisem) | 7, MITM z podmianą klucza publicznego w katalogu | zaufany katalog / PKI, weryfikacja odcisku klucza |
| Z6 | Wybory bitów, baz i intensywności są prawdziwie losowe i prywatne | 8, słaby generator | QRNG; testy losowości wykrywają obciążenie, ale nie przewidywalność |
| Z7 | Ewa jest ograniczona tylko prawami fizyki; wszystkie błędy przypisujemy jej | 6, granica dla ataków indywidualnych (Fuchs) vs ogólnych (Shor–Preskill) | odpowiednio konserwatywny próg QBER |
| Z8 | Parametry są estymowane ze skończonej próbki z jawnym ε | „Ewa w szumie”: mały ułamek przechwytu ginie w fluktuacjach | rachunek skończonego klucza (Tomamichel et al. 2012) |

Wspólny wniosek: atak, który nie podnosi QBER (2, 4, 5, 7, 8), łamie jedno z założeń Z2–Z6, a nie wzór na długość klucza. Do tej listy dochodzi PNS (3): za 32,8 km Ewa zna prawie cały klucz przy niezmienionym QBER i zysku, bo łamie Z1.

Przy Z5a widać różnicę między QKD a ML-KEM. QKD potrzebuje uwierzytelnienia, które jest bezpieczne tylko w chwili sesji. Złamanie ML-DSA po fakcie nie odsłania wcześniejszych kluczy. Klucz z ML-KEM traci bezpieczeństwo, gdy w dowolnym momencie w przyszłości padną problemy kratowe.

## 6. Znane uproszczenia

- e_d i Y₀ nie zależą od L ani od czasu.
- Brak dyspersji i dryfu polaryzacji w kanale.
- Czas martwy i afterpulsy są opcjonalne i domyślnie wyłączone (rozdz. 14).
- Skończony klucz: Tomamichel et al. 2012 dla źródła jednofotonowego i Lim et al. 2014 dla WCP z decoy (rozdz. 9); bez skończonego klucza dla MDI i device-independent.
- Uzgadnianie: Cascade albo LDPC z kodami regularnymi (rozdz. 16).
- Generator Alice to domyślnie generator pseudolosowy numpy; QRNG jest modelem (rozdz. 15), a nie fizycznym źródłem.

## 7. Postprocessing

- **Losowość publiczna:** próbka do estymacji, permutacje Cascade i ziarna haszy pochodzą ze strumienia `RandomStreams.public`. Jest jawny, bo i tak jest ogłaszany przez kanał klasyczny, a osobny strumień nie przesuwa losowań stron.
- **Wyciek:** liczy się jeden bit na każdą parzystość ogłoszoną przez Alice w Cascade (bloki i połówki w BINARY) oraz bity hasza weryfikacyjnego. Parzystości Boba mówią tylko, gdzie są błędy, więc nie są liczone osobno. Bazy i ujawniona próbka nie ujawniają nic o reszcie klucza.
- **Hasz:** macierz Toeplitza T[i, j] = s[i − j + n − 1]; ta sama rodzina 2-uniwersalna służy do weryfikacji (64 bity, ε_cor = 2⁻⁶⁴) i do wzmacniania prywatności.
- **Długość klucza (asymptotycznie):** ℓ = ⌊n·s·[1 − h(e_ph)] − leak_EC − leak_ver⌋. Dla źródła jednofotonowego s = 1 i e_ph to błąd fazowy. Dla WCP s = Q₁/Q_μ i e_ph = e₁ (GLLP). Skończony klucz: rozdz. 9.
- **Wiedza pasywnej Ewy:** dokładnie dim(span P ∩ span T) bitów klucza końcowego, gdzie P to ujawnione funkcje liniowe, a T macierz wzmacniania prywatności. Dlatego zapomniany wyciek daje Ewie co najmniej ℓ − (n − rank P) bitów bez żadnego ataku na kanał kwantowy.

## 8. Ataki w światłowodzie

- **Położenie Ewy:** parametr `position` ∈ [0, 1] dzieli światłowód na odcinek do Ewy (bez błędu optyki) i od Ewy do Boba (z błędem optyki). Złożenie odcinków ma tę samą transmisję co cały kanał. Błąd optyki jest po stronie Boba, więc Ewa mierzy stan Alice bez obrotu.
- **Wiedza Ewy:** `EveLog` (impulsy zmierzone, baza, wynik) jest osobny od rekordów Alice i Boba. `Counts` liczy wśród wykrytych impulsów te zmierzone przez Ewę, te z trafnym wynikiem i te, które zna na pewno (mierzyła w bazie Alice).
- **Intercept-resend (atak 1):** Ewa mierzy w losowej bazie idealnym detektorem i wysyła 1 foton; impuls bez fotonów w jej punkcie zostaje próżnią. Model analityczny w `analysis/intercept_resend.py` jest dokładną wartością oczekiwaną dla symulatora (mieszanina: impuls nieprzechwycony, próżnia u Ewy, foton w bazie Alice, foton w bazie sprzężonej).
- **QBER(p) dla WCP nie jest liniowe:** prosta e_d + p(¼ − e_d/2) jest dokładna dla p ∈ {0, 1}, a pomiędzy obowiązuje tylko przy μT₁ ≪ 1.
- **Oślepianie (atak 2):** detektor ma tryb liniowy (`LinearMode`, progi Clavis2 z Lydersena et al. 2010): ciągłe światło ≥ `blind_power_w` wyłącza tryb Geigera i ciemne zliczenia. Od tej chwili klika tylko impuls klasyczny, z p-stwem rosnącym liniowo między P_never a P_always (założenie modelu). Nieoślepiony detektor traktuje jasny impuls jak P·τ/hν fotonów, więc po prostu klika. Ewa mierzy tuż za Alice, wysyła impulsy wyzwalające bezstratną linią i dobiera ich częstość tak, by zysk był jak bez ataku. Odtwarza też spodziewany QBER, odwracając część bitów.
- **Obrót a okno ataku:** stały obrót polaryzacji (błąd optyki) dzieli jasny impuls w bazie sprzężonej w proporcji ½ ± √(e_d(1 − e_d)). Bez kompensacji okno doskonałego ataku znika.
- **Obrony:** monitor mocy (odczep + szum, próg na odczycie) i losowe tłumienie przed detektorami (test z: odsetek kliknięć w bramkach tłumionych vs oczekiwany czynnik). Tłumik nie zmienia stanu oślepienia, bo ten zależy od średniego fotoprądu.
- **PNS (atak 3):** Ewa tuż za Alice mierzy liczbę fotonów (QND), blokuje impulsy jednofotonowe, z wielofotonowych zabiera 1 foton (zna bit po ujawnieniu baz), a resztę wysyła bezstratnie. Strategia (`pns_strategy`) dobiera blokowanie i przepuszczanie tak, by zysk sygnału był jak bez ataku. Od L_kryt = 32,8 km (GYS, μ = 0,48) wystarcza do tego samo przepuszczanie impulsów wielofotonowych. Ewa nie widzi intensywności, więc decoy dostaje tę samą strategię i jego zysk wychodzi niespójny.
- **Time-shift (atak 4):** sprawność detektora j zależy od czasu przyjścia impulsu: η_j(t) = η·exp(−[(t − c_j)² − c_j²]/(2σ_j²)), więc w nominalnym czasie t = 0 oba detektory mają η (`EfficiencyMismatch`). Krzywe dopasowane do Tabeli I(a) Zhao et al. 2008: Δ = c₁ − c₀ = 100 ps z tekstu artykułu, równe szczyty (założenie), środki i szerokości z czterech liczb detekcji po odjęciu ciemnych zliczeń (wychodzi σ = 150 i 140 ps). Ewa tuż za Alice przesuwa impuls do A z p-stwem p_A (równowaga zer i jedynek Boba), inaczej do B, i wyrównuje zysk linią bezstratną z tłumieniem. QBER od błędu optyki nie zależy od niedopasowania (błędne i poprawne detekcje skalują się tą samą sprawnością).
- **Obrona przed time-shift:** Bob w każdej bramce losuje, czy obrócić polaryzację o 90° przed PBS, i odwraca odczyt tam, gdzie obrócił (`Link.random_assignment`). Przesunięcie Ewy faworyzuje detektor, a nie bit, więc jej informacja spada do zera.
- **Ograniczenia ataku 4:** krzywe to Gaussy, a nie zmierzony profil bramki; przegląd po Δ używa rodziny symetrycznej (równe szerokości 145 ps), bo dopasowane krzywe mają różne szerokości i przy małym Δ oba przesunięcia faworyzowałyby ten sam detektor.
- **Trojan horse (atak 5):** Ewa wpuszcza do Alice sondę o μ_in fotonach; wraca stan koherentny o μ_out = μ_in·10^(−A/10), A = izolacja + odbicie + 2 × tłumik Alice, z polaryzacją ustawioną przez modulator (najgorszy przypadek: modulator działa na sondę tak samo jak na sygnał). Sygnału Ewa nie dotyka. Sondę trzyma do ujawnienia baz; wynik jej optymalnego pomiaru losujemy z dokładnego rozkładu: Helstrom dla dwóch stanów koherentnych (stała faza) albo licznik fotonów (Alice losuje fazę), wzory (1) i (7) Gisina et al. 2006.
- **Baza z sondy:** Helstrom dla stanów mieszanych ρ_Z = ½(ρ_H + ρ_V), ρ_X = ½(ρ_D + ρ_A) w przestrzeni Focka dwóch modów polaryzacji, obciętej na łącznej liczbie fotonów (ogon Poissona < 10⁻¹²); weryfikacja w QuTiP z innym obcięciem. Bez odniesienia fazowego jeden foton nie niesie informacji o bazie (przewaga ~ μ²/8 zamiast ~ 0,27√μ).
- **Ograniczenia ataku 5:** nie modelujemy filtrów spektralnych ani czasu otwarcia modulatora; wpływ sondy na źródło (np. zmiana μ w systemach dwukierunkowych) pomijamy. Z Lucamarinim et al. 2015 porównujemy budżet izolacji: sonda ograniczona progiem uszkodzenia światłowodu, μ_in = N/f_A (`lidt_probe_photons`), daje ich Tabelę I co do decybela (170, 200 i 230 dB przy 1 GHz, 1 MHz i 1 kHz dla μ_out = 10⁻⁶). Ich stopy klucza pod atakiem (wzór (6)) nie liczymy.
- **Słaby generator (atak 8):** generator Alice to wstrzykiwany `RandomSource`: MT19937 o znanym ziarnie, obciążone bity i bazy albo symetryczne łańcuchy Markowa. Ewa odtwarza znany generator (`ReplayPredictor`, te same wywołania w tej samej kolejności co `WCPSource.emit`) i albo czyta cały surowy klucz bez dotykania światłowodu, albo robi intercept-resend w odtworzonej bazie. Przy obciążonych bazach mierzy zawsze w częstszej (`intercept_resend_model` z `right_basis`).
- **Deficyt entropii:** wzmacnianie prywatności zakłada jednostajne bity. Przy obciążeniu p brakuje 1 − h(p) bitu na bit (asymptotycznie), w rachunku min-entropii 1 + log₂ max(p, 1 − p). Sifting zostawia impulsy w odstępach geometrycznych, więc korelacja Markowa λ = 2s − 1 spada w kluczu do rλ/(1 − (1 − r)λ).
- **Testy losowości:** monobit i runs według NIST SP 800-22 (z warunkiem wstępnym testu serii) oraz autokorelacja z przesunięciem 1; moc z przybliżenia normalnego, sprawdzona Monte Carlo. Testy łapią obciążenie i korelacje, nie przewidywalność.
- **Decoy z danych:** `postprocessing/decoy.py` liczy Y₁^L, e₁^U i R z liczników MC (vacuum+weak, asymptotycznie). `defenses/decoy_check.py` testuje spójność: z Q_μ i Y₀ wyznacza η̂ i przewiduje Q_ν. Szum estymaty decoy jest duży: słaby decoy daje mało detekcji (σ(E_ν) ≈ 12% względnie przy 10⁸ impulsach i 50 km), a e₁^U i R go dziedziczą.

## 9. Skończony klucz

- **Rachunek:** `postprocessing/finite_key.py` implementuje twierdzenie Tomamichela et al. 2012 w dwóch wydrukowanych wersjach (tekst główny i Methods) oraz uproszczenie (Hoeffding na samej próbce) do porównania. Protokół asymetryczny: n bitów klucza, k bitów testu, z góry ustalony próg Q_tol, przerwanie, gdy QBER z próbki go przekroczy. M(n, k) = (√n + √k)².
- **Robustness:** p-stwo przerwania uczciwego przebiegu liczone jest dokładnie z ogona rozkładu dwumianowego.
- **Zakres Tomamichela et al.:** źródło jednofotonowe.
- **WCP z decoy:** `postprocessing/finite_key_decoy.py` implementuje wzory (1)–(5) Lima et al. 2014: asymetryczny wybór baz (q dla bazy klucza), trzy intensywności μ₁ > μ₂ + μ₃, μ₂ > μ₃ ≥ 0, granice s₀, s₁ i v₁ z poprawką Hoeffdinga √(n/2 · ln(21/ε_sec)) oraz granica błędu fazowego z funkcją γ. Każdy krok zgadza się z `qkd` 0.2.0 do 10⁻¹². `analysis/finite_key_decoy.py` liczy oczekiwane liczniki dla modelu z artykułu (z afterpulsami) albo dla detektorów symulatora i optymalizuje (q, p₁, p₂, μ₁, μ₂) ewolucją różnicową. Na danych Monte Carlo rachunek działa przez `counts_from_simulation`; bazy wybiera `BiasedSource` z rng.py, więc przebieg jest jednoprocesowy.
- **Rozbieżności z tekstem arXiv v1:** η_sys zamiast η_ch w modelu błędu, leak_EC jako liczba bitów i ε_sec = 10⁻¹⁰ zamiast κℓ: z η_ch błąd optyki wychodzi ok. 10 razy za duży, a z ε_sec = κℓ zasięg przy n_X = 10⁴ to ok. 128 km zamiast 135 km z tekstu. Dokumentacja `qkd` powołuje się na Tupkary et al. (arXiv:2502.10340), według których granica (5) nie jest ścisła; nie weryfikowaliśmy tego.

## 10. Atak indywidualny

- **Konstrukcja:** optymalny atak indywidualny to fazowo-kowariantny kloner (częściowa zamiana o kąt η w podprzestrzeni {|01⟩, |10⟩}) w układzie obróconym o 120° wokół (1, 1, 1), tak by bazy Z i X leżały na równiku. Daje on D = sin²(η/2) u Boba i błąd Ewy ½ − √(D(1 − D)), czyli dokładnie granicę Fuchsa, dla wszystkich czterech stanów BB84.
- **Weryfikacja:** QuTiP buduje tę samą operację niezależnie, z generatora i operatorów Pauliego. Wierności Boba i Ewy wynikają ze śladów częściowych, a prawdopodobieństwa wyników z projektorów.
- **Zakres:** atak działa na pojedynczych fotonach i idealnych detektorach (granica teoretyczna), więc ma własny sampler zamiast toru impulsów.

## 11. Uwierzytelnianie i MITM

- **Wegman–Carter:** tag = (h_k(m) + r) mod p, gdzie h_k to hasz wielomianowy nad GF(p), p = 2⁶¹ − 1 (kawałki 60-bitowe plus długość wiadomości). Klucz k jest jeden na sesję, r to świeży klucz jednorazowy na tag. P-stwo fałszerstwa ≤ L/p, także po zobaczeniu prawdziwej pary (m, tag). Klucz pochodzi z puli współdzielonej przez Alice i Boba (`KeyPool`), której Ewa nie zna.
- **Tryb:** domyślnie tag na transkrypt rundy (po jednym na kierunek, przed użyciem klucza), bo tag na każdą wiadomość zużywa więcej klucza, niż QKD wytwarza.
- **MITM:** Ewa przecina światłowód i prowadzi dwa niezależne BB84. Bez uwierzytelniania oba się udają z QBER jak bez ataku, a Ewa zna oba klucze. Z uwierzytelnianiem jej podrobiony tag nie przechodzi weryfikacji i protokół się przerywa.
- **ML-DSA:** każda strona podpisuje swój transkrypt rundy (8 B długości + spakowane bity) ML-DSA z FIPS 204, z kontekstem `qkdsim/bb84-transcript`. Backend: `cryptography` (OpenSSL); zgodność z `dilithium-py` sprawdzona w obie strony. Podpis nie zużywa klucza, ale kosztuje 2420 / 3309 / 4627 B (poziom 44 / 65 / 87) na kierunek i rundę.
- **Model zaufania (Z5b):** klucze publiczne są rozprowadzone z góry, bez PKI. Odbiorca sprawdza podpis kluczem ze swojego katalogu dla deklarowanego nadawcy. Symulacja trzyma osobno prawdziwe klucze publiczne (ścieżka uczciwej strony) i katalog ofiary, więc zatruty katalog psuje tylko weryfikację wiadomości Ewy, a nie prawdziwej strony.
- **Scenariusze złamania:** „wyrocznia fałszerstw” to Ewa z kluczami prywatnymi ofiar w trakcie sesji (ML-DSA złamane na żywo). „Zatruty katalog” to klucz Ewy w katalogu ofiary (złamane Z5b). Złamanie po sesji modeluje wyciek kluczy prywatnych po wzmacnianiu prywatności: wiedzę Ewy liczy liniowa Ewa (rozdz. 7) z samego transkryptu.
- **Hybryda:** `HybridAuth` wymaga poprawności tagu Wegmana–Cartera i podpisu ML-DSA, więc Ewa musi pokonać oba. `hybrid_key` łączy klucz QKD i sekret ML-KEM przez HKDF-SHA256: wynik jest tajny, dopóki tajna jest choć jedna składowa (założenie: HKDF jest dobrym ekstraktorem).
- **Ograniczenia:** czas CPU Wegmana–Cartera pochodzi z pętli w Pythonie i nie jest reprezentatywny dla metody. Nie modelujemy kanałów bocznych implementacji ML-DSA ani ML-KEM; są tylko wierszem w tabeli modeli zagrożeń.

## 12. MDI-QKD

- **Przekaźnik:** dzielnik 50:50, PBS za każdym wyjściem i cztery detektory progowe (rys. 1 Lo, Curty, Qi 2012). Udany pomiar Bella to dokładnie dwa kliknięcia o ortogonalnych polaryzacjach: (1H, 2V) lub (1V, 2H) daje |ψ⁻⟩, (1H, 1V) lub (2H, 2V) daje |ψ⁺⟩. Bob odwraca bit poza bazą diagonalną z wynikiem |ψ⁺⟩ (Tabela 1).
- **Fizyka:** przy ustalonych fazach φ_A, φ_B każdy detektor dostaje stan koherentny, więc liczby fotonów w czterech trybach są niezależnymi zmiennymi Poissona. Monte Carlo losuje fazę względną, a model analityczny całkuje po niej. Parę pojedynczych fotonów (Y₁₁, e₁₁) liczymy z interferencji dwufotonowej, z przypadkami, gdy dociera jeden foton, i z ciemnymi zliczeniami. Sprawność detektorów jest wliczona w transmisję wejść (jednorodne straty komutują z optyką liniową).
- **Błąd optyki:** fizyczne obroty polaryzacji, w ramieniu Boba i w wyjściu 1 dzielnika, po połowie e_d (podpis rys. 2 artykułu). `qkd` (Ma i Razavi) traktuje e_d fenomenologicznie: E = e_d(1 − E₀) + (1 − e_d)E₀. Bez błędu optyki oba modele się pokrywają; porównujemy je w tests/crosscheck/test_mdi_qkd.py.
- **Stopa:** wzór (1) artykułu z nieskończenie wieloma stanami decoy (Y₁₁ i e₁₁ z modelu), μ_A = μ_B optymalizowane, przekaźnik pośrodku. Tło 6,02·10⁻⁶ liczymy na parę detektorów (3,01·10⁻⁶ na detektor, jak w dokumentacji `qkd`; artykuł tego nie precyzuje, a tło na detektor dałoby 177 km).
- **Ewa jako przekaźnik:** `MeasureAndAnnounce` mierzy oba impulsy w bazie Z i ogłasza |ψ⁻⟩, gdy wyniki się różnią. Ataki na detektory (2, 4) tracą sens, bo detektory i tak należą do Ewy; jedyne, co może zrobić, to mierzyć stany Alice i Boba, a to widać w bazie X.
- **Ograniczenia:** brak skończonego klucza dla MDI (Curty et al. 2014; `qkd` go ma) i estymacji decoy z danych; jeden przebieg Monte Carlo to jedna intensywność na stronę.

## 13. Splątane pary: BBM92, CHSH i device-independent

- **Źródło SPDC:** dwie niezależne pary trybów, (A_H, B_V) i (A_V, B_H), każda o termicznej liczbie par z średnią λ. Wykładnik stanu jest singletem SU(2), więc w każdej wspólnej bazie Alice ma n₁ fotonów w jednym trybie i n₂ w drugim, a Bob odwrotnie. Każdy foton dociera z p-stwem η swojego ramienia, u Boba trafia do złego detektora z p-stwem e_d; detektory progowe z ciemnymi zliczeniami, podwójne kliknięcie daje losowy bit. `analysis/entangled.pair_statistics` sumuje to dokładnie po (n₁, n₂), `entangled.simulate_pairs` losuje.
- **Zgodność z Ma, Fungiem i Lo 2007:** zysk (ich wzór (9)) zgadza się do 10⁻¹⁰. Ich QBER (wzór (10)) zakłada, że koincydencje bez wspólnej pary mylą się z p-stwem ½; różnica względem modelu dokładnego jest poniżej 0,1%.
- **CHSH:** urządzenia jak czarne skrzynki, korelacja V cos 2(a − b), sprawność η, brak kliknięcia czytany jako +1. S = 2√2Vη² + 2(1 − η)², QBER ustawień klucza η²(1 − V)/2 + η(1 − η), stopa device-independent przeciw atakom kolektywnym z granicy Acína i in. 2007. Próg S > 2 to 2/(1 + √2); próg dodatniej stopy wychodzi 92,31% wobec 92,4% podanych przez Pironio i in. 2009 (za dokumentacją `qkd`).
- **Fałszywe naruszenie:** `DetectorControl` (Gerhardt i in. 2011) to lokalny model, w którym Ewa wybiera ustawienia i wyniki, a detektory klikają tylko przy jej ustawieniu. Na samych koincydencjach daje S = 4p (2√2 przy p = 1/√2), na wszystkich rundach S = p + ½ < 2.
- **Ograniczenia:** test CHSH zakłada idealne źródło par (bez wielu par i ciemnych zliczeń) z białym szumem; brak skończonego klucza dla DI i ataków ogólnych.

## 14. Pamięć detektora: czas martwy i afterpulsing

- **Model:** `DetectorPair(dead_time_gates=τ, afterpulse=p)`. Po kliknięciu detektor jest martwy przez τ kolejnych bramek (model nieparaliżowalny: kliknięcia w martwych bramkach przepadają i nie przedłużają martwoty). Z p-stwem p klika sam w pierwszej żywej bramce; taki afterpuls może wywołać następny. Każdy detektor ma własną pamięć; stan nie przechodzi między paczkami 2²⁰ impulsów. Przy τ = 0 i p = 0 wyniki są identyczne co do bitu jak bez pamięci.
- **Wzór odnowienia:** przy niezależnych świeżych kliknięciach z p-stwem f na bramkę stacjonarne p-stwo kliknięcia to c = 1/(τ + 1 + (1 − p)(1/f − 1)); dla p = 0 daje to f/(1 + τf), dla τ = 0 f/(1 − p(1 − f)). e18 sprawdza go Monte Carlo dla f od 10⁻⁴ do 0,3.
- **Tor BB84 z afterpulsami (τ = 0):** afterpuls w bramce t + 1 zależy tylko od kliknięcia w bramce t, a nie od fotonów w t + 1. Działa więc jak niezależne ciemne zliczenie z p-stwem p·c, a dokładny model z rozdz. 4 wystarczy uzupełnić o to zliczenie (`analysis/wcp.afterpulse_dark_count`). Monte Carlo zgadza się z nim w granicach 2σ przy 4·10⁷ impulsach.
- **Wzory Lima et al. 2014:** R = D(1 + p_ap), e = p_dc + e_mis(1 − e^(−ημ)) + p_ap D/2. To pierwszy rząd modelu dokładnego, gdy p_ap = p/(1 − p) (łączna liczba afterpulsów na kliknięcie, z kaskadami). Przy D ≈ 5% zawyżają QBER o 0,04 pp, bo pomijają afterpulsy pochłonięte przez świeże kliknięcie i podwójne kliknięcia. Afterpulsy 4% podnoszą QBER o ok. 2 pp, cztery razy więcej niż cały błąd optyki 0,5%.
- **Ograniczenia:** p-stwo afterpulsu nie zależy od odstępu od lawiny (realne rozkłady są wykładnicze w czasie); czas martwy w bramkach, a nie w nanosekundach; brak zależności sprawności od czasu po martwocie (podstawa ataków typu „dead-time”, Weier i in. 2011).

## 15. Kwantowy generator losowy (przedłużenie ataku 8)

- **Model:** `qrng.BeamSplitterQrng`, dzielnik wiązki z dwoma detektorami (Jennewein i in. 2000; przegląd Herrero-Collantes, Garcia-Escartin 2017). W takcie stan koherentny o średniej μ, transmisja t, sprawności (η₀, η₁), ciemne zliczenia, czas martwy i afterpulsy z rozdz. 14. Bit 0, gdy klika tylko D0, bit 1, gdy tylko D1; reszta taktów odpada.
- **Obciążenie dokładnie:** stan koherentny dzieli się na dwa niezależne stany koherentne, a pamięć każdego detektora zależy tylko od jego kliknięć, więc procesy kliknięć D0 i D1 są niezależne. Stąd P(1) = c₁(1 − c₀)/(c₀(1 − c₁) + c₁(1 − c₀)) ze stacjonarnymi p-stwami kliknięcia c_i z wzoru odnowienia, także z pamięcią. Korelacji kolejnych bitów nie liczymy analitycznie: czas martwy odwraca bit (P(równy poprzedniemu) ≈ 0,34 przy τ = 10), afterpuls go powtarza.
- **Min-entropia:** metoda najczęstszej wartości z NIST SP 800-90B (6.3.1, górna granica ufności 99%), warunkowo na k poprzednich bitach, w najgorszym kontekście. Przy k = 0 korelacje są niewidoczne (H ≈ 0,95), przy k = 4 H ≈ 0,49.
- **Ekstrakcja:** Toeplitz z `postprocessing/hashing.py`, m = ⌊nH − 2 log₂(1/ε)⌋ z bloku n = 65 536 (lemat o resztkowym haszowaniu), jedno jawne ziarno dla wszystkich bloków (silny ekstraktor). `qrng.QrngSource` to RandomSource Alice z wyjścia, po ekstrakcji albo surowy.
- **Atak 8 z QRNG (e19):** surowe bity niedopasowanego generatora (η₁/η₀ = 0,7, P(X) = 0,41) jako bazy Alice dają Ewie trafienie bazy z p-stwem 0,59; po ekstrakcji wraca ½. Model z e13 (`intercept_resend_model` z `right_basis`).
- **Obrona: testy zdrowia** (`defenses/health_tests.py`, NIST SP 800-90B 4.4): RCT z C = 1 + ⌈20/H⌉ i APT z C = 1 + CRITBINOM(1024, 2^(−H), 1 − 2⁻²⁰); progi zgadzają się z Tabelą 2 normy. Oślepiony detektor: alarm RCT po C bitach. Dryf sprawności: p-stwo alarmu APT na okno zgodne ze wzorem dwumianowym.
- **Ograniczenia:** brak korelacji między bitami a stanem lasera (szum fazy, fluktuacje mocy), H z kalibracji zamiast pełnego zestawu estymatorów 90B, ekstrakcja z jednym ziarnem bez osobnej analizy jego jakości; ataki na QRNG z zewnątrz (oślepianie) tylko jako awaria detektora.

## 16. Uzgadnianie LDPC

- **Protokół:** `postprocessing/ldpc.reconcile`, wybierany przez `PostprocessingConfig(reconciliation="ldpc")`. Alice wysyła syndrom Hx każdej ramki; Bob dekoduje algorytmem BP (sum-product w LLR, kanał BSC(Q̂)) do x̂ z Hx̂ = s. Jedna wiadomość na ramkę w jedną stronę, plus ziarno macierzy raz na sesję. Klucz dzielimy na równe ramki do 16 384 bitów; ostatnią skracamy zerami znanymi obu stronom. Ramka niezdekodowana odpada (`KeyBalance.discarded`), a jej syndrom i tak liczymy do wycieku.
- **Kod:** stały stopień zmiennych d_v = 3, stopnie kontroli ⌊E/m⌋ i ⌈E/m⌉ (Gallager 1962), m = ⌈f·n·h(Q̂)⌉ z zakładaną sprawnością f = 1,6. Krawędzie z losowej permutacji bez krawędzi podwójnych; cykli długości 4 nie usuwamy.
- **Ewolucja gęstości:** gęstości LLR na siatce co 0,05 do ±25 (Richardson, Urbanke 2001), węzeł kontroli z tablicą par, węzeł zmiennej przez splot, normalizacja masy co iterację (bez niej błędy zaokrągleń rosną ×11 na iterację). Próg BP kodu (3,6) wychodzi 0,0841 wobec 0,084 z literatury, niezależnie od kroku siatki 0,05 albo 0,025.
- **Wyniki (e20):** przy Q = 1–5% kody regularne potrzebują f ≈ 1,30–1,68 dla FER = 10% przy ramce 16 384 bitów, a granica z ewolucji gęstości to 1,26–1,52. Cascade daje 1,12–1,16, ale potrzebuje 170–850 wiadomości na 16 384 bity. Na torze GYS 25 km klucz końcowy z LDPC jest o ok. 18% krótszy niż z Cascade, a uzgadnianie zajmuje 5 wiadomości zamiast 1916.
- **Liniowa Ewa:** wiersze syndromu idą do transkryptu przy `record_rows=True`, więc `leakage.linear_eve_knowledge` działa jak dla Cascade (wymaga zdekodowania wszystkich ramek).
- **Ograniczenia:** kody regularne zamiast nieregularnych optymalizowanych ewolucją gęstości, brak adaptacji sprawności przez przebijanie i skracanie (Elkouss i in. 2009 osiągają f ≈ 1,1), brak usuwania krótkich cykli i dekodera z wczesnym zatrzymaniem po stagnacji.

## 17. Obrony z progiem i krzywe ROC (e17)

- **Test progu QBER (ataki 1 i 6, `defenses/qber_test.py`):** Alice i Bob ujawniają k bitów i przerywają, gdy błędów jest więcej niż t, gdzie t ustalone z góry tak, by uczciwy przebieg przerywał się z p-stwem ≤ α. Liczba błędów to Bin(k, Q), więc ROC jest dokładna: FPR = P(Bin(k, Q₀) > t), TPR = P(Bin(k, Q₁) > t). Przy k = 1000 i tle 2%: przechwycenie 4% impulsów wykryte w 23,4% sesji (jak Monte Carlo w e07), atak Fuchsa z D = 5% w 99,8%. Tę samą funkcję progu używa e07.
- **Monitor czasu przyjścia (atak 4, `defenses/timing_monitor.py`):** Bob znakuje czas kliknięcia z rozmyciem Gaussa σ i liczy kliknięcia poza oknem ±w; bez ataku to 2Φ(−w/σ), a Ewa przesuwa impulsy o −355 i 395 ps, więc większość jej kliknięć wypada z okna. Przy 30 kliknięciach na test wykrycie 100% przy rozmyciu 50 i 150 ps; Monte Carlo na przesunięciach z prawdziwego przebiegu ataku zgadza się z modelem dwumianowym.
- **Monitor światła wchodzącego do Alice (atak 5, `defenses/input_monitor.py`):** ta sama fotodioda co w e05 (odczep 1%, szum odczytu), średnia z 1000 odczytów przy 2 MHz. Alarm przy średniej powyżej z_{1−α}σ/√W; największa sonda, którą monitor przeoczy z p-stwem ≥ 1%, ma 5,7·10⁵ fotonów na impuls. Wtedy do wycieku ≤ 10⁻³ bitu na impuls przy odbiciu −20 dB wystarcza 69,2 dB izolacji: monitor ogranicza μ_in, izolacja μ_out.
- **Ograniczenia:** monitor czasu zakłada gaussowskie rozmycie i znany nominalny czas przyjścia (bez dryfu); monitor wejściowy nie widzi sond poza pasmem fotodiody ani impulsów krótszych niż czas całkowania.
