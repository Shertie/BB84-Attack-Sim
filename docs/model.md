# Model fizyczny, konwencje i założenia

Dokument referencyjny dla kodu w `src/qkdsim/`. Zmiana modelu = zmiana tego pliku w tym samym commicie.

## 1. Zakres

- Protokół: BB84 z kodowaniem polaryzacyjnym, bazy Z (H/V) i X (D/A).
- Kanał: światłowód 1550 nm, straty wykładnicze w długości, stały błąd optyki.
- Źródło: słabe impulsy koherentne (WCP), opcjonalnie z kilkoma intensywnościami (decoy).
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
| Losowość | wybory Alice i Boba przez wstrzykiwany `RandomSource`; przyroda (liczba fotonów, wynik pomiaru) i Ewa mają osobne generatory. Wszystkie cztery strumienie pochodzą z jednego ziarna (`RandomStreams.from_seed`, ADR 0003); żadnych globalnych `np.random.*` |
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

Wartości kontrolne wynikające z tych parametrów: ROADMAP.md, rozdz. 3.2.

## 4. Model kroków toru

1. **Źródło:** liczba fotonów n ~ Poisson(μ_i) dla intensywności wybranej w danym impulsie. n jest zapisywane, bo potrzebuje go atak PNS, ale Alice i Bob go nie znają (ADR 0001).
2. **Kanał:** liczba ocalałych fotonów ~ Binomial(n, 10^(−αL/10)). Błąd optyki to obrót polaryzacji o kąt θ_e z sin²θ_e = e_d.
3. **Bob:** wybór bazy, PBS (φ = 0 dla Z, π/4 dla X), rozkład fotonów na detektory ~ Binomial(n, cos²(θ − φ)). Prawdopodobieństwo liczy się jako (1 + cos 2(θ − φ))/2, co daje dokładnie 0 i 1 dla stanów z bazy analizatora; wprost liczone cos² zostawiłoby resztę rzędu 10⁻³³. Dalej: każdy foton wykryty z prawdopodobieństwem η, a do tego niezależne ciemne zliczenia p_d na detektor i bramkę.
4. **Klasyfikacja zdarzeń:** brak kliknięcia, pojedyncze, podwójne. Podwójne kliknięcie dostaje losowy bit i nie jest odrzucane (squashing, Beaudry, Moroder, Lütkenhaus 2008).
5. **Później (F3, F6):** tryb liniowy detektora (oślepianie), krzywe sprawności w czasie η_i(t) (time-shift), czas martwy i afterpulsing (F7).

Własność do testów: dla stanu koherentnego liczby fotonów w dwóch wyjściach PBS są niezależnymi rozkładami Poissona o średnich μηcos² i μηsin². Próbkowanie przez n i rozkład dwumianowy musi dawać ten sam rozkład (F1.13).

### Implementacja i model analityczny (F1.5–F1.9)

- Detektor z k fotonami po PBS klika z p-stwem 1 − (1 − p_d)(1 − η)^k: jedno losowanie na detektor i impuls, dokładnie równoważne dwumianowi i ciemnemu zliczeniu.
- Ciemne zliczenia na detektor: p_d = 1 − √(1 − Y₀), żeby łączne tło obu detektorów wynosiło Y₀.
- `analysis/wcp.py` ma dwa modele:
  - wzory Ma et al. (sumy (7) i (8)), na których stoją stopy klucza;
  - dokładną wartość oczekiwaną dla detektorów symulatora (niezależne rozkłady Poissona w obu detektorach, podwójne kliknięcie z losowym bitem).

  Zysk jest w obu modelach identyczny. QBER różni się o mniej niż 3·10⁻⁵ względnie dla GYS na 0–200 km. MC jest walidowane względem modelu dokładnego, ale przechodzi też test względem wzoru z artykułu.
- Wzór Ma et al. na E_μ może przekroczyć ½ dla e_d > (1 − Y₀)/2 (ROADMAP, K16). Model dokładny nigdy.

## 5. Założenia dowodu bezpieczeństwa

Lista, na której w F5.1 powstanie mapa „atak → złamane założenie → obrona → metryka”. Kolumna „Atak” to przewidywanie z planu, do potwierdzenia wynikami.

| # | Założenie | Atak, który je łamie | Obrona |
| --- | --- | --- | --- |
| Z1 | Rozkład liczby fotonów źródła jest znany i uwzględniony w długości klucza (GLLP / decoy) | 3, PNS (gdy liczymy klucz tak, jakby impulsy były jednofotonowe) | stany decoy |
| Z2 | Moduł Alice nie ujawnia ustawień poza wysłanym stanem (brak kanałów bocznych) | 5, Trojan horse | izolatory, monitor mocy, filtr |
| Z3 | Detektory Boba działają w trybie jednofotonowym, a ich sprawność nie zależy od bitu, bazy ani czasu przyjścia | 2, oślepianie; 4, time-shift | monitor mocy/fotoprądu, losowa sprawność, losowe przypisanie bit ↔ detektor, MDI |
| Z4 | Pomiar Boba jest opisany modelem squashing (podwójne kliknięcia dostają losowy bit) | odrzucanie podwójnych kliknięć jako „optymalizacja” implementacji | losowy bit zamiast odrzucenia |
| Z5a | Kanał klasyczny jest uwierzytelniony, a uwierzytelnienie nie zostaje złamane **w trakcie** sesji (później już nie musi wytrzymać) | 7, MITM; MITM z fałszerstwem podpisu w czasie rzeczywistym | Wegman–Carter z kluczem współdzielonym, ML-DSA lub hybryda (wariant E) |
| Z5b | Klucze publiczne stron są autentyczne (tylko przy uwierzytelnianiu podpisem) | 7, MITM z podmianą klucza publicznego w katalogu | zaufany katalog / PKI, weryfikacja odcisku klucza |
| Z6 | Wybory bitów, baz i intensywności są prawdziwie losowe i prywatne | 8, słaby generator | QRNG; testy losowości wykrywają obciążenie, ale nie przewidywalność |
| Z7 | Ewa jest ograniczona tylko prawami fizyki; wszystkie błędy przypisujemy jej | 6, granica dla ataków indywidualnych (Fuchs) vs ogólnych (Shor–Preskill) | odpowiednio konserwatywny próg QBER |
| Z8 | Parametry są estymowane ze skończonej próbki z jawnym ε | „Ewa w szumie”: mały ułamek przechwytu ginie w fluktuacjach | rachunek skończonego klucza (Tomamichel et al. 2012) |

Wspólny wniosek z planu: atak, który nie podnosi QBER (2, 4, 5, 7, 8), łamie jedno z założeń Z2–Z6, a nie wzór na długość klucza.

Przy Z5a widać różnicę między QKD a ML-KEM (ROADMAP, rozdz. 2.2a). QKD potrzebuje uwierzytelnienia, które jest bezpieczne tylko w chwili sesji. Złamanie ML-DSA po fakcie nie odsłania wcześniejszych kluczy. Klucz z ML-KEM traci bezpieczeństwo, gdy w dowolnym momencie w przyszłości padną problemy kratowe.

## 6. Znane uproszczenia (na starcie)

- e_d i Y₀ nie zależą od L ani od czasu.
- Brak dyspersji i dryfu polaryzacji w kanale.
- Brak czasu martwego i afterpulsingu (do F3/F7).
- Skończony klucz tylko dla modelu jednofotonowego. WCP + decoy (Lim et al. 2014) dopiero w F7.

## 7. Postprocessing (F2)

- **Losowość publiczna:** próbka do estymacji, permutacje Cascade i ziarna haszy pochodzą ze strumienia `RandomStreams.public`. Jest jawny, bo i tak jest ogłaszany przez kanał klasyczny, a osobny strumień nie przesuwa losowań stron.
- **Wyciek:** liczy się jeden bit na każdą parzystość ogłoszoną przez Alice w Cascade (bloki i połówki w BINARY) oraz bity hasza weryfikacyjnego. Parzystości Boba mówią tylko, gdzie są błędy, więc nie są liczone osobno. Bazy i ujawniona próbka nie ujawniają nic o reszcie klucza.
- **Hasz:** macierz Toeplitza T[i, j] = s[i − j + n − 1]; ta sama rodzina 2-uniwersalna służy do weryfikacji (64 bity, ε_cor = 2⁻⁶⁴) i do wzmacniania prywatności.
- **Długość klucza (asymptotycznie):** ℓ = ⌊n·s·[1 − h(e_ph)] − leak_EC − leak_ver⌋. Dla źródła jednofotonowego s = 1 i e_ph to błąd fazowy. Dla WCP s = Q₁/Q_μ i e_ph = e₁ (GLLP). Poprawki skończonego klucza dochodzą w F4.
- **Wiedza pasywnej Ewy:** dokładnie dim(span P ∩ span T) bitów klucza końcowego, gdzie P to ujawnione funkcje liniowe, a T macierz wzmacniania prywatności. Dlatego zapomniany wyciek daje Ewie co najmniej ℓ − (n − rank P) bitów bez żadnego ataku na kanał kwantowy.

## 8. Ataki w światłowodzie (F3)

- **Położenie Ewy:** parametr `position` ∈ [0, 1] dzieli światłowód na odcinek do Ewy (bez błędu optyki) i od Ewy do Boba (z błędem optyki). Złożenie odcinków ma tę samą transmisję co cały kanał. Błąd optyki jest po stronie Boba, więc Ewa mierzy stan Alice bez obrotu.
- **Wiedza Ewy:** `EveLog` (impulsy zmierzone, baza, wynik) jest osobny od rekordów Alice i Boba. `Counts` liczy wśród wykrytych impulsów te zmierzone przez Ewę, te z trafnym wynikiem i te, które zna na pewno (mierzyła w bazie Alice).
- **Intercept-resend (atak 1):** Ewa mierzy w losowej bazie idealnym detektorem i wysyła 1 foton; impuls bez fotonów w jej punkcie zostaje próżnią. Model analityczny w `analysis/intercept_resend.py` jest dokładną wartością oczekiwaną dla symulatora (mieszanina: impuls nieprzechwycony, próżnia u Ewy, foton w bazie Alice, foton w bazie sprzężonej).
- **QBER(p) dla WCP nie jest liniowe** (ROADMAP, K14): prosta e_d + p(¼ − e_d/2) jest dokładna dla p ∈ {0, 1}, a pomiędzy obowiązuje tylko przy μT₁ ≪ 1.
- **Oślepianie (atak 2):** detektor ma tryb liniowy (`LinearMode`, progi Clavis2 z Lydersena et al. 2010): ciągłe światło ≥ `blind_power_w` wyłącza tryb Geigera i ciemne zliczenia. Od tej chwili klika tylko impuls klasyczny, z p-stwem rosnącym liniowo między P_never a P_always (założenie modelu). Nieoślepiony detektor traktuje jasny impuls jak P·τ/hν fotonów, więc po prostu klika. Ewa mierzy tuż za Alice, wysyła impulsy wyzwalające bezstratną linią i dobiera ich częstość tak, by zysk był jak bez ataku. Odtwarza też spodziewany QBER, odwracając część bitów.
- **Obrót a okno ataku:** stały obrót polaryzacji (błąd optyki) dzieli jasny impuls w bazie sprzężonej w proporcji ½ ± √(e_d(1 − e_d)). Bez kompensacji okno doskonałego ataku znika (ROADMAP, K5).
- **Obrony:** monitor mocy (odczep + szum, próg na odczycie) i losowe tłumienie przed detektorami (test z: odsetek kliknięć w bramkach tłumionych vs oczekiwany czynnik). Tłumik nie zmienia stanu oślepienia, bo ten zależy od średniego fotoprądu.
- **PNS (atak 3):** Ewa tuż za Alice mierzy liczbę fotonów (QND), blokuje impulsy jednofotonowe, z wielofotonowych zabiera 1 foton (zna bit po ujawnieniu baz), a resztę wysyła bezstratnie. Strategia (`pns_strategy`) dobiera blokowanie i przepuszczanie tak, by zysk sygnału był jak bez ataku. Od L_kryt = 32,8 km (GYS, μ = 0,48) wystarcza do tego samo przepuszczanie impulsów wielofotonowych. Ewa nie widzi intensywności, więc decoy dostaje tę samą strategię i jego zysk wychodzi niespójny.
- **Decoy z danych:** `postprocessing/decoy.py` liczy Y₁^L, e₁^U i R z liczników MC (vacuum+weak, asymptotycznie). `defenses/decoy_check.py` testuje spójność: z Q_μ i Y₀ wyznacza η̂ i przewiduje Q_ν. Szum estymaty decoy jest duży (ROADMAP, K17).

## 9. Skończony klucz (F4)

- **Rachunek:** `postprocessing/finite_key.py` implementuje twierdzenie Tomamichela et al. 2012 w dwóch wydrukowanych wersjach (tekst główny i Methods) oraz uproszczenie z planu (Hoeffding na samej próbce) do porównania. Protokół asymetryczny: n bitów klucza, k bitów testu, z góry ustalony próg Q_tol, przerwanie, gdy QBER z próbki go przekroczy. M(n, k) = (√n + √k)².
- **Robustness:** p-stwo przerwania uczciwego przebiegu liczone jest dokładnie z ogona rozkładu dwumianowego.
- **Zakres:** tylko źródło jednofotonowe. Skończony klucz dla WCP z decoy (Lim et al. 2014) to F7 (ROADMAP, K3 i K17).

## 10. Atak indywidualny (F4)

- **Konstrukcja:** optymalny atak indywidualny to fazowo-kowariantny kloner (częściowa zamiana o kąt η w podprzestrzeni {|01⟩, |10⟩}) w układzie obróconym o 120° wokół (1, 1, 1), tak by bazy Z i X leżały na równiku. Daje on D = sin²(η/2) u Boba i błąd Ewy ½ − √(D(1 − D)), czyli dokładnie granicę Fuchsa, dla wszystkich czterech stanów BB84.
- **Weryfikacja:** QuTiP buduje tę samą operację niezależnie, z generatora i operatorów Pauliego. Wierności Boba i Ewy wynikają ze śladów częściowych, a prawdopodobieństwa wyników z projektorów.
- **Zakres:** atak działa na pojedynczych fotonach i idealnych detektorach (granica teoretyczna), więc ma własny sampler zamiast toru impulsów.

## 11. Uwierzytelnianie i MITM (F4)

- **Wegman–Carter:** tag = (h_k(m) + r) mod p, gdzie h_k to hasz wielomianowy nad GF(p), p = 2⁶¹ − 1 (kawałki 60-bitowe plus długość wiadomości). Klucz k jest jeden na sesję, r to świeży klucz jednorazowy na tag. P-stwo fałszerstwa ≤ L/p, także po zobaczeniu prawdziwej pary (m, tag). Klucz pochodzi z puli współdzielonej przez Alice i Boba (`KeyPool`), której Ewa nie zna.
- **Tryb:** domyślnie tag na transkrypt rundy (po jednym na kierunek, przed użyciem klucza), bo tag na każdą wiadomość zużywa więcej klucza, niż QKD wytwarza (ROADMAP, K19).
- **MITM:** Ewa przecina światłowód i prowadzi dwa niezależne BB84. Bez uwierzytelniania oba się udają z QBER jak bez ataku, a Ewa zna oba klucze. Z uwierzytelnianiem jej podrobiony tag nie przechodzi weryfikacji i protokół się przerywa.
- **ML-DSA (wariant E):** każda strona podpisuje swój transkrypt rundy (8 B długości + spakowane bity) ML-DSA z FIPS 204, z kontekstem `qkdsim/bb84-transcript`. Backend: `cryptography` (OpenSSL); zgodność z `dilithium-py` sprawdzona w obie strony. Podpis nie zużywa klucza, ale kosztuje 2420 / 3309 / 4627 B (poziom 44 / 65 / 87) na kierunek i rundę.
- **Model zaufania (Z5b):** klucze publiczne są rozprowadzone z góry, bez PKI. Odbiorca sprawdza podpis kluczem ze swojego katalogu dla deklarowanego nadawcy. Symulacja trzyma osobno prawdziwe klucze publiczne (ścieżka uczciwej strony) i katalog ofiary, więc zatruty katalog psuje tylko weryfikację wiadomości Ewy, a nie prawdziwej strony.
- **Scenariusze złamania:** „wyrocznia fałszerstw” to Ewa z kluczami prywatnymi ofiar w trakcie sesji (ML-DSA złamane na żywo). „Zatruty katalog” to klucz Ewy w katalogu ofiary (złamane Z5b). Złamanie po sesji modeluje wyciek kluczy prywatnych po wzmacnianiu prywatności: wiedzę Ewy liczy liniowa Ewa z F2.8 z samego transkryptu.
- **Hybryda:** `HybridAuth` wymaga poprawności tagu Wegmana–Cartera i podpisu ML-DSA, więc Ewa musi pokonać oba. `hybrid_key` łączy klucz QKD i sekret ML-KEM przez HKDF-SHA256: wynik jest tajny, dopóki tajna jest choć jedna składowa (założenie: HKDF jest dobrym ekstraktorem).
- **Ograniczenia:** czas CPU Wegmana–Cartera pochodzi z pętli w Pythonie i nie jest reprezentatywny dla metody. Nie modelujemy kanałów bocznych implementacji ML-DSA ani ML-KEM; są tylko wierszem w tabeli modeli zagrożeń.
