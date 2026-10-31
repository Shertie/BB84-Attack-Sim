## Odtworzenie Gisina et al. 2006 (rys. 7, μ_out = 0,1)

| Faza | Artykuł | Model |
| --- | --- | --- |
| stała (pomiar Helstroma) | 0,135 bitu | 0,1350 bitu |
| losowa (licznik fotonów) | 0,095 bitu | 0,0952 bitu |

## Monte Carlo w łączu GYS (25 km, μ = 0,48, 2,0·10^7 impulsów na punkt)

| μ_out | Faza | Informacja Ewy (MC) | Model | z QBER | z zysku |
| --- | --- | --- | --- | --- | --- |
| 0,001 | stała | 0,0016 ± 0,0003 | 0,0014 | -2,04 | +1,28 |
| 0,010 | stała | 0,0139 ± 0,0008 | 0,0143 | +0,77 | +0,58 |
| 0,100 | stała | 0,1321 ± 0,0023 | 0,1350 | +0,41 | -1,43 |
| 1,000 | stała | 0,7773 ± 0,0035 | 0,7808 | +2,09 | -1,72 |
| 0,001 | losowa | 0,0007 ± 0,0001 | 0,0010 | +1,61 | -0,49 |
| 0,010 | losowa | 0,0104 ± 0,0004 | 0,0100 | -0,27 | -0,21 |
| 0,100 | losowa | 0,0977 ± 0,0012 | 0,0952 | +0,87 | -1,14 |
| 1,000 | losowa | 0,6329 ± 0,0019 | 0,6321 | -0,94 | -0,90 |

Sygnału Ewa nie dotyka, więc QBER i zysk są takie jak bez ataku (kolumny z).

## Baza w czasie rzeczywistym

Intercept-resend z bazą odgadniętą z sondy mieści się pod progiem QBER 11%, gdy Ewa trafia bazę z p-stwem ≥ 0,835. Potrzebuje do tego μ_out ≥ 1,65 ze stałą fazą i ≥ 3,15 z losową. Przy małym μ_out przewaga w odgadywaniu bazy rośnie jak 0,27·√μ ze stałą fazą, ale tylko jak μ²/8 z losową: bez odniesienia fazowego pojedynczy foton nic nie mówi o bazie.

## Wymagana izolacja (μ_in fotonów sondy, tłumik Alice 0 dB)

| Odbicie (Gisin et al. 2006) | μ_in | Wyciek ≤ 1,0·10^-3 bitu | Wyciek ≤ 1,0·10^-6 bitu |
| --- | --- | --- | --- |
| łącznik ze skośnym szlifem (−40 dB) | 1,0·10^6 | 52 dB (50 dB z losową fazą) | 82 dB (80 dB z losową fazą) |
| łącznik ze skośnym szlifem (−40 dB) | 1,0·10^9 | 82 dB (80 dB z losową fazą) | 112 dB (110 dB z losową fazą) |
| łącznik ze skośnym szlifem (−40 dB) | 1,0·10^12 | 112 dB (110 dB z losową fazą) | 142 dB (140 dB z losową fazą) |
| modulator fazy (−20 dB) | 1,0·10^6 | 72 dB (70 dB z losową fazą) | 102 dB (100 dB z losową fazą) |
| modulator fazy (−20 dB) | 1,0·10^9 | 102 dB (100 dB z losową fazą) | 132 dB (130 dB z losową fazą) |
| modulator fazy (−20 dB) | 1,0·10^12 | 132 dB (130 dB z losową fazą) | 162 dB (160 dB z losową fazą) |
| lustro (−1 dB) | 1,0·10^6 | 91 dB (89 dB z losową fazą) | 121 dB (119 dB z losową fazą) |
| lustro (−1 dB) | 1,0·10^9 | 121 dB (119 dB z losową fazą) | 151 dB (149 dB z losową fazą) |
| lustro (−1 dB) | 1,0·10^12 | 151 dB (149 dB z losową fazą) | 181 dB (179 dB z losową fazą) |

Każdy rząd wielkości μ_in to 10 dB izolacji więcej, a losowa faza oszczędza 1,6 dB przy małym wycieku (czynnik 1/ln 2).

Koszt w kluczu: przy 25 km decoy vacuum+weak daje 0,221 bitu klucza na bit po siftingu. Wyciek I na impuls trzeba usunąć dodatkowym wzmacnianiem prywatności: I = 10⁻³ zabiera 0,45% klucza, I = 0,135 (przykład Gisina) 61%.


## Budżet izolacji według Lucamariniego et al. 2015 (μ_out = 1,0·10^-6, N = 10²⁰ fotonów/s)

| f_A | μ_in = N/f_A | γ z wzoru (nasz rachunek) | γ w Tabeli I | |Ṙ| + 2|Ȧ| + n|İ| | Informacja Ewy przy μ_out (stała / losowa faza) |
| --- | --- | --- | --- | --- | --- |
| 1,0·10^9 Hz | 1,0·10^11 | 170,0 dB | 170 dB | 40 + 2·35 + 1·60 = 170 dB | 1,4·10^-6 / 1,0·10^-6 bitu |
| 1,0·10^9 Hz | 1,0·10^11 | 170,0 dB | 170 dB | 50 + 2·0 + 2·60 = 170 dB | 1,4·10^-6 / 1,0·10^-6 bitu |
| 1,0·10^6 Hz | 1,0·10^14 | 200,0 dB | 200 dB | 40 + 2·30 + 2·50 = 200 dB | 1,4·10^-6 / 1,0·10^-6 bitu |
| 1,0·10^6 Hz | 1,0·10^14 | 200,0 dB | 200 dB | 50 + 2·0 + 3·50 = 200 dB | 1,4·10^-6 / 1,0·10^-6 bitu |
| 1,0·10^3 Hz | 1,0·10^17 | 230,0 dB | 230 dB | 40 + 2·35 + 2·60 = 230 dB | 1,4·10^-6 / 1,0·10^-6 bitu |
| 1,0·10^3 Hz | 1,0·10^17 | 230,0 dB | 230 dB | 50 + 2·0 + 3·60 = 230 dB | 1,4·10^-6 / 1,0·10^-6 bitu |

Wzór (4) artykułu, μ_out = Nγ/f_A, to nasz μ_out = μ_in·10^(−A/10) z μ_in ograniczonym progiem uszkodzenia światłowodu; składowe izolacji sumują się jak w naszym `round_trip_attenuation_db` (odbicie raz, tłumik dwa razy). Kryterium artykułu μ_out = 10⁻⁶ daje Ewie ok. 10⁻⁶ bitu na impuls; przy naszym progu wycieku 1,0·10^-3 bitu wystarcza μ_out ≤ 6,9·10^-4, czyli przy 1 GHz 142 dB zamiast 170 dB. Stopy klucza pod atakiem (wzór (6) artykułu) nie liczymy.
