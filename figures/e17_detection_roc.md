## Test progu QBER (próbka 1000 bitów, tło 2,00%)

| Atak | QBER pod atakiem | Wykrycie przy 1% fałszywych alarmów | AUC |
| --- | --- | --- | --- |
| atak 1, p = 4% | 2,78% | 23,4% | 0,872 |
| atak 1, p = 8% | 3,57% | 75,9% | 0,984 |
| atak 6, D = 5% | 5,00% | 99,8% | 1,000 |
| atak 6, D = 8% | 8,00% | 100,0% | 1,000 |

Liczba błędów w próbce to Bin(k, Q), więc krzywe są dokładne. Atak 1 na 4% impulsów wykrywa się w 23% sesji, jak w Monte Carlo e07. Atak 6 daje QBER równy D, więc przy tle 2% test wykrywa go już przy D = 5%; poniżej tła nie odróżni go od szumu, ale wtedy rachunek klucza i tak przypisuje cały QBER Ewie (założenie Z7).

## Monitor czasu przyjścia (atak 4, 30 kliknięć na test, okno ±200 ps)

| Rozmycie | Poza oknem: uczciwie / atak | Wykrycie (model / MC) | Fałszywe alarmy (MC) | AUC |
| --- | --- | --- | --- | --- |
| 50 ps | 0,01% / 100,0% | 100,0% / 100,0% | 0,1% | 1,000 |
| 150 ps | 18,24% / 89,1% | 100,0% / 100,0% | 0,4% | 1,000 |

Ewa przesuwa impulsy o -355 i 395 ps (p_A = 23,0%), więc nawet przy rozmyciu 150 ps większość kliknięć wypada poza okno. Losowe przypisanie bit ↔ detektor (e11) usuwa jej wiedzę, a monitor czasu wykrywa sam atak.

## Monitor światła wchodzącego do Alice (atak 5)

Fotodioda jak w e05: odczep 1%, szum 1,0·10^-8 W na odczyt, średnia z 1000 odczytów, 2 MHz, 1% fałszywych alarmów. Sondę przeoczy z p-stwem ≥ 1% tylko do 5,7·10^5 fotonów na impuls. Żeby wyciek nie przekroczył 1·10^-3 bitu na impuls przy odbiciu od modulatora (−20 dB), wystarczy wtedy 69,2 dB izolacji: monitor ogranicza μ_in, izolacja μ_out (e12).
