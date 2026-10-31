## Ewolucja gęstości: progi BP kodów o d_v = 3

| R | Stopnie kontroli (ułamki krawędzi) | Próg BP Q* | f = (1 − R)/h(Q*) |
| --- | --- | --- | --- |
| 0,50 | 6: 1,00 | 0,0841 | 1,201 |
| 0,60 | 7: 0,47, 8: 0,53 | 0,0586 | 1,243 |
| 0,70 | 10: 1,00 | 0,0374 | 1,303 |
| 0,80 | 15: 1,00 | 0,0203 | 1,399 |
| 0,85 | 20: 1,00 | 0,0132 | 1,479 |
| 0,90 | 30: 1,00 | 0,0074 | 1,590 |

Kod (3,6): próg 0,0841 wobec 0,084 u Richardsona i Urbankego (2001); siatka LLR co 0,05 do ±25, bisekcja do 2·10⁻⁴. Kody regularne tracą do pojemności tym więcej, im wyższa sprawność: przy R = 0,9 f ≈ 1,59 nawet dla nieskończonej ramki.

## Wodospad kodu (3,6): p-stwo błędu ramki

| Q | n = 1024 | n = 8192 | n = 65536 |
| --- | --- | --- | --- |
| 5,0% | 0,005 | 0,000 | 0,000 |
| 6,0% | 0,025 | 0,000 | 0,000 |
| 7,0% | 0,090 | 0,000 | 0,000 |
| 7,5% | 0,235 | 0,000 | 0,000 |
| 8,0% | 0,435 | 0,283 | 0,000 |
| 8,2% | 0,520 | 0,317 | 0,133 |

Liczba prób: 200 dla n = 1024, 60 dla n = 8192, 15 dla n = 65536; do 100 iteracji BP. Z długością ramki krzywa ostrzeje i zbliża się do progu z ewolucji gęstości.

## Reżim QKD: ramka 16384 bitów, 20 prób na punkt

| Q | FER przy f = 1,20 | FER przy f = 1,30 | FER przy f = 1,40 | FER przy f = 1,50 | FER przy f = 1,60 | FER przy f = 1,70 | FER przy f = 1,80 | FER przy f = 2,00 | LDPC: f przy FER = 10% | granica DE | Cascade: f | Cascade: wiadomości | LDPC: wiadomości na ramkę |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1% | 1,00 | 1,00 | 1,00 | 0,70 | 0,30 | 0,05 | 0,05 | 0,00 | 1,68 | 1,52 | 1,122 | 172 | 1 |
| 2% | 1,00 | 0,95 | 0,60 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 1,48 | 1,40 | 1,136 | 343 | 1 |
| 3% | 1,00 | 0,85 | 0,20 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 1,45 | 1,32 | 1,150 | 496 | 1 |
| 5% | 1,00 | 0,10 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 1,30 | 1,26 | 1,162 | 846 | 1 |

LDPC wysyła jeden syndrom na ramkę w jedną stronę (plus ziarno macierzy raz na sesję); Cascade prowadzi setki wymian parzystości. Za to kody regularne potrzebują większego f: ramka skończona dokłada swoje do granicy z ewolucji gęstości. Praktyka używa kodów nieregularnych i adaptacji sprawności (przebijanie, skracanie), co daje f ≈ 1,1 (Elkouss i in. 2009).

## Pełny postprocessing: GYS 25 km, μ = 0,48, 2·10^7 impulsów

| Uzgadnianie | Klucz po próbce | QBER próbki | Wyciek EC | f | Wiadomości EC | Odrzucone bity | Błędy resztkowe | Weryfikacja | Klucz końcowy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cascade | 57767 | 3,35% | 14045 | 1,161 | 1916 | 0 | 0 | tak | 31433 |
| ldpc | 57767 | 3,35% | 19564 | 1,617 | 5 | 0 | 0 | tak | 25914 |
