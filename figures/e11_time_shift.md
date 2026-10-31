## Odtworzenie Tabeli I(c) Zhao et al. 2008 z danych artykułu

| Wielkość | Artykuł | Z Tabeli I(a)–(b) |
| --- | --- | --- |
| p_A (zera i jedynki Boba w równowadze) | 23,0% | 23,03% |
| QBER E | 5,68% | 5,681% |
| K_U (Ewa zna niedopasowanie) | 1131 bitów | 1130,9 bitów |
| K_L (Alice i Bob nie widzą ataku) | 1297 bitów | 1296,8 bitów |
| informacja Ewy na wykryty bit | — | 0,293 |

Krzywe Gaussa dopasowane do Tabeli I(a) (Δ = 100 ps, równe szczyty, po odjęciu ciemnych zliczeń): środki -51,8 i 48,2 ps, szerokości 150,4 i 140,1 ps względem nominalnego przyjścia (przecięcia krzywych); przesunięcia Ewy A = -354,9 ps, B = 395,1 ps.

## Te same krzywe w łączu GYS (75 km, μ = 0,48, 1·10^8 impulsów)

| Wielkość | Bez ataku | Atak (model) | Atak (MC) | Atak + obrona (MC) |
| --- | --- | --- | --- | --- |
| zysk w zgodnej bazie | 5,763e-04 | 5,763e-04 | 5,736e-04 | 5,751e-04 |
| QBER | 3,44% | 3,44% | 3,44% | 3,52% |
| informacja Ewy na wykryty bit | 0 | 0,314 | 0,313 ± 0,005 | 0,000 ± 0,000 |
| K_U / K_L na wykryty bit | — | 0,423 / 0,225 | 0,424 / 0,225 | — |

Ewa przesuwa impuls do A z p-stwem 23,0% i tłumi do 63,4% za linią bezstratną. Spadek sprawności przy przesunięciu ukrywa od 65,6 km; bliżej Bob widzi niższy zysk.

## Od jakiego Δ atak łamie bezpieczeństwo (K_L > K_U), rodzina symetryczna

| μ | Δ* | Krzywe Zhao (Δ = 100 ps): K_U / K_L na wykryty bit |
| --- | --- | --- |
| 0,10 | 89 ps | 0,400 / 0,423 (atak łamie) |
| 0,48 | 140 ps | 0,423 / 0,225 (atak nie łamie) |

Przy większym μ GLLP odrzuca więcej klucza z impulsów wielofotonowych, więc time-shift musi dać Ewie więcej, zanim przekroczy to, co Alice i Bob i tak usuwają.
