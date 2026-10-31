## BBM92 ze źródłem SPDC pośrodku (detektory GYS po obu stronach)

Zasięg BBM92: 286,1 km; decoy BB84 (vacuum+weak, e06): 140,5 km. Przypadkowa koincydencja wymaga ciemnych zliczeń po obu stronach naraz, więc źródło w środku sięga dużo dalej niż decoy BB84; ceną jest niska stopa, bo wielokrotne pary ograniczają jasność źródła (λ ≈ 0,04 pary na impuls).

| L | λ optymalne | Zysk (MC / model) | QBER (MC / model) |
| --- | --- | --- | --- |
| 0 km | 4,07·10^-2 | 1,84·10^-4 / 1,83·10^-4 | 6,51% / 6,55% |
| 30 km | 3,98·10^-2 | 4,15·10^-5 / 4,21·10^-5 | 6,69% / 6,56% |
| 60 km | 3,93·10^-2 | 1,02·10^-5 / 9,76·10^-6 | 6,08% / 6,56% |

## CHSH i device-independent (stan maksymalnie splątany)

| V | S > 2 od η = | Stopa DI > 0 od η = |
| --- | --- | --- |
| 1,00 | 82,84% | 92,31% |
| 0,99 | 83,33% | 92,81% |

Próg dla S > 2 to 2/(1 + √2) = 82,84%. Próg dla klucza device-independent (Acín i in. 2007) wychodzi 92,31%, wobec 92,4% w Pironio i in. 2009 (za dokumentacją `qkd`). 
| η | S, wszystkie rundy (MC / model) | S, koincydencje (MC) | QBER klucza (MC / model) |
| --- | --- | --- | --- |
| 0,75 | 1,715 / 1,716 | 2,828 | 18,79% / 18,75% |
| 0,80 | 1,886 / 1,890 | 2,826 | 15,81% / 16,00% |
| 0,85 | 2,090 / 2,089 | 2,829 | 12,89% / 12,75% |
| 0,90 | 2,314 / 2,311 | 2,830 | 9,01% / 9,00% |
| 0,95 | 2,560 / 2,558 | 2,830 | 4,72% / 4,75% |
| 1,00 | 2,833 / 2,828 | 2,833 | 0,00% / 0,00% |

## Fałszywe naruszenie CHSH (η = 0,10)

| Źródło | S, koincydencje | S, wszystkie rundy |
| --- | --- | --- |
| uczciwe pary | 2,835 | 1,650 |
| Ewa steruje detektorami | 2,821 | 1,201 |

Na samych koincydencjach Ewa sterująca detektorami (Gerhardt i in. 2011) daje S = 2√2 jak uczciwe pary, choć jest lokalnym modelem. Po odrzuceniu rund bez kliknięcia test CHSH zakłada fair sampling i przestaje cokolwiek dowodzić. Licząc wszystkie rundy, Ewa ma S ≤ 2, ale przy tej sprawności uczciwe pary też, więc device-independent nie zadziała.
