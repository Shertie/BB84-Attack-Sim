## Odtworzenie Lim et al. 2014 (rys. 1 i tekst)

| Wielkość | Artykuł | Model |
| --- | --- | --- |
| zasięg przy n_X = 10⁴ | 135 km | 135,2 km |
| R(n_X = 10⁹) / R(n_X = 10⁷) przy 100 km | ≈ 1,75 | 1,780 |

Parametry z artykułu: η_Bob = 10%, p_dc = 6·10⁻⁷, p_ap = 4%, e_mis = 0,5%, f = 1,16, α = 0,2 dB/km, μ₃ = 2·10⁻⁴, ε_cor = 10⁻¹⁵, optymalizacja po (q, p₁, p₂, μ₁, μ₂). Liczby z tekstu odtwarza ε_sec = 10⁻¹⁰ i η_sys w modelu błędu (ROADMAP, K20 i K21).

| n_X | Zasięg (Lim) | Zasięg (GYS, symulator) |
| --- | --- | --- |
| 10^4 | 135,2 km | — |
| 10^5 | 162,3 km | 121,1 km |
| 10^6 | 173,2 km | — |
| 10^7 | 178,9 km | 138,8 km |
| 10^8 | 182,1 km | — |
| 10^9 | 184,2 km | 144,3 km |
| ∞ (trzy intensywności, bez poprawek) | 187,6 km | 148,0 km |

Dla porównania asymptotyczny decoy vacuum+weak z e06 (bazy 50/50, μ = 0,48, ν = 0,05): 140,5 km. Asymetryczny wybór baz i trzy intensywności dają w granicy asymptotycznej więcej, a skończony blok zabiera część zasięgu: przy n_X = 10^5 zasięg GYS jest krótszy o 27 km.

## Rachunek Lima na licznikach z Monte Carlo (GYS, n_X ≈ 10^5)

| L | Impulsy N | n_X (MC) | ℓ z modelu | ℓ z MC | Różnica |
| --- | --- | --- | --- | --- | --- |
| 10 km | 8,70·10^7 | 99838 | 16639 | 16220 | -2,5% |
| 25 km | 1,81·10^8 | 100181 | 16424 | 16138 | -1,7% |
| 50 km | 6,15·10^8 | 99995 | 15505 | 14786 | -4,6% |

Różnice są rzędu rozrzutu statystycznego: przy n_X = 10⁵ błędy w bazie testowej dla μ₂ to kilkaset zdarzeń, więc granica błędu fazowego waha się o kilka procent, a ℓ razem z nią.

Zastrzeżenie: dokumentacja `qkd` 0.2.0 powołuje się na Tupkary et al. (arXiv:2502.10340), według których oszacowanie błędu fazowego (5) nie jest ścisłą granicą. Nie weryfikowaliśmy tego.
