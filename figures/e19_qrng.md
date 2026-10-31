## Surowe bity (5·10^7 taktów, μ = 1,0, η₀ = 10%)

| Generator | Bity na takt: MC / wzór | P(1): MC / wzór | z | P(bit = poprzedni) | H_min, k = 0 | H_min, k = 1 | H_min, k = 2 | H_min, k = 3 | H_min, k = 4 | p monobit | p runs | p autokorelacja |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| idealny | 0,09280 / 0,09279 | 0,5001 / 0,5000 | 0,3 | 0,500 | 0,995 | 0,991 | 0,988 | 0,984 | 0,977 | 2,0e-01 | 6,4e-02 | 6,4e-02 |
| η₁/η₀ = 0,9 | 0,08844 / 0,08848 | 0,4732 / 0,4731 | 0,4 | 0,501 | 0,920 | 0,918 | 0,914 | 0,909 | 0,900 | 0,0e+00 | 0,0e+00 | 4,3e-03 |
| τ = 10, p = 3% | 0,06459 / 0,06461 | 0,5002 / 0,5000 | 1,2 | 0,337 | 0,996 | 0,588 | 0,531 | 0,528 | 0,524 | 8,6e-01 | 0,0e+00 | 0,0e+00 |
| oba | 0,06255 / 0,06252 | 0,4819 / 0,4820 | -0,6 | 0,344 | 0,946 | 0,551 | 0,501 | 0,498 | 0,489 | 2,1e-286 | 0,0e+00 | 0,0e+00 |

Wzór: P(1) = c₁(1 − c₀)/(c₀(1 − c₁) + c₁(1 − c₀)) ze stacjonarnymi p-stwami kliknięcia c_i (e18); procesy kliknięć D0 i D1 są niezależne, więc wzór jest dokładny także z pamięcią. σ z rozrzutu średnich w 100 paczkach (bity są skorelowane). H_min: najczęstsza wartość z górną granicą ufności 99% (NIST SP 800-90B, 6.3.1) w najgorszym z 2ᵏ kontekstów k poprzednich bitów; H_min i testy na pierwszych 1 000 000 bitach, kalibracja ekstraktora na całej próbce.

Czas martwy odwraca bity (detektor, który kliknął, milknie na τ taktów), afterpuls je powtarza. Przy k = 0 korelacji nie widać; min-entropia z kontekstem spada do ok. ½.

## Ekstrakcja Toeplitzem (blok 65536, ε = 1·10^-10)

| Generator | H z kalibracji (k = 4) | Bity wyjścia na blok | p monobit | p runs | p autokorelacja | H_min, k = 0 | H_min, k = 1 | H_min, k = 2 | H_min, k = 3 | H_min, k = 4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| idealny | 0,989 | 64746 | 0,62 | 0,83 | 0,83 | 0,996 | 0,994 | 0,989 | 0,984 | 0,976 |
| η₁/η₀ = 0,9 | 0,914 | 59802 | 0,28 | 0,46 | 0,46 | 0,995 | 0,993 | 0,989 | 0,978 | 0,967 |
| τ = 10, p = 3% | 0,527 | 34477 | 0,93 | 0,13 | 0,13 | 0,997 | 0,993 | 0,989 | 0,984 | 0,975 |
| oba | 0,493 | 32253 | 0,46 | 0,41 | 0,41 | 0,996 | 0,993 | 0,991 | 0,983 | 0,969 |

m = ⌊nH − 2 log₂(1/ε)⌋ (lemat o resztkowym haszowaniu). Wyjście przechodzi testy i ma min-entropię taką jak idealny generator (spadek z k to kara za ufność przy 2ᵏ kontekstach).

## Atak 8 z QRNG: bazy Alice z generatora η₁/η₀ = 0,7 (GYS 10 km, μ = 0,48)

| Przebieg | QBER: MC / model | z | Bity znane Ewie: MC / model | z |
| --- | --- | --- | --- | --- |
| bez Ewy, QRNG po ekstrakcji | 3,34% / 3,31% | 0,3 | 0,00% / 0,00% | 0,0 |
| intercept-resend w częstszej bazie, surowy QRNG | 22,33% / 22,45% | -0,4 | 58,66% / 59,00% | -1,1 |
| intercept-resend w częstszej bazie, QRNG po ekstrakcji | 26,94% / 26,65% | 1,0 | 50,14% / 49,99% | 0,4 |

Surowy generator daje P(X) = 0,410, więc Ewa mierzy w bazie Alice z p-stwem 0,590 zamiast ½ (model z e13). Po ekstrakcji (H = 0,752) przewaga znika, a QBER wraca do poziomu ataku w losowej bazie.

## Testy zdrowia NIST SP 800-90B (α = 2⁻²⁰)

- Generator „oba” (H = 0,493): progi RCT C = 42, APT C = 796 na 1024. Na 3 126 851 nowych bitach: alarmy APT w 0 z 3053 okien, RCT brak alarmu.
- Oślepiony D1 (η₁ = 0): wszystkie bity równe 0; RCT alarmuje przy bicie 41 (C − 1 = 41).
- Dryf sprawności D1 przy H = 0,914 z kalibracji generatora η₁/η₀ = 0,9 (APT C = 620), 2000 okien na punkt:

| η₁/η₀ | P(1) | P-stwo alarmu APT na okno: MC / wzór | z |
| --- | --- | --- | --- |
| 0,9 | 0,473 | 0,0000 / 0,0000 | 0,0 |
| 0,8 | 0,443 | 0,0000 / 0,0006 | -1,1 |
| 0,7 | 0,410 | 0,1070 / 0,1017 | 0,8 |
| 0,6 | 0,373 | 0,5985 / 0,5856 | 1,2 |
| 0,5 | 0,331 | 0,6645 / 0,6694 | -0,5 |

Wzór: p·P(Bin(W − 1, p) ≥ C − 1) + (1 − p)·P(Bin(W − 1, 1 − p) ≥ C − 1). Test w wersji z normy patrzy tylko na wartość pierwszej próbki okna, więc przy dużym dryfie wykrywa ok. P(częstszej wartości) okien; po kilku oknach alarm jest pewny.
