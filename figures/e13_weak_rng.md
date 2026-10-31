## Znane ziarno (MT19937), GYS 25 km

| Atak | QBER | Bity znane Ewie na pewno |
| --- | --- | --- |
| bez ataku (model) | 3,31% | 0% |
| pasywny: Ewa odtwarza bity i bazy | 3,51% | 100,00% |
| intercept-resend w odtworzonej bazie | 3,25% | 99,97% |

Zysk przy ataku pasywnym: z = 0,39 względem modelu bez ataku. Ewa nie musi nawet dotykać światłowodu.

Ten sam generator w 100 ciągach po 1 000 000 bitów odrzucony na poziomie α = 0,01: monobit 1%, runs 1%, autokorelacja 1%. Testy statystyczne nie widzą przewidywalności.

## Obciążone bazy: Ewa mierzy zawsze w częstszej bazie

| p(X) | QBER (MC) | QBER (model) | Bity znane Ewie (MC) | Model |
| --- | --- | --- | --- | --- |
| 0,50 | 26,63% | 26,66% | 50,0% | 50,0% |
| 0,55 | 23,72% | 24,32% | 55,1% | 55,0% |
| 0,60 | 22,07% | 21,99% | 59,7% | 60,0% |
| 0,70 | 17,41% | 17,32% | 70,3% | 70,0% |
| 0,80 | 12,33% | 12,65% | 80,5% | 80,0% |
| 0,90 | 7,82% | 7,98% | 90,0% | 90,0% |

## Obciążone bity: deficyt entropii klucza

Przy p = 0,55 brakuje 0,0072 bitu entropii na bit (asymptotycznie, Shannon), czyli 3,3% klucza przy 25 km (0,221 bitu klucza na bit po siftingu). W rachunku min-entropii, istotnym dla skończonego bloku, deficyt to 0,1375 bitu, czyli więcej niż połowa klucza.

## Korelacje Markowa w bitach Alice (sifting zostawia 0,32% impulsów)

| s (surowe bity) | Sąsiednie bity klucza równe (MC) | Model | Deficyt przed / po siftingu |
| --- | --- | --- | --- |
| 0,900 | 0,5045 ± 0,0020 | 0,5064 | 0,531 / 0,0001 |
| 0,990 | 0,5678 ± 0,0019 | 0,5684 | 0,919 / 0,0135 |
| 0,999 | 0,8084 ± 0,0016 | 0,8073 | 0,989 / 0,2928 |

Sifting zostawia bity w odstępach geometrycznych, więc korelacja λ = 2s − 1 spada do rλ/(1 − (1 − r)λ). Na dłuższym łączu zostaje tylko korelacja bliska pełnej powtarzalności.

## Moc testów losowości (α = 0,01, 300 ciągów na punkt)

| Alternatywa | n | monobit | runs | autokorelacja |
| --- | --- | --- | --- | --- |
| obciążenie p = 0,55 | 100 | 5% (6%) | 2% (1%) | 1% (1%) |
| obciążenie p = 0,55 | 1000 | 73% (72%) | 22% (21%) | 1% (1%) |
| obciążenie p = 0,55 | 10000 | 100% (100%) | 100% (100%) | 5% (6%) |
| obciążenie p = 0,55 | 100000 | 100% (100%) | 100% (100%) | 69% (72%) |
| Markow s = 0,55 | 100 | 2% (2%) | 4% (6%) | 5% (6%) |
| Markow s = 0,55 | 1000 | 2% (2%) | 67% (72%) | 70% (72%) |
| Markow s = 0,55 | 10000 | 1% (2%) | 100% (100%) | 100% (100%) |
| Markow s = 0,55 | 100000 | 2% (2%) | 100% (100%) | 100% (100%) |

W nawiasach: przybliżenie normalne z analysis/weak_rng.py.
