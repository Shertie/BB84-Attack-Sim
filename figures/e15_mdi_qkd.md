## Zasięg MDI-QKD (parametry z rys. 2 Lo, Curty, Qi 2012)

| Układ | Zasięg |
| --- | --- |
| MDI-QKD, przekaźnik pośrodku (artykuł: ponad 200 km) | 207,5 km |
| decoy BB84, te same detektory, bazy asymetryczne | 159,6 km |

Przekaźnik: sprawność 14,5%, ciemne zliczenia 3,01·10^-6 na detektor (tło 6,02·10^-6 z artykułu na parę detektorów), błąd optyki 1,5% jako dwa obroty po połowie, f = 1,16, 0,2 dB/km; nieskończenie wiele stanów decoy, μ_A = μ_B optymalne w każdym punkcie. Przy tle 6,02·10^-6 na detektor zasięg spada poniżej 200 km. W tym modelu MDI sięga dalej niż decoy BB84, ale nie dwa razy: artykuł porównuje z innym układem.

## Monte Carlo przekaźnika (μ_A = μ_B = 0,30, 2·10^7 par impulsów)

| L | Q_Z (MC / model) | E_Z (MC / model) | E_X (MC / model) |
| --- | --- | --- | --- |
| 0 km | 9,15·10^-4 / 9,23·10^-4 | 3,98% / 4,31% | 24,94% / 25,45% |
| 25 km | 2,96·10^-4 / 3,00·10^-4 | 4,26% / 4,31% | 26,27% / 25,57% |
| 50 km | 9,30·10^-5 / 9,66·10^-5 | 5,16% / 4,34% | 25,93% / 25,65% |
| 75 km | 3,42·10^-5 / 3,09·10^-5 | 4,09% / 4,39% | 24,44% / 25,70% |

Największe |z| w tabeli: 1,46. Przy jednej intensywności QBER w bazie X wynosi ok. 25% także bez ataku: to impulsy wielofotonowe; błąd fazowy liczy się z par pojedynczych fotonów (stany decoy).

## Ewa jako przekaźnik (25 km, μ = 0,30)

| Przekaźnik | QBER w bazie klucza | QBER w bazie testowej | Bity klucza znane Ewie |
| --- | --- | --- | --- |
| uczciwy | 3,80% | 25,54% | 0,0% |
| Ewa: pomiar Z, ogłasza ψ⁻ | 0,00% | 49,95% | 100,0% |

Ewa kontroluje cały przekaźnik, więc oślepianie czy time-shift nic jej nie dodają: detektory i tak są jej. Żeby znać klucz, musi mierzyć stany Alice i Boba, a pomiar w bazie Z daje w bazie X 50% błędów, wyraźnie ponad ~25% uczciwego przekaźnika przy WCP. Także dla par pojedynczych fotonów jej ogłoszenia nie zależą od bitów X, więc e¹¹_X = ½ i klucza nie ma. Dla porównania w BB84 oślepienie detektorów zostawia QBER bez zmian, a Ewa zna cały klucz (e05).
