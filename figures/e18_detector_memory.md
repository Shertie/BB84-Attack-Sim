## Nasycenie detektora (4·10^6 bramek na punkt)

| Świeże kliknięcia f | τ = 10, p = 0% (MC / wzór) | τ = 100, p = 0% (MC / wzór) | τ = 0, p = 4% (MC / wzór) |
| --- | --- | --- | --- |
| 1·10^-4 | 9,200e-05 / 9,990e-05 | 1,028e-04 / 9,901e-05 | 1,147e-04 / 1,042e-04 |
| 3·10^-4 | 3,048e-04 / 2,991e-04 | 3,087e-04 / 2,913e-04 | 3,188e-04 / 3,125e-04 |
| 1·10^-3 | 1,006e-03 / 9,901e-04 | 9,327e-04 / 9,091e-04 | 1,019e-03 / 1,042e-03 |
| 3·10^-3 | 2,937e-03 / 2,913e-03 | 2,333e-03 / 2,308e-03 | 3,066e-03 / 3,125e-03 |
| 1·10^-2 | 9,190e-03 / 9,091e-03 | 4,989e-03 / 5,000e-03 | 1,030e-02 / 1,041e-02 |
| 3·10^-2 | 2,308e-02 / 2,308e-02 | 7,509e-03 / 7,500e-03 | 3,130e-02 / 3,121e-02 |
| 1·10^-1 | 4,998e-02 / 5,000e-02 | 9,090e-03 / 9,091e-03 | 1,039e-01 / 1,037e-01 |
| 3·10^-1 | 7,501e-02 / 7,500e-02 | 9,676e-03 / 9,677e-03 | 3,085e-01 / 3,086e-01 |

Największe |z| w tabeli: 2,14 (σ z liczby kliknięć). Czas martwy τ bramek obcina p-stwo kliknięcia do f/(1 + τf); przy f·τ ≥ 1 detektor jest martwy przez większość czasu. Afterpulsy dokładają p·f(1 − f) kliknięć, czyli w pierwszym rzędzie D·p_ap, jak w modelu Lima et al.

## Model Lima et al. 2014 a Monte Carlo (μ = 0,5, 4·10^7 impulsów na punkt)

Detektor ma p = 4% na lawinę, czyli łącznie p_ap = p/(1 − p) = 4,17% afterpulsów na kliknięcie. „Dokładny”: afterpuls jako niezależne ciemne zliczenie z p-stwem p·c; „Lim”: ich wzory z tym p_ap.

| L | Zysk: MC / dokładny / Lim | z | QBER: MC / dokładny / Lim | z | QBER bez afterpulsów |
| --- | --- | --- | --- | --- | --- |
| 0 km | 5,068·10^-2 / 5,071·10^-2 / 5,080·10^-2 | -1,0 | 2,42% / 2,45% / 2,48% | -1,9 | 0,51% |
| 25 km | 1,634·10^-2 / 1,633·10^-2 / 1,634·10^-2 | 0,2 | 2,50% / 2,47% / 2,48% | 0,9 | 0,52% |
| 50 km | 5,211·10^-3 / 5,196·10^-3 / 5,197·10^-3 | 1,4 | 2,52% / 2,49% / 2,49% | 0,6 | 0,54% |
| 100 km | 5,180·10^-4 / 5,219·10^-4 / 5,220·10^-4 | -1,1 | 2,45% / 2,59% / 2,59% | -0,9 | 0,72% |

Wzory Lima et al. są liniowe w p_ap: pomijają to, że afterpuls ginie, gdy ten sam detektor i tak klika, i że z fotonem w drugim detektorze daje podwójne kliknięcie. Przy L = 0 km (D ≈ 5%) zawyżają QBER o 0,04 pp, na dłuższych odcinkach różnica znika. Afterpulsy podnoszą QBER o ok. 2,0 pp (p_ap/(2(1 + p_ap))), czterokrotnie więcej niż cały błąd optyki 0,5% w tym modelu.
