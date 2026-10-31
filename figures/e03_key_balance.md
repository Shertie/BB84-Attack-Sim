## Bilans klucza (GYS, μ = 0,48)

| Krok | 25 km | 50 km |
| --- | --- | --- |
| Wysłane impulsy N | 100 000 000 | 300 000 000 |
| Bity po siftingu | 321 527 | 290 141 |
| Próbka do estymacji (ujawniona) | 32 153 | 29 014 |
| QBER z próbki | 3,18% | 3,38% |
| Klucz do korekcji n | 289 374 | 261 127 |
| Błędy przed korekcją | 9 534 | 9 012 |
| Wyciek Cascade leak_EC [bit] | 69 898 | 65 372 |
| Sprawność Cascade f | 1,156 | 1,156 |
| Wyciek weryfikacji [bit] | 64 | 64 |
| Udział detekcji jednofotonowych Q₁/Q_μ | 0,621 | 0,619 |
| Błąd jednofotonowy e₁ | 3,31% | 3,32% |
| Klucz końcowy ℓ [bit] | 72 020 | 62 247 |
| ℓ / N [bit na impuls] | 7,20e-04 | 2,07e-04 |
| Wzór asymptotyczny R(L) [bit na impuls] | 7,54e-04 | 2,23e-04 |
| Przy 2 MHz [bit/s] | 1 440 | 415 |

ℓ/N jest o kilka procent niższe od wzoru asymptotycznego. 10% bitów po siftingu idzie na próbkę (strata ×0,9), co tylko częściowo kompensuje Cascade z f ≈ 1,16 zamiast 1,22 przyjętego we wzorze. Udział jednofotonowych i e₁ pochodzą z modelu (granica decoy); estymatę decoy z danych Monte Carlo pokazuje e06.

## Pasywna Ewa i zapomniany wyciek

Klucz 1980 bitów z 25 km. Ewa zna wszystkie parzystości Cascade i hasz weryfikacji: rząd 654 niezależnych funkcji liniowych klucza.

| Długość klucza końcowego | ℓ [bit] | Bity znane Ewie z pewnością |
| --- | --- | --- |
| z odjęciem wycieku | 575 | 0 |
| bez odjęcia wycieku | 1338 | 12 |
