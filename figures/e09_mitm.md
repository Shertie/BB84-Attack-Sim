## Atak MITM (GYS, Ewa w połowie 25 km)

| Scenariusz | Sesja Alice ↔ Ewa | Sesja Ewa ↔ Bob |
| --- | --- | --- |
| bez uwierzytelniania | klucz 5429 bitów, QBER 3,56% | klucz 5662 bitów, QBER 3,24% |
| Wegman–Carter, tag na wiadomość | przerwanie przy wiadomości nr 2 (sample_bits) | przerwanie przy wiadomości nr 0 (sample_positions) |
| Wegman–Carter, tag na transkrypt | przerwanie przy wiadomości nr 400 (transkrypt bob) | przerwanie przy wiadomości nr 382 (transkrypt alice) |

Bez uwierzytelniania klucze Alice i Boba są różne (zgodność 51% bitów, jak dla losowych), a Ewa zna oba: w sesji z Alice była Bobem, w sesji z Bobem była Alice. QBER na obu odcinkach jest taki jak bez ataku.

Model liczby wiadomości na rundę przy 25 km i 10 000 000 impulsach: 971; w symulacji: 1 000.

| Impulsy na rundę N | Zasięg z dodatnim bilansem (tag na transkrypt) |
| --- | --- |
| 1 000 000 | 60,5 km |
| 10 000 000 | 104,5 km |
| 100 000 000 | 132,0 km |
