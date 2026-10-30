## Scenariusze MITM (GYS, Ewa w połowie 25 km)

| Scenariusz | Przewidywanie | Sesja Alice ↔ Ewa | Sesja Ewa ↔ Bob | Zgodne |
| --- | --- | --- | --- | --- |
| (a) bez uwierzytelniania | MITM udany | klucz 6078 bitów, QBER 2,76% | klucz 5147 bitów, QBER 3,88% | tak |
| (b) Wegman–Carter | wykryty | przerwanie przy wiadomości nr 336 (transkrypt bob) | przerwanie przy wiadomości nr 391 (transkrypt alice) | tak |
| (c) ML-DSA-65 | wykryty | przerwanie przy wiadomości nr 336 (transkrypt bob) | przerwanie przy wiadomości nr 391 (transkrypt alice) | tak |
| (d) ML-DSA-65 złamane w trakcie sesji | MITM udany | klucz 6078 bitów, QBER 2,76% | klucz 5147 bitów, QBER 3,88% | tak |
| (f) klucz Ewy w katalogu ofiary | MITM udany | klucz 6078 bitów, QBER 2,76% | klucz 5147 bitów, QBER 3,88% | tak |
| (g) hybryda WC + ML-DSA-65, ML-DSA złamane w trakcie | wykryty | przerwanie przy wiadomości nr 336 (transkrypt bob) | przerwanie przy wiadomości nr 391 (transkrypt alice) | tak |

## Złamanie po sesji: QKD a ML-KEM

| Scenariusz | Co Ewa ma | Co Ewa wie o kluczu sesji |
| --- | --- | --- |
| (e) ML-DSA-65 złamane po sesji | transkrypt i klucze prywatne ML-DSA | 0 z 5662 bitów (ujawniony transkrypt to 2605 niezależnych funkcji liniowych klucza przed PA) |
| ML-KEM-768 złamane po sesji (HNDL) | szyfrogram i klucz dekapsulacji | 256 z 256 bitów sekretu |
| hybryda KDF(QKD ‖ ML-KEM), HNDL | sekret ML-KEM, bez klucza QKD | nic: klucz KDF różny od zgadywania |

## Narzut uwierzytelniania w uczciwej rundzie (25 km, 10 000 000 impulsów)

| Metoda | Tag | Tagi i podpisy | Klucz zużyty [bit] | Łącze klasyczne [B] | CPU [ms] |
| --- | --- | --- | --- | --- | --- |
| Wegman–Carter | na transkrypt | 2 | 183 | 16 | 4,3 |
| Wegman–Carter | na wiadomość | 982 | 59 963 | 7 856 | 10,5 |
| ML-DSA-65 | na transkrypt | 2 | 0 | 6 618 | 1,5 |
| ML-DSA-65 | na wiadomość | 982 | 0 | 3 249 438 | 970,9 |
| hybryda WC + ML-DSA-65 | na transkrypt | 2 | 183 | 6 634 | 7,5 |

Klucz wytworzony w tej rundzie: 15 308 bitów. Wegman–Carter liczy w kolumnie CPU jedno wyliczenie tagu (odbiorca powtarza je przy weryfikacji); ML-DSA podpis i weryfikację. Pierwsza runda Wegmana–Cartera zużywa dodatkowo 61 bitów na klucz haszu, potem tylko klucz jednorazowy na tag.

Decyzja z F4.16: podpis na transkrypt. Podpis na wiadomość kosztuje 491 razy więcej bajtów (3 249 438 B, czyli więcej niż sam klucz) i 635 razy więcej czasu, a bezpieczeństwo jest takie samo: Ewa i tak zostaje wykryta przed użyciem klucza, bo klucz powstaje dopiero po sprawdzeniu transkryptu.

## Czas na transkrypcie Alice z tej rundy (mediana, ta maszyna)

Transkrypt: 978 208 bitów.

| Metoda | Klucz publiczny [B] | Tag/podpis [B] | Tag/podpis [µs] | Weryfikacja [µs] |
| --- | --- | --- | --- | --- |
| Wegman–Carter | — | 8 | 4163 | 4163 |
| ML-DSA-44 | 1312 | 2420 | 578 | 303 |
| ML-DSA-65 | 1952 | 3309 | 866 | 355 |
| ML-DSA-87 | 2592 | 4627 | 974 | 433 |

Hasz Wegmana–Cartera jest tu pętlą Pythona po 60-bitowych kawałkach (schemat Hornera), więc jego czas mówi o implementacji, a nie o metodzie: w C albo w sprzęcie taki hasz liczy się z przepustowością rzędu GB/s. Czas ML-DSA pochodzi z OpenSSL przez `cryptography` i obejmuje skrót SHAKE256 całego transkryptu; podpis ma zmienny czas, bo losuje, aż przejdzie test odrzucenia (rejection sampling).

## Zasięg z dodatnim bilansem klucza na rundę

| Impulsy na rundę N | Wegman–Carter (122 bity na rundę) | ML-DSA (0 bitów) |
| --- | --- | --- |
| 1 000 000 | 60,5 km | 140,5 km |
| 10 000 000 | 104,5 km | 140,5 km |
| 100 000 000 | 132,0 km | 140,5 km |

ML-DSA nie zużywa klucza, więc bilans jest dodatni wszędzie, gdzie R(L) > 0. Ceną jest założenie obliczeniowe, które musi stać w czasie sesji (scenariusz d), i kilobajty na rundę zamiast 16 B.

## Modele zagrożeń (F4.20)

| Zagrożenie | QKD + ML-DSA | ML-KEM-768 | Hybryda KDF(QKD ‖ ML-KEM) |
| --- | --- | --- | --- |
| Harvest now, decrypt later (Mosca: x + y > z) | odporne: podpis musi wytrzymać tylko czas sesji, a klucz nie zależy od założeń obliczeniowych; po wycieku klucza podpisu Ewa zna 0 bitów (e) | podatne: nagrany szyfrogram (1088 B) i późniejsze złamanie dają 256/256 bitów sekretu | odporne, dopóki tajny jest klucz QKD |
| Przyszłe złamanie problemów kratowych | przeszłe klucze bezpieczne (e); przyszłe sesje można uwierzytelniać Wegmanem–Carterem kluczem z poprzednich rund | wszystkie nagrane sesje odkryte | przeszłe klucze bezpieczne, zostaje składowa QKD |
| Złamanie podpisu w czasie rzeczywistym | MITM udany (d); z hybrydą WC + ML-DSA wykryty (g) | MITM udany: ML-KEM też wymaga uwierzytelnienia (np. ML-DSA w TLS) | bezpieczna, jeśli część QKD uwierzytelnia hybryda WC + ML-DSA (g) |
| Ataki na implementację | sprzęt: oślepianie (atak 2, e05), Trojan horse i time-shift (ataki 4, 5); obroną jest monitoring, a nie dowód | oprogramowanie: kanały boczne, np. KyberSlash (czas dzielenia zależny od sekretu w implementacjach Kybera, 2024) | trzeba złamać obie implementacje naraz |
| Zasięg i infrastruktura | R(L) > 0 do ok. 140 km (GYS, decoy vacuum+weak, e06); osobny światłowód i detektory | bez limitu zasięgu, zwykła sieć; 1184 B klucza publicznego i 1088 B szyfrogramu | ograniczenia QKD plus koszt ML-KEM |
| Zaufane węzły pośrednie | dalej niż zasięg: łańcuch węzłów zaufanych, z których każdy zna klucz | end-to-end, bez węzłów zaufanych | węzeł zaufany zna tylko składową QKD; sekret ML-KEM jest end-to-end |
