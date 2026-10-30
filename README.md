# bb84-attacks

Symulator BB84 z atakami. Każdy wynik Monte Carlo jest porównywany ze wzorem analitycznym albo z wynikiem z publikacji.

Model: polaryzacyjny BB84, światłowód 1550 nm, źródło słabych impulsów koherentnych (WCP), dwa detektory SPAD po stronie Boba. Parametry startowe z eksperymentu GYS (Ma, Qi, Zhao, Lo 2005).

Pomost do kryptografii postkwantowej: kanał klasyczny można uwierzytelniać kluczem współdzielonym (Wegman–Carter), podpisem ML-DSA (FIPS 204) albo hybrydowo. Symulator pokazuje, kiedy takie uwierzytelnianie chroni klucz QKD, i porównuje QKD z ML-KEM w różnych modelach zagrożeń.

> **Status:** faza F1. Etap 1 (idealny BB84: `rng`, `pulses`, `protocol`, `metrics`) jest gotowy; realistyczny kanał i detektory (etap 2), ataki i wykresy kontrolne powstają w kolejnych krokach, patrz [ROADMAP.md](ROADMAP.md).

## Szybki start

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -e '.[dev,crosscheck]'
```

```bash
make check
```

`make check` uruchamia lintera (ruff), sprawdzanie typów (mypy) i szybkie testy. Pozostałe cele: `make test-slow` (testy statystyczne), `make figures` (wszystkie wykresy z `experiments/`), `make report` (raport LaTeX).

## Dokumentacja

- [ROADMAP.md](ROADMAP.md): fazy, bramki jakości, harmonogram
- [Plan projektu](<assets/Symulator BB84 z atakami – plan projektu.md>): założenia i teoria
- [docs/model.md](docs/model.md): założenia fizyczne, konwencje, parametry
- [docs/architecture.md](docs/architecture.md): tor impulsów i punkty ataku P1–P8
- [docs/adr/](docs/adr/): decyzje architektoniczne
- [docs/related.md](docs/related.md): istniejące symulatory i czym ten projekt się od nich różni

## Licencja

MIT, patrz [LICENSE](LICENSE). Artykuły w `assets/` mają własne licencje i nie są częścią repozytorium.
