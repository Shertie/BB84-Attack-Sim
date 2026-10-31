"""Ścisłe wczytywanie konfiguracji."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from qkdsim.config import ConfigError, from_mapping, load_yaml
from qkdsim.source import WCPSource


@dataclass(frozen=True)
class Example:
    seed: int
    source: WCPSource
    lengths_km: tuple[float, ...]
    label: str = "domyślna"
    note: str | None = None


GOOD = {
    "seed": 1,
    "source": {"intensities": [0.48, 0.05], "probabilities": [0.9, 0.1]},
    "lengths_km": [0, 25.5],
}


def test_nested_dataclass_and_tuple_conversion() -> None:
    cfg = from_mapping(Example, GOOD)
    assert cfg.source == WCPSource((0.48, 0.05), (0.9, 0.1))
    assert cfg.lengths_km == (0.0, 25.5)
    assert isinstance(cfg.lengths_km[0], float)
    assert cfg.label == "domyślna"


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="nieznane klucze"):
        from_mapping(Example, GOOD | {"sede": 2})


def test_unknown_nested_key_is_rejected_with_path() -> None:
    bad = GOOD | {"source": {"intensities": [0.5], "probabilites": [1.0]}}
    with pytest.raises(ConfigError, match=r"config\.source"):
        from_mapping(Example, bad)


def test_missing_required_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="brak wymaganego klucza 'seed'"):
        from_mapping(Example, {k: v for k, v in GOOD.items() if k != "seed"})


@pytest.mark.parametrize(
    ("key", "value"), [("seed", 1.5), ("seed", True), ("lengths_km", "25"), ("lengths_km", ["a"])]
)
def test_wrong_types_are_rejected(key: str, value: object) -> None:
    with pytest.raises(ConfigError):
        from_mapping(Example, GOOD | {key: value})


def test_optional_field_accepts_null_and_string() -> None:
    assert from_mapping(Example, GOOD | {"note": None}).note is None
    assert from_mapping(Example, GOOD | {"note": "x"}).note == "x"


def test_range_errors_from_post_init_carry_path() -> None:
    bad = GOOD | {"source": {"intensities": [-1.0]}}
    with pytest.raises(ConfigError, match=r"config\.source.*nieujemne"):
        from_mapping(Example, bad)


def test_load_yaml_with_underscored_integers(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "seed: 1_000_000\nsource:\n  intensities: [0.48]\nlengths_km: [0, 50]\n", encoding="utf-8"
    )
    cfg = load_yaml(path, Example)
    assert cfg.seed == 1_000_000
