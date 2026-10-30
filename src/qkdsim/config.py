"""Ścisłe wczytywanie konfiguracji YAML do zamrożonych dataclass (ADR 0005).

Nieznany klucz, brakujący klucz bez wartości domyślnej i zły typ to błąd. Literówka w YAML
nie może po cichu dać wartości domyślnej. Zakresy wartości sprawdzają `__post_init__`
poszczególnych klas (np. WCPSource, FiberChannel).
"""

import dataclasses
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Błąd konfiguracji ze ścieżką do problematycznego klucza."""


def load_yaml(path: str | Path, cls: type[T]) -> T:
    """Wczytuje plik YAML do dataclass `cls`."""
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return from_mapping(cls, data, where=str(path))


def from_mapping(cls: type[T], data: object, where: str = "config") -> T:
    """Buduje dataclass `cls` ze słownika, sprawdzając klucze i typy rekurencyjnie."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} nie jest dataclass")
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: oczekiwany słownik, jest {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{where}: nieznane klucze {unknown}; dozwolone: {sorted(fields)}")
    values: dict[str, Any] = {}
    for name, field in fields.items():
        if name in data:
            values[name] = _convert(hints[name], data[name], f"{where}.{name}")
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{where}: brak wymaganego klucza '{name}'")
    try:
        return cls(**values)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{where}: {error}") from error


def _convert(hint: Any, value: object, where: str) -> Any:
    origin = typing.get_origin(hint)
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        return from_mapping(hint, value, where)
    if origin is tuple:
        args = typing.get_args(hint)
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{where}: oczekiwana lista, jest {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigError(f"{where}: oczekiwane {len(args)} elementy, jest {len(value)}")
        return tuple(
            _convert(a, v, f"{where}[{i}]")
            for i, (a, v) in enumerate(zip(args, value, strict=True))
        )
    if origin in (typing.Union, types.UnionType):
        for option in typing.get_args(hint):
            try:
                return _convert(option, value, where)
            except ConfigError:
                continue
        raise ConfigError(f"{where}: wartość {value!r} nie pasuje do {hint}")
    if hint is type(None):
        if value is not None:
            raise ConfigError(f"{where}: oczekiwane null, jest {value!r}")
        return None
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: oczekiwane true/false, jest {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: oczekiwana liczba całkowita, jest {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where}: oczekiwana liczba, jest {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: oczekiwany napis, jest {value!r}")
        return value
    raise TypeError(f"{where}: nieobsługiwany typ pola {hint!r}")
