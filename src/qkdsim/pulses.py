"""Impulsy jako struktura tablic (ADR 0001).

Pulses niesie pełną prawdę o każdym impulsie. Funkcje fizyki czytają tylko pola fizyczne
(n, theta, t, power); bit, basis i intensity to etykiety Alice potrzebne do metryk.
Estymacje Alice i Boba nie przyjmują Pulses, tylko własne rekordy (protocol.py).
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import numpy.typing as npt

_DTYPES: dict[str, type[np.generic]] = {
    "n": np.int64,
    "bit": np.uint8,
    "basis": np.uint8,
    "intensity": np.uint8,
    "theta": np.float64,
    "t": np.float64,
    "power": np.float64,
}


@dataclass(frozen=True, eq=False)
class Pulses:
    """Tablice długości N, jedna pozycja na impuls. Tablice są tylko do odczytu."""

    n: npt.NDArray[np.int64]  # liczba fotonów
    bit: npt.NDArray[np.uint8]  # bit Alice
    basis: npt.NDArray[np.uint8]  # baza Alice: 0 = Z, 1 = X
    intensity: npt.NDArray[np.uint8]  # indeks w tabeli intensywności źródła
    theta: npt.NDArray[np.float64]  # kąt polaryzacji [rad]
    t: npt.NDArray[np.float64]  # przesunięcie czasu [ns]
    power: npt.NDArray[np.float64]  # moc klasyczna [W]; NaN dla impulsów kwantowych

    def __post_init__(self) -> None:
        size: int | None = None
        for field in dataclasses.fields(self):
            arr = getattr(self, field.name)
            if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                raise ValueError(f"Pulses.{field.name}: oczekiwana jednowymiarowa tablica numpy")
            if arr.dtype != _DTYPES[field.name]:
                raise TypeError(
                    f"Pulses.{field.name}: oczekiwany dtype {np.dtype(_DTYPES[field.name])}, "
                    f"jest {arr.dtype}"
                )
            if size is None:
                size = arr.shape[0]
            elif arr.shape[0] != size:
                raise ValueError(f"Pulses.{field.name}: długość {arr.shape[0]}, oczekiwana {size}")
            view = arr.view()
            view.flags.writeable = False
            object.__setattr__(self, field.name, view)

    def __len__(self) -> int:
        return int(self.n.shape[0])

    @classmethod
    def build(
        cls,
        *,
        n: npt.ArrayLike,
        bit: npt.ArrayLike,
        basis: npt.ArrayLike,
        theta: npt.ArrayLike,
        intensity: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        power: npt.ArrayLike | None = None,
    ) -> Self:
        """Tworzy impulsy z konwersją typów; brakujące pola dostają wartości domyślne."""
        n_arr = _convert(n, np.int64, "n")
        size = n_arr.shape[0]
        return cls(
            n=n_arr,
            bit=_convert(bit, np.uint8, "bit"),
            basis=_convert(basis, np.uint8, "basis"),
            theta=_convert(theta, np.float64, "theta"),
            intensity=(
                np.zeros(size, dtype=np.uint8)
                if intensity is None
                else _convert(intensity, np.uint8, "intensity")
            ),
            t=np.zeros(size) if t is None else _convert(t, np.float64, "t"),
            power=np.full(size, np.nan) if power is None else _convert(power, np.float64, "power"),
        )

    def replace(self, **changes: npt.NDArray[Any]) -> Self:
        """Nowy obiekt ze zmienionymi polami; oryginał zostaje bez zmian."""
        return dataclasses.replace(self, **changes)


def _convert(values: npt.ArrayLike, dtype: type[np.generic], name: str) -> npt.NDArray[Any]:
    """Konwersja typu, która nie zawija po cichu wartości spoza zakresu dtype."""
    source = np.asarray(values)
    converted = source.astype(dtype)
    if np.issubdtype(dtype, np.integer) and not np.array_equal(converted, source):
        raise ValueError(f"Pulses.{name}: wartości nie mieszczą się w {np.dtype(dtype)}")
    return converted
