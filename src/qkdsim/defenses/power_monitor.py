"""Obrona (a) przed oślepianiem: monitor mocy optycznej na wejściu Boba (F3.9).

Bob odprowadza ułamek `tap_fraction` światła do fotodiody i porównuje odczyt w każdym oknie
pomiarowym z progiem. Sygnał kwantowy daje moc rzędu femtowatów, więc odczyt bez ataku to sam
szum `noise_w`; ciągłe światło oślepiające (setki µW) daje odczyt o rzędy wielkości większy.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PowerMonitor:
    tap_fraction: float = 0.01  # ułamek mocy kierowany do monitora
    noise_w: float = 1e-8  # odchylenie standardowe odczytu w jednym oknie [W]

    def __post_init__(self) -> None:
        if not 0.0 < self.tap_fraction < 1.0 or self.noise_w <= 0.0:
            raise ValueError("wymagane 0 < tap_fraction < 1 i noise_w > 0")

    def readings(
        self, cw_power_w: float, windows: int, rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """Odczyty monitora w `windows` oknach przy ciągłej mocy `cw_power_w` na wejściu Boba."""
        return self.tap_fraction * cw_power_w + self.noise_w * rng.standard_normal(windows)
