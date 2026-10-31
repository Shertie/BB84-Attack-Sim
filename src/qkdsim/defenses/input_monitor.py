"""Obrona przed atakiem 5: monitor światła wchodzącego do Alice (Gisin i in. 2006), rozdz. 7.1.

Ta sama fotodioda co w obronie przed oślepianiem (`PowerMonitor`: odczep i szum odczytu), tylko
na wejściu Alice. Sonda Ewy o μ_in fotonach na impuls przy częstości `repetition_hz` to średnia
moc P = μ_in·hν·f. Alarm, gdy średnia z `windows` odczytów przekroczy z_{1−α}·σ/√W; średnia
odczytów jest gaussowska, więc p-stwo wykrycia to Φ((tap·P − próg)/(σ/√W)). Największa sonda,
którą monitor przeoczy z p-stwem ≥ β, ogranicza μ_in, a izolacja ogranicza dalej μ_out
(analysis/trojan_horse.py).
"""

import math
from dataclasses import dataclass

from scipy.stats import norm

from qkdsim.defenses.power_monitor import PowerMonitor
from qkdsim.detector import PHOTON_ENERGY_J


@dataclass(frozen=True)
class InputMonitor:
    monitor: PowerMonitor
    repetition_hz: float = 2e6  # GYS (ROADMAP, K1)
    windows: int = 1000

    def _sigma(self) -> float:
        return self.monitor.noise_w / math.sqrt(self.windows)

    def alarm_level_w(self, false_alarm: float = 0.01) -> float:
        """Próg średniego odczytu przy p-stwie fałszywego alarmu `false_alarm`."""
        return float(norm.isf(false_alarm)) * self._sigma()

    def probe_power_w(self, probe_photons: float) -> float:
        return probe_photons * PHOTON_ENERGY_J * self.repetition_hz

    def detection_probability(self, probe_photons: float, false_alarm: float = 0.01) -> float:
        signal = self.monitor.tap_fraction * self.probe_power_w(probe_photons)
        return float(norm.sf((self.alarm_level_w(false_alarm) - signal) / self._sigma()))

    def largest_hidden_probe(self, missed: float = 0.01, false_alarm: float = 0.01) -> float:
        """Największe μ_in, które monitor przeoczy z p-stwem ≥ `missed`."""
        level = self.alarm_level_w(false_alarm) + float(norm.isf(missed)) * self._sigma()
        return level / (self.monitor.tap_fraction * PHOTON_ENERGY_J * self.repetition_hz)

    def auc(self, probe_photons: float) -> float:
        """Pole pod krzywą ROC dla średniej odczytów: Φ(d′/√2), d′ = tap·P/(σ/√W)."""
        shift = self.monitor.tap_fraction * self.probe_power_w(probe_photons) / self._sigma()
        return float(norm.cdf(shift / math.sqrt(2.0)))
