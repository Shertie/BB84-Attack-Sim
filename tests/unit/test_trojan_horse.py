"""Atak 5, Trojan horse: wiedza Ewy o bicie i bazie, izolacja, atak w torze."""

import math

import numpy as np
import pytest

from conftest import SEED
from golden import (
    GISIN_INFORMATION_COHERENT,
    GISIN_INFORMATION_RANDOM_PHASE,
    GISIN_MU,
    GISIN_ROUNDING,
)
from qkdsim.analysis.trojan_horse import (
    basis_guess_probability,
    bit_guess_probability,
    bit_information,
    helstrom_probability,
    lidt_probe_photons,
    probe_density,
    required_isolation_db,
    resend_qber,
    returned_photons,
    round_trip_attenuation_db,
    state_guess,
    state_guess_certificate,
    tolerable_photons,
)
from qkdsim.attacks.trojan_horse import TrojanHorse, modulator_bit
from qkdsim.optics import polarization_angle
from qkdsim.pulses import Pulses


def test_reproduces_gisin_figure_7() -> None:
    assert bit_information(GISIN_MU) == pytest.approx(
        GISIN_INFORMATION_COHERENT, abs=GISIN_ROUNDING
    )
    assert bit_information(GISIN_MU, phase_randomized=True) == pytest.approx(
        GISIN_INFORMATION_RANDOM_PHASE, abs=GISIN_ROUNDING
    )


def test_small_probe_limits() -> None:
    """I ≈ μ/ln 2 ze stałą fazą i I ≈ μ z losową: losowa faza zmniejsza wyciek 1/ln 2 razy."""
    mu = 1e-6
    assert bit_information(mu) == pytest.approx(mu / math.log(2), rel=1e-4)
    assert bit_information(mu, phase_randomized=True) == pytest.approx(mu, rel=1e-5)


@pytest.mark.parametrize("mu", [0.01, 0.3, 2.0])
@pytest.mark.parametrize("random_phase", [False, True])
def test_fock_helstrom_matches_closed_form(mu: float, random_phase: bool) -> None:
    rho_h = probe_density(mu, 0.0, random_phase, 40)
    rho_v = probe_density(mu, math.pi / 2, random_phase, 40)
    assert np.trace(rho_h) == pytest.approx(1.0, abs=1e-12)
    numeric = helstrom_probability(rho_h, rho_v)
    assert numeric == pytest.approx(bit_guess_probability(mu, random_phase), abs=1e-12)


def test_basis_leaks_much_less_with_random_phase() -> None:
    """Bez odniesienia fazowego pojedynczy foton nic nie mówi o bazie: przewaga ~ μ²/8."""
    mu = 0.01
    coherent = basis_guess_probability(mu) - 0.5
    random = basis_guess_probability(mu, phase_randomized=True) - 0.5
    assert random / mu**2 == pytest.approx(1 / 8, rel=0.01)
    assert coherent / math.sqrt(mu) == pytest.approx(0.27, rel=0.02)
    assert basis_guess_probability(0.0) == 0.5


def test_isolation_budget() -> None:
    assert round_trip_attenuation_db(60.0, 20.0, 10.0) == 100.0
    assert returned_photons(1e9, 100.0) == pytest.approx(0.1)
    isolation = required_isolation_db(1e9, 1e-3, reflection_db=20.0, attenuator_db=5.0)
    mu = returned_photons(1e9, round_trip_attenuation_db(isolation, 20.0, 5.0))
    assert bit_information(mu) == pytest.approx(1e-3, rel=1e-9)
    assert tolerable_photons(1e-3, phase_randomized=True) > tolerable_photons(1e-3)
    with pytest.raises(ValueError):
        tolerable_photons(1.5)


def test_modulator_setting_decodes_all_four_states() -> None:
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)
    bases = np.array([0, 0, 1, 1], dtype=np.uint8)
    pulses = Pulses.build(
        n=np.ones(4), bit=bits, basis=bases, theta=polarization_angle(bits, bases)
    )
    assert np.array_equal(modulator_bit(pulses), bits)


@pytest.mark.parametrize("random_phase", [False, True])
def test_attack_leaves_signal_and_samples_eves_measurement(random_phase: bool) -> None:
    rng = np.random.default_rng(SEED)
    size = 200_000
    bits, bases = rng.integers(0, 2, size, dtype=np.uint8), rng.integers(0, 2, size, dtype=np.uint8)
    pulses = Pulses.build(
        n=rng.poisson(0.48, size), bit=bits, basis=bases, theta=polarization_angle(bits, bases)
    )
    attack = TrojanHorse(1e9, 100.0, phase_randomized=random_phase)
    out, log = attack.intercept(pulses, rng)
    assert out is pulses
    mu = attack.returned_photons
    if random_phase:
        assert log.measured.mean() == pytest.approx(1 - math.exp(-mu), abs=4e-3)
        assert np.array_equal(log.bit[log.measured], bits[log.measured])
    else:
        assert (log.bit == bits).mean() == pytest.approx(bit_guess_probability(mu), abs=4e-3)
        assert (log.basis == 255).all()


def test_lidt_bound_reproduces_lucamarini_table_i() -> None:
    """μ_out = Nγ/f_A z N = 10²⁰/s: μ_out = 10⁻⁶ wymaga 170, 200 i 230 dB (Tabela I artykułu)."""
    for clock, gamma in ((1e9, 170.0), (1e6, 200.0), (1e3, 230.0)):
        assert returned_photons(lidt_probe_photons(clock), gamma) == pytest.approx(1e-6)
    assert round_trip_attenuation_db(60.0, 40.0, 35.0) == 170.0  # pierwszy wiersz tabeli


@pytest.mark.parametrize("mu", [0.01, 0.3, 1.0, 4.0])
def test_four_state_measurement_is_certified_optimal(mu: float) -> None:
    """Warunki Holevo–Yuena–Kennedy’ego–Laxa: Γ − ¼|ψ_j⟩⟨ψ_j| ⪰ 0; nie gorzej niż pierwiastkowy."""
    guess = state_guess(mu)
    assert state_guess_certificate(guess) > -1e-7
    assert np.allclose(guess.confusion.sum(axis=1), 1.0)
    assert guess.probability >= float(np.mean(np.diag(guess.states) ** 2)) - 1e-12
    assert guess.probability <= basis_guess_probability(mu) + 1e-9  # stan wyznacza bazę


def test_four_state_limits_and_resend_qber() -> None:
    weak, strong = state_guess(1e-4), state_guess(30.0)
    assert weak.probability == pytest.approx(0.25, abs=0.01)
    assert strong.probability == pytest.approx(1.0, abs=1e-6)
    assert resend_qber(weak) == pytest.approx(0.5, abs=0.01)  # zgadywanie: ¼·1 + ½·½
    assert resend_qber(strong, misalignment=0.033) == pytest.approx(0.033, abs=1e-3)
