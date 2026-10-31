"""Uwierzytelnianie postkwantowe: ML-DSA, scenariusze MITM i „everlasting security”."""

from typing import Any

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric import mldsa, mlkem
from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87

from conftest import SEED
from qkdsim.analysis.key_rate import GYS
from qkdsim.attacks.mitm import run_mitm
from qkdsim.channel import FiberChannel
from qkdsim.detector import DetectorPair, dark_count_per_detector
from qkdsim.pipeline import Link, run_sifted
from qkdsim.postprocessing.auth import AuthenticatedChannel, KeyPool, WegmanCarter
from qkdsim.postprocessing.hybrid_key import hybrid_key
from qkdsim.postprocessing.leakage import linear_eve_knowledge
from qkdsim.postprocessing.pqc_auth import (
    CONTEXT,
    FIPS_204_SIZES,
    HybridAuth,
    generate,
    mldsa_auth,
)
from qkdsim.postprocessing.run import PostprocessingConfig, postprocess
from qkdsim.rng import RandomStreams
from qkdsim.source import WCPSource

LINK = Link(
    WCPSource((0.48,)),
    FiberChannel(12.5, GYS.alpha_db_per_km, GYS.misalignment),
    DetectorPair(GYS.eta_bob, dark_count_per_detector(GYS.background)),
)
PURE_PYTHON = {44: ML_DSA_44, 65: ML_DSA_65, 87: ML_DSA_87}
PUBLIC: dict[int, Any] = {
    44: mldsa.MLDSA44PublicKey,
    65: mldsa.MLDSA65PublicKey,
    87: mldsa.MLDSA87PublicKey,
}


@pytest.mark.parametrize("level", [44, 65, 87])
def test_sizes_match_fips_204(level: int) -> None:
    private, public = generate(level)
    assert len(public) == FIPS_204_SIZES[level][0]
    assert len(private.sign(b"m", CONTEXT)) == FIPS_204_SIZES[level][1]


@pytest.mark.parametrize("level", [44, 65, 87])
def test_cryptography_and_dilithium_py_interoperate(level: int) -> None:
    """Podpis jedną implementacją, weryfikacja drugą, w obie strony."""
    message = b"transkrypt rundy"
    private, public = generate(level)
    assert PURE_PYTHON[level].verify(public, message, private.sign(message, CONTEXT), ctx=CONTEXT)
    pk, sk = PURE_PYTHON[level].keygen()
    signature = PURE_PYTHON[level].sign(sk, message, ctx=CONTEXT)
    PUBLIC[level].from_public_bytes(pk).verify(signature, message, CONTEXT)  # rzuca, gdy zły


SCENARIOS = {
    "a": (None, False),
    "c": (lambda: mldsa_auth(65), True),
    "d": (lambda: mldsa_auth(65, "forgery_oracle"), False),
    "f": (lambda: mldsa_auth(65, "poisoned_directory"), False),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_mitm_scenarios(name: str) -> None:
    factory, detected = SCENARIOS[name]
    out = run_mitm(LINK, LINK, 600_000, SEED, authenticator=factory)
    assert out.aborted is detected


def test_hybrid_survives_a_real_time_signature_break() -> None:
    pool = np.random.default_rng(SEED).integers(0, 2, 100_000, dtype=np.uint8)

    def hybrid() -> HybridAuth:
        return HybridAuth(WegmanCarter(KeyPool(pool.copy())), mldsa_auth(65, "forgery_oracle"))

    out = run_mitm(LINK, LINK, 600_000, SEED, authenticator=hybrid)
    assert out.aborted


def test_signature_uses_no_shared_key_but_costs_bytes() -> None:
    out = run_mitm(
        LINK, LINK, 600_000, SEED, authenticator=lambda: mldsa_auth(65, "forgery_oracle")
    )
    side = out.alice_side
    assert side.key_bits_used == 0
    assert side.bytes_sent == 2 * FIPS_204_SIZES[65][1]  # po jednym podpisie na kierunek


def test_signature_broken_after_session_reveals_nothing_about_past_key() -> None:
    """Scenariusz (e): Ewa dostaje klucze prywatne po sesji; zna tylko podsłuchany transkrypt."""
    streams = RandomStreams.from_seed(SEED)
    key = run_sifted(LINK, 400_000, streams)
    auth = mldsa_auth(65)
    channel = AuthenticatedChannel(auth)
    result = postprocess(key, streams.public, PostprocessingConfig(record_rows=True), channel)
    assert auth.tags_used == 2
    # Po sesji klucz Alice wycieka: Ewa podpisuje odtąd jako Alice...
    stolen = auth.signers["alice"]
    forged = stolen.sign(b"przyszla sesja", CONTEXT)
    PUBLIC[65].from_public_bytes(auth.public_keys["alice"]).verify(
        forged, b"przyszla sesja", CONTEXT
    )
    # ...ale o kluczu z tej sesji wie tyle, ile z podsłuchanego transkryptu: nic.
    eve = linear_eve_knowledge(result.channel, result.balance.reconciled, result.pa_seed)
    assert eve.known_final_bits == 0


def test_mlkem_secret_falls_to_harvest_now_decrypt_later() -> None:
    """Kontrast: nagrany szyfrogram ML-KEM plus przyszłe złamanie = cały sekret."""
    decapsulation_key = mlkem.MLKEM768PrivateKey.generate()
    secret, recorded_ciphertext = decapsulation_key.public_key().encapsulate()
    assert decapsulation_key.decapsulate(recorded_ciphertext) == secret  # 256/256 bitów


def test_hybrid_key_survives_harvest_now_decrypt_later() -> None:
    """Ewa z sekretem ML-KEM (HNDL), ale bez klucza QKD, nie zna klucza hybrydowego."""
    qkd_key = np.random.default_rng(SEED).integers(0, 2, 256, dtype=np.uint8)
    decapsulation_key = mlkem.MLKEM768PrivateKey.generate()
    secret, recorded_ciphertext = decapsulation_key.public_key().encapsulate()
    key = hybrid_key(qkd_key, secret)
    assert hybrid_key(qkd_key, secret) == key
    stolen = decapsulation_key.decapsulate(recorded_ciphertext)
    assert hybrid_key(np.zeros_like(qkd_key), stolen) != key
    assert hybrid_key(qkd_key, bytes(32)) != key  # sam klucz QKD też nie wystarcza do zgadnięcia
