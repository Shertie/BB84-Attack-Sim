"""Uwierzytelnianie podpisem ML-DSA (FIPS 204) i hybryda z Wegmanem–Carterem.

Backend: `cryptography` (≥ 47, ML-DSA z OpenSSL). Implementacja w czystym Pythonie
(`dilithium-py`) służy tylko do testu zgodności.

Model zaufania: każda strona ma katalog kluczy publicznych rozprowadzony z góry, bez PKI.
Odbiorca weryfikuje podpis kluczem z katalogu dla deklarowanego nadawcy. Scenariusze ataku
zmieniają tylko to, czym Ewa może podpisać i co jest w katalogu:
- „honest”: Ewa ma tylko własny klucz, więc jej podpis nie przechodzi;
- „forgery_oracle”: ML-DSA złamane w trakcie sesji (Ewa podpisuje jako ofiara);
- „poisoned_directory”: w katalogu ofiary zamiast klucza drugiej strony jest klucz Ewy (Z5b).
"""

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import mldsa

from qkdsim.postprocessing.auth import WegmanCarter
from qkdsim.rng import Bits

PrivateKey = mldsa.MLDSA44PrivateKey | mldsa.MLDSA65PrivateKey | mldsa.MLDSA87PrivateKey
Scenario = Literal["honest", "forgery_oracle", "poisoned_directory"]
CONTEXT = b"qkdsim/bb84-transcript"

_KEYS: dict[int, tuple[Any, Any]] = {
    44: (mldsa.MLDSA44PrivateKey, mldsa.MLDSA44PublicKey),
    65: (mldsa.MLDSA65PrivateKey, mldsa.MLDSA65PublicKey),
    87: (mldsa.MLDSA87PrivateKey, mldsa.MLDSA87PublicKey),
}
# FIPS 204, Tabela 2: rozmiary klucza publicznego i podpisu [B].
FIPS_204_SIZES = {44: (1312, 2420), 65: (1952, 3309), 87: (2592, 4627)}


def generate(level: int) -> tuple[PrivateKey, bytes]:
    """Klucz prywatny ML-DSA danego poziomu i surowy klucz publiczny."""
    private = _KEYS[level][0].generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return private, public


def transcript_bytes(bits: Bits) -> bytes:
    """Długość w bitach (8 B) + bity spakowane; długość chroni przed dopełnieniem zerami."""
    return len(bits).to_bytes(8, "little") + np.packbits(bits).tobytes()


@dataclass
class MLDSAAuth:
    """Podpis ML-DSA transkryptu za interfejsem `Authenticator`."""

    level: int
    signers: dict[str, PrivateKey]  # prawdziwe klucze prywatne stron
    public_keys: dict[str, bytes]  # prawdziwe klucze publiczne (uczciwa ścieżka)
    directory: dict[str, bytes]  # katalog ofiary: nim sprawdza wiadomości od rzekomej strony
    eve_signers: dict[str, PrivateKey]  # czym Ewa może podpisać jako dana strona
    tags_used: int = field(default=0, init=False)
    bytes_sent: int = field(default=0, init=False)
    cpu_seconds: float = field(default=0.0, init=False)

    @property
    def key_bits_used(self) -> int:
        return 0  # podpis nie zużywa wspólnego klucza

    def _sign_and_verify(self, signer: PrivateKey, public_key: bytes, bits: Bits) -> bool:
        data = transcript_bytes(bits)
        start = time.perf_counter()
        signature = signer.sign(data, CONTEXT)
        public = _KEYS[self.level][1].from_public_bytes(public_key)
        try:
            public.verify(signature, data, CONTEXT)
            accepted = True
        except InvalidSignature:
            accepted = False
        self.cpu_seconds += time.perf_counter() - start
        self.tags_used += 1
        self.bytes_sent += len(signature)
        return accepted

    def genuine(self, sender: str, bits: Bits) -> None:
        if not self._sign_and_verify(self.signers[sender], self.public_keys[sender], bits):
            raise RuntimeError("prawdziwy podpis nie przeszedł weryfikacji")

    def impostor_passes(self, sender: str, bits: Bits, rng: np.random.Generator) -> bool:
        return self._sign_and_verify(self.eve_signers[sender], self.directory[sender], bits)


def mldsa_auth(level: int = 65, scenario: Scenario = "honest") -> MLDSAAuth:
    """Klucze Alice, Boba i Ewy oraz katalog odbiorcy zgodnie ze scenariuszem."""
    alice, alice_pub = generate(level)
    bob, bob_pub = generate(level)
    eve, eve_pub = generate(level)
    signers = {"alice": alice, "bob": bob}
    public_keys = {"alice": alice_pub, "bob": bob_pub}
    directory = dict(public_keys)
    eve_signers: dict[str, PrivateKey] = {"alice": eve, "bob": eve}
    if scenario == "forgery_oracle":
        eve_signers = dict(signers)
    elif scenario == "poisoned_directory":
        directory = {"alice": eve_pub, "bob": eve_pub}
    return MLDSAAuth(level, signers, public_keys, directory, eve_signers)


@dataclass
class HybridAuth:
    """Wegman–Carter i ML-DSA razem: Ewa musi pokonać oba, żeby przejść weryfikację."""

    wegman_carter: WegmanCarter
    signature: MLDSAAuth

    @property
    def tags_used(self) -> int:
        return self.signature.tags_used

    @property
    def key_bits_used(self) -> int:
        return self.wegman_carter.key_bits_used

    @property
    def bytes_sent(self) -> int:
        return self.wegman_carter.bytes_sent + self.signature.bytes_sent

    def genuine(self, sender: str, bits: Bits) -> None:
        self.wegman_carter.genuine(sender, bits)
        self.signature.genuine(sender, bits)

    def impostor_passes(self, sender: str, bits: Bits, rng: np.random.Generator) -> bool:
        wc = self.wegman_carter.impostor_passes(sender, bits, rng)
        sig = self.signature.impostor_passes(sender, bits, rng)
        return wc and sig
