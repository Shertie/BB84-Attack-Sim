"""Hybrydowe łączenie kluczy: KDF(klucz QKD ‖ sekret ML-KEM) (F4.21, opcjonalne).

Klucz wynikowy pozostaje tajny, dopóki tajna jest choć jedna składowa (HKDF-SHA256 jako
ekstraktor; ten sam układ „Z ‖ T” co sekret hybrydowy w NIST SP 800-56C Rev. 2). Ewa, która
nagrała szyfrogram ML-KEM i po latach złamała kraty (HNDL), zna sekret ML-KEM, ale nie klucz
QKD, więc nie zna klucza wynikowego. Odwrotnie: atak na sprzęt QKD nie wystarcza, dopóki
stoi ML-KEM.
"""

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from qkdsim.postprocessing.pqc_auth import transcript_bytes
from qkdsim.rng import Bits

INFO = b"qkdsim/hybrid-key"


def hybrid_key(qkd_key: Bits, kem_secret: bytes, length: int = 32) -> bytes:
    """`length` bajtów klucza z klucza QKD (bity, z długością) i sekretu ML-KEM."""
    material = transcript_bytes(qkd_key) + kem_secret
    return HKDF(hashes.SHA256(), length, salt=None, info=INFO).derive(material)
