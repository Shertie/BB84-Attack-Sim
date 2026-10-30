"""Uwierzytelnianie kanału klasycznego: Wegman–Carter (F4.12, ADR 0006).

Tag wiadomości m: t = (h_k(m) + r) mod p, gdzie
- h_k to hasz wielomianowy nad GF(p): kawałki m to współczynniki wielomianu w punkcie k
  (plus długość, żeby wiadomości różnej długości się nie zderzały);
- k to klucz haszu ze wspólnej puli Alice i Boba, jeden na całą sesję;
- r to świeży klucz jednorazowy na każdy tag.

Dla m' ≠ m różnica h_k(m') − h_k(m) to niezerowy wielomian stopnia ≤ L (L = liczba kawałków),
więc ma najwyżej L pierwiastków: fałszerstwo udaje się z p-stwem ≤ L/p, także po zobaczeniu
prawdziwej pary (m, t), bo r ukrywa h_k(m). Domyślnie p = 2⁶¹ − 1 (liczba pierwsza Mersenne'a);
mniejsze p służą do pokazania p-stwa fałszerstwa w Monte Carlo.

`AuthenticatedChannel` tagi wiadomości albo całego transkryptu rundy (`close_round`). Wiadomości
od `impostor` (Ewa udająca jedną ze stron) dostają podrobiony tag; nieudana weryfikacja
przerywa protokół wyjątkiem `AuthenticationFailure`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from qkdsim.classical import ClassicalChannel, ClassicalHook, Message
from qkdsim.rng import Bits

MERSENNE_61 = 2**61 - 1


class AuthenticationFailure(Exception):
    """Tag nie przeszedł weryfikacji: protokół jest przerywany."""

    def __init__(self, message_index: int, kind: str) -> None:
        super().__init__(f"nieudana weryfikacja wiadomości nr {message_index} ({kind})")
        self.message_index = message_index
        self.kind = kind


@dataclass
class KeyPool:
    """Wspólny klucz Alice i Boba (wcześniej współdzielony albo z poprzednich rund QKD)."""

    bits: Bits
    used: int = 0

    def take(self, n: int) -> int:
        if self.used + n > len(self.bits):
            raise RuntimeError("wyczerpana pula klucza do uwierzytelniania")
        chunk = self.bits[self.used : self.used + n]
        self.used += n
        return int.from_bytes(np.packbits(chunk, bitorder="little").tobytes(), "little")

    @property
    def remaining(self) -> int:
        return len(self.bits) - self.used


def message_bits(message: Message) -> Bits:
    """Deterministyczna serializacja wiadomości (rodzaj, nadawca, ładunek) do bitów."""
    parts = [message.sender.encode(), message.kind.encode()]
    payload = message.payload
    items = payload if isinstance(payload, list | tuple) else [payload]
    for item in items:
        if isinstance(item, np.ndarray):
            parts.append(np.ascontiguousarray(item).tobytes())
        elif item is not None:
            parts.append(repr(item).encode())
    raw = b"\x00".join(parts)
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))


class Authenticator(Protocol):
    """Metoda uwierzytelniania kanału klasycznego (ADR 0006).

    `genuine`: prawdziwy nadawca taguje dane, a odbiorca je weryfikuje (zużywa zasoby).
    `impostor_passes`: najlepsza próba Ewy udającej `sender`; True, jeśli odbiorca ją przyjmie.
    """

    @property
    def tags_used(self) -> int: ...

    @property
    def key_bits_used(self) -> int: ...

    @property
    def bytes_sent(self) -> int: ...

    def genuine(self, sender: str, bits: Bits) -> None: ...

    def impostor_passes(self, sender: str, bits: Bits, rng: np.random.Generator) -> bool: ...


@dataclass
class WegmanCarter:
    """Hasz wielomianowy nad GF(p) z kluczem k plus klucz jednorazowy r na tag."""

    pool: KeyPool
    prime: int = MERSENNE_61
    key: int = field(init=False)
    tags_used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.key = self.pool.take(self.prime.bit_length()) % self.prime

    @property
    def chunk_bits(self) -> int:
        """Kawałek wiadomości musi być mniejszy od p."""
        return self.prime.bit_length() - 1

    @property
    def tag_bits(self) -> int:
        return self.prime.bit_length()

    def chunks(self, bits: Bits) -> int:
        return -(-len(bits) // self.chunk_bits)

    def polynomial_hash(self, bits: Bits) -> int:
        size = self.chunk_bits
        padded = np.concatenate([bits, np.zeros((-len(bits)) % size, dtype=np.uint8)])
        weights = 1 << np.arange(size, dtype=np.uint64)
        values = (padded.reshape(-1, size).astype(np.uint64) * weights).sum(axis=1)
        acc = 0
        for value in values.tolist():
            acc = (acc * self.key + value) % self.prime
        return (acc * self.key + len(bits)) % self.prime

    def next_pad(self) -> int:
        self.tags_used += 1
        return self.pool.take(self.prime.bit_length()) % self.prime

    def tag(self, bits: Bits) -> int:
        return (self.polynomial_hash(bits) + self.next_pad()) % self.prime

    def verify(self, bits: Bits, tag: int) -> bool:
        """Odbiorca liczy oczekiwany tag z tym samym, następnym kluczem jednorazowym."""
        return self.tag(bits) == tag

    @property
    def key_bits_used(self) -> int:
        return self.pool.used

    @property
    def bytes_sent(self) -> int:
        return self.tags_used * -(-self.tag_bits // 8)

    def genuine(self, sender: str, bits: Bits) -> None:
        self.tag(bits)  # nadawca zużywa klucz jednorazowy, odbiorca sprawdza go tak samo

    def impostor_passes(self, sender: str, bits: Bits, rng: np.random.Generator) -> bool:
        # Ewa nie zna ani k, ani r: najlepsze, co może, to zgadnąć tag.
        return self.verify(bits, int(rng.integers(0, self.prime)))


class AuthenticatedChannel(ClassicalChannel):
    """Kanał klasyczny z tagiem na wiadomość albo na transkrypt rundy."""

    def __init__(
        self,
        auth: Authenticator,
        mode: Literal["message", "transcript"] = "transcript",
        impostor: str | None = None,
        forger: np.random.Generator | None = None,
        hooks: Sequence[ClassicalHook] = (),
    ) -> None:
        super().__init__(hooks)
        self.auth = auth
        self.mode = mode
        self.impostor = impostor
        self.forger = forger if forger is not None else np.random.default_rng()
        self.tags_sent = 0
        self._pending: dict[str, list[Bits]] = {"alice": [], "bob": []}

    def _check(self, sender: str, bits: Bits, kind: str) -> None:
        self.tags_sent += 1
        if sender == self.impostor:
            if not self.auth.impostor_passes(sender, bits, self.forger):
                raise AuthenticationFailure(len(self.transcript), kind)
        else:
            self.auth.genuine(sender, bits)

    def send(self, message: Message) -> Message:
        bits = message_bits(message)
        if self.mode == "message":
            self._check(message.sender, bits, message.kind)
        else:
            self._pending[message.sender].append(bits)
        delivered = super().send(message)
        return delivered

    def close_round(self) -> None:
        """Tryb transkryptu: każda strona taguje wszystko, co wysłała w tej rundzie."""
        if self.mode != "transcript":
            return
        for sender, pieces in self._pending.items():
            if pieces:
                self._check(sender, np.concatenate(pieces), f"transkrypt {sender}")
        self._pending = {"alice": [], "bob": []}
