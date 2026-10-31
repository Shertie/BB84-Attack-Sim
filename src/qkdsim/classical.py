"""Kanał klasyczny z transkryptem.

Każda wiadomość niesie dwie liczby:

- `size_bits`: koszt komunikacji;
- `leaked_bits`: ile bitów informacji o kluczu (po usunięciu próbki) ujawnia. Bazy i próbka
  do estymacji nie ujawniają nic o reszcie klucza; parzystości Cascade i hasz weryfikacyjny
  ujawniają po jednym bicie na bit.

Transkrypt to wszystko, co słyszy pasywna Ewa. Hooki `Message → Message` podpinają
atak MITM i uwierzytelnianie.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

ClassicalHook = Callable[["Message"], "Message"]


@dataclass(frozen=True)
class Message:
    """Wiadomość w kanale klasycznym: nadawca, rodzaj, rozmiar i bity ujawnione Ewie."""

    sender: str  # "alice" albo "bob"
    kind: str  # np. "sample", "cascade_parities", "cascade_binary", "verify_hash"
    size_bits: int
    leaked_bits: int = 0
    payload: Any = None

    def __post_init__(self) -> None:
        if self.sender not in ("alice", "bob"):
            raise ValueError(f"nieznany nadawca {self.sender!r}")
        if self.size_bits < 0 or self.leaked_bits < 0:
            raise ValueError("rozmiar i wyciek muszą być nieujemne")


class ClassicalChannel:
    """Kanał z transkryptem; wiadomości przechodzą przez hooki w kolejności."""

    def __init__(self, hooks: Sequence[ClassicalHook] = ()) -> None:
        self._hooks = tuple(hooks)
        self.transcript: list[Message] = []

    def send(self, message: Message) -> Message:
        for hook in self._hooks:
            message = hook(message)
        self.transcript.append(message)
        return message

    def leaked_bits(self, prefix: str = "") -> int:
        """Suma ujawnionych bitów w wiadomościach, których rodzaj zaczyna się od `prefix`."""
        return sum(m.leaked_bits for m in self.transcript if m.kind.startswith(prefix))

    def size_bits(self, prefix: str = "") -> int:
        return sum(m.size_bits for m in self.transcript if m.kind.startswith(prefix))

    def messages(self, prefix: str = "") -> list[Message]:
        return [m for m in self.transcript if m.kind.startswith(prefix)]

    def close_round(self) -> None:
        """Koniec rundy postprocessingu; kanał uwierzytelniony taguje tu transkrypt."""
