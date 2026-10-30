"""Entropia binarna."""

import math


def h2(x: float) -> float:
    """h(x) = −x log₂ x − (1 − x) log₂(1 − x), z h(0) = h(1) = 0."""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"argument entropii binarnej musi być w [0, 1], jest {x}")
    if x in (0.0, 1.0):
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)
