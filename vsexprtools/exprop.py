from __future__ import annotations

from enum import Enum
from itertools import cycle
from typing import Iterator, List

__all__ = [
    'ExprOp'
]


class ExprOp(str, Enum):
    # 1 Argument
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    SIN = "sin"
    COS = "cos"
    ABS = "abs"
    NOT = "not"
    DUP = "dup"
    DUPN = "dupN"
    TRUNC = "trunc"
    ROUND = "round"
    FLOOR = "floor"

    # 2 Arguments
    MAX = "max"
    MIN = "min"
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "pow"
    GT = ">"
    LT = "<"
    EQ = "="
    GTE = ">="
    LTE = "<="
    AND = "and"
    OR = "or"
    XOR = "xor"
    SWAP = "swap"
    SWAPN = "swapN"
    MOD = "%"

    # 3 Arguments
    TERN = "?"
    CLAMP = "clamp"

    # Special Operators
    REL_PIX = '{char:s}[{x:s},{y:s}]'
    ABS_PIX = '{x:s} {y:s} {char:s}[]'

    def __str__(self) -> str:
        return self.value

    def __next__(self) -> ExprOp:
        return self

    def __iter__(self) -> Iterator[ExprOp]:
        return cycle([self])

    def __mul__(self, n: int) -> List[ExprOp]:  # type: ignore[override]
        return [self] * n
