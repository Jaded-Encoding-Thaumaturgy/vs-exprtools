from __future__ import annotations

from enum import Enum
from itertools import cycle
from math import isqrt
from typing import Any, Iterable, Iterator, List, SupportsFloat

from .types import ConvMode, StrList
from .util import aka_expr_available, flatten

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

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self.format(args, **kwargs)

    def __str__(self) -> str:
        return self.value

    def __next__(self) -> ExprOp:
        return self

    def __iter__(self) -> Iterator[ExprOp]:
        return cycle([self])

    def __mul__(self, n: int) -> List[ExprOp]:  # type: ignore[override]
        return [self] * n

    @classmethod
    def clamp(cls, min: float, max: float, c: str = '') -> StrList:

        if aka_expr_available:
            return StrList([c, min, max, ExprOp.CLAMP])

        return StrList([c, min, ExprOp.MAX, max, ExprOp.MAX])

    @classmethod
    def matrix(
        cls, var: str, radius: int, mode: ConvMode = ConvMode.SQUARE, exclude: Iterable[tuple[int, int]] = []
    ) -> StrList:
        exclude = list(exclude)

        if mode != ConvMode.SQUARE:
            coordinates = [
                (xy, 0) if mode is ConvMode.HORIZONTAL else (0, xy)
                for xy in range(-radius, radius + 1)
            ]
        else:
            coordinates = [
                (x, y)
                for x in range(-radius, radius + 1)
                for y in range(-radius, radius + 1)
            ]

        return StrList([
            ExprOp.REL_PIX(var, x, y) for (x, y) in coordinates if (x, y) not in exclude
        ])

