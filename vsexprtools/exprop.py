from __future__ import annotations

from itertools import cycle
from math import isqrt
from typing import Any, Iterable, Iterator, SupportsFloat

from vstools import ConvMode, CustomStrEnum, StrList, flatten

from .util import aka_expr_available

__all__ = [
    'ExprOp'
]


class ExprOp(CustomStrEnum):
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
    REL_PIX = '{char:s}[{x:d},{y:d}]'
    ABS_PIX = '{x:d} {y:d} {char:s}[]'

    def __call__(self, *pos_args: Any, **kwargs: Any) -> str:
        args = list[Any](pos_args)

        while True:
            try:
                return self.format(*args, **kwargs)
            except KeyError as key:
                try:
                    kwargs.update({str(key)[1:-1]: args.pop(0)})
                except IndexError:
                    raise key

    def __str__(self) -> str:
        return self.value

    def __next__(self) -> ExprOp:
        return self

    def __iter__(self) -> Iterator[ExprOp]:
        return cycle([self])

    def __mul__(self, n: int) -> list[ExprOp]:  # type: ignore[override]
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
                for y in range(-radius, radius + 1)
                for x in range(-radius, radius + 1)
            ]

        return StrList([
            var if x == y == 0 else
            ExprOp.REL_PIX(var, x, y)
            for (x, y) in coordinates
            if (x, y) not in exclude
        ])

    @classmethod
    def convolution(
        cls, var: str, matrix: Iterable[SupportsFloat] | Iterable[Iterable[SupportsFloat]],
        bias: float | None = None, divisor: float | bool = True, saturate: bool = True,
        mode: ConvMode = ConvMode.SQUARE
    ) -> StrList:
        convolution = list[float](flatten(matrix))  # type: ignore

        conv_len = len(convolution)

        if not conv_len % 2:
            raise ValueError('ExprOp.convolution: convolution length must be odd!')
        elif conv_len < 3:
            raise ValueError('ExprOp.convolution: you must pass at least 3 convolution items!')
        elif mode is ConvMode.SQUARE and conv_len != isqrt(conv_len) ** 2:
            raise ValueError(
                'ExprOp.convolution: with square mode, convolution must represent a square (radius*radius n items)!'
            )

        if mode != ConvMode.SQUARE:
            radius = conv_len // 2
        else:
            radius = isqrt(conv_len) // 2

        rel_pixels = cls.matrix(var, radius, mode)

        output = StrList([])

        expr_conv = [
            rel_pix if weight == 1 else [rel_pix, weight, ExprOp.MUL]
            for rel_pix, weight in zip(rel_pixels, convolution)
            if weight != 0
        ]

        output.extend(expr_conv)
        output.extend(ExprOp.ADD * (len(expr_conv) - 1))

        if divisor is not False:
            if divisor is True:
                divisor = sum(map(float, convolution))

            if divisor not in {0, 1}:
                output.extend([divisor, ExprOp.DIV])

        if bias is not None:
            output.extend([bias, ExprOp.ADD])

        if not saturate:
            output.append(ExprOp.ABS)

        return output
