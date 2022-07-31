from __future__ import annotations

import string
from functools import partial, update_wrapper
from math import ceil, floor
from types import FunctionType
from typing import Any, ByteString, Callable, Deque, Iterable, List, Mapping, Sequence, Set, TypeVar, Union, overload

import vapoursynth as vs
from vsutil import depth, disallow_variable_format, get_depth

from .types import (
    ByteData, ComparatorFunc, FrameRange, PlanesT, StrList, SupportsAllComparisons, SupportsDunderGE, SupportsDunderGT,
    SupportsDunderLE, SupportsDunderLT, SupportsFloatOrIndex, SupportsRichComparison, SupportsString, SupportsTrunc,
    VSFunction
)

AnythingElse = TypeVar(
    'AnythingElse', bound=Union[
        type, int, str, None, SupportsFloatOrIndex, ByteData, SupportsAllComparisons,
        SupportsTrunc, SupportsString, SupportsRichComparison, VSFunction, ComparatorFunc, StrList, ByteString,
        SupportsDunderLT, SupportsDunderGT, SupportsDunderLE, SupportsDunderGE, Set, Mapping, Deque  # type: ignore
    ]
)


__all__ = [
    # VS variables
    'EXPR_VARS', 'aka_expr_available',
    # VS Functions
    'expect_bits',
    # Math stuff
    'clamp', 'cround', 'mod_x', 'mod2', 'mod4',
    # Array stuff
    'to_arr', 'normalise_seq', 'flatten',
    # VS helpers
    'normalise_planes', 'norm_expr_planes', 'normalize_franges',
    'shift_clip', 'shift_clip_multi', 'get_w', 'get_h',
    # Other utils
    'copy_func',
]

core = vs.core

EXPR_VARS = (alph := list(string.ascii_lowercase))[(idx := alph.index('x')):] + alph[:idx]

try:
    aka_expr_available = bool(core.akarin.Expr)
except AttributeError:
    aka_expr_available = False


Nb = TypeVar('Nb', float, int)


def clamp(val: Nb, min_val: Nb, max_val: Nb) -> Nb:
    return min_val if val < min_val else max_val if val > max_val else val


def cround(x: float) -> int:
    return floor(x + 0.5) if x > 0 else ceil(x - 0.5)


def mod_x(val: int | float, x: int) -> int:
    return max(x * x, cround(val / x) * x)


mod2 = partial(mod_x, x=2)

mod4 = partial(mod_x, x=4)


@overload
def normalise_seq(val: Sequence[AnythingElse], length_max: int = 3) -> List[AnythingElse]:
    pass


@overload
def normalise_seq(val: AnythingElse, length_max: int = 3) -> List[AnythingElse]:
    pass


def normalise_seq(val: Any, length_max: int = 3) -> Any:
    if not isinstance(val, Sequence):
        return [val] * length_max

    val = list(val) + [val[-1]] * (length_max - len(val))

    return val[:length_max]


def normalise_planes(clip: vs.VideoNode, planes: PlanesT = None, pad: bool = False) -> List[int]:
    assert clip.format

    if planes is None:
        planes = list(range(clip.format.num_planes))
    else:
        planes = to_arr(planes)

    if pad:
        return normalise_seq(planes, clip.format.num_planes)

    return list(set(sorted(planes)))


def norm_expr_planes(
    clip: vs.VideoNode, expr: str | List[str], planes: PlanesT = None, **kwargs: Any
) -> List[str]:
    assert clip.format

    expr_array = normalise_seq(to_arr(expr), clip.format.num_planes)

    planes = normalise_planes(clip, planes)

    string_args = [(key, normalise_seq(value)) for key, value in kwargs.items()]

    return [
        exp.format(**{key: value[i] for key, value in string_args})
        if i in planes else '' for i, exp in enumerate(expr_array, 0)
    ]


@overload
def to_arr(val: Sequence[AnythingElse]) -> List[AnythingElse]:
    pass


@overload
def to_arr(val: AnythingElse) -> List[AnythingElse]:
    pass


def to_arr(val: Any) -> Any:
    return val if type(val) in {list, tuple, range, zip, set, map, enumerate} else [val]


@overload
def flatten(items: Iterable[Iterable[AnythingElse]]) -> Iterable[AnythingElse]:
    ...


@overload
def flatten(items: Iterable[AnythingElse]) -> Iterable[AnythingElse]:
    ...


def flatten(items: Any) -> Any:
    for val in items:
        if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
            for sub_x in flatten(val):
                yield sub_x
        else:
            yield val


@disallow_variable_format
def expect_bits(clip: vs.VideoNode, expected_depth: int = 16) -> tuple[int, vs.VideoNode]:
    return (bits := get_depth(clip)), depth(clip, expected_depth) if bits != expected_depth else clip


def copy_func(f: Callable[..., Any]) -> FunctionType:
    try:
        g = FunctionType(
            f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__
        )
        g = update_wrapper(g, f)
        g.__kwdefaults__ = f.__kwdefaults__
        return g
    except BaseException:  # for builtins
        return f  # type: ignore


def normalize_franges(franges: FrameRange, /) -> Iterable[int]:
    if isinstance(franges, int):
        return [franges]
    elif isinstance(franges, tuple):
        start, stop = franges
        step = -1 if stop < start else 1
        return range(start, stop + step, step)
    else:
        return franges


def shift_clip(clip: vs.VideoNode, offset: int) -> vs.VideoNode:
    if offset > clip.num_frames - 1:
        raise ValueError("shift_clip: Offset can't be greater than the clip length!")

    if offset < 0:
        return clip[0] * abs(offset) + clip[:offset]
    elif offset > 0:
        return clip[offset:] + clip[-1] * offset

    return clip


def shift_clip_multi(clip: vs.VideoNode, shift: FrameRange = (-1, 1)) -> List[vs.VideoNode]:
    ranges = normalize_franges(shift)

    return [shift_clip(clip, x) for x in ranges]


@overload
def get_w(height: int, ar: float = 16 / 9, mod: int = 2, /) -> int:
    ...


@overload
def get_w(height: int, ref: vs.VideoNode, /) -> int:
    ...


def get_w(height: int, ar_or_ref: vs.VideoNode | float = 16 / 9, mod: int | None = None, /) -> int:
    if isinstance(ar_or_ref, vs.VideoNode):
        assert (ref := ar_or_ref).format
        aspect_ratio = ref.width / ref.height
        mod = 1 << ref.format.subsampling_w
    else:
        aspect_ratio = ar_or_ref
        if mod is None:
            mod = 0 if height % 2 else 2

    width = height * aspect_ratio

    if mod:
        return mod_x(width, mod)

    return round(width)


@overload
def get_h(width: int, ar: float = 16 / 9, mod: int = 2, /) -> int:
    ...


@overload
def get_h(width: int, ref: vs.VideoNode, /) -> int:
    ...


def get_h(width: int, ar_or_ref: vs.VideoNode | float = 16 / 9, mod: int | None = None, /) -> int:
    if isinstance(ar_or_ref, vs.VideoNode):
        assert (ref := ar_or_ref).format
        aspect_ratio = ref.height / ref.width
        mod = 1 << ref.format.subsampling_h
    else:
        aspect_ratio = ar_or_ref
        if mod is None:
            mod = 0 if width % 2 else 2

    height = width * aspect_ratio

    if mod:
        return mod_x(height, mod)

    return round(height)
