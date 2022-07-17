from __future__ import annotations

from typing import Any, Callable, Iterable, List, Sequence, TypeVar, Union
from vsutil import EXPR_VARS

import vapoursynth as vs

core = vs.core

FINT = TypeVar('FINT', bound=Callable[..., vs.VideoNode])
FFLOAT = TypeVar('FFLOAT', bound=Callable[..., vs.VideoNode])
PlanesT = Union[int, Sequence[int], None]
VSFunc = Callable[[vs.VideoNode], vs.VideoNode]

wmean_matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]
mean_matrix = [1, 1, 1, 1, 1, 1, 1, 1, 1]


try:
    aka_expr_available = bool(core.akarin.Expr)
except AttributeError:
    aka_expr_available = False


T = TypeVar('T', bound=Union[int, float, str])


def normalise_seq(x: T | Sequence[T], length_max: int = 3) -> List[T]:
    if not isinstance(x, Sequence):
        return [x] * length_max

    x = list(x) + [x[-1]] * (length_max - len(x))

    return x[:length_max]


def normalise_planes(clip: vs.VideoNode, planes: PlanesT = None, pad: bool = False) -> List[int]:
    assert clip.format

    if planes is None:
        planes = list(range(clip.format.num_planes))
    else:
        planes = to_arr(planes)

    if pad:
        return normalise_seq(planes, clip.format.num_planes)

    return list(set(planes))


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


def to_arr(array: Sequence[T] | T) -> List[T]:
    return list(
        array if (type(array) in {list, tuple, range, zip, set, map, enumerate}) else [array]  # type: ignore
    )


def flatten(items: Iterable[T]) -> Iterable[T]:
    for val in items:
        if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
            for sub_x in flatten(val):
                yield sub_x
        else:
            yield val  # type: ignore
