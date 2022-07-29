from __future__ import annotations

from math import ceil
from typing import Any, List, Sequence

import vapoursynth as vs

from .exprop import ExprOp
from .types import SingleOrArrOpt, StrArr, StrArrOpt, SupportsString
from .util import EXPR_VARS, aka_expr_available, flatten, norm_expr_planes, normalise_planes, to_arr

__all__ = [
    'expr_func', 'combine', 'expr'
]

core = vs.core


def expr_func(
    clips: vs.VideoNode | Sequence[vs.VideoNode], expr: str | Sequence[str],
    format: int | None = None, opt: bool = False, boundary: bool = False
) -> vs.VideoNode:
    args = (clips, expr, format, opt, boundary)
    return core.akarin.Expr(*args) if aka_expr_available else core.std.Expr(*args[:3])


def _combine_norm__ix(ffix: StrArrOpt, n_clips: int) -> List[SupportsString]:
    if ffix is None:
        return [''] * n_clips

    ffix = [ffix] if isinstance(ffix, tuple) else to_arr(ffix)  # type: ignore

    return ffix * max(1, ceil(n_clips / len(ffix)))  # type: ignore


def combine(
    clips: Sequence[vs.VideoNode], operator: ExprOp = ExprOp.MAX, suffix: StrArrOpt = None, prefix: StrArrOpt = None,
    expr_suffix: StrArrOpt = None, expr_prefix: StrArrOpt = None, planes: SingleOrArrOpt[int] = None,
    **expr_kwargs: Any
) -> vs.VideoNode:
    n_clips = len(clips)

    prefixes, suffixes = (_combine_norm__ix(x, n_clips) for x in (prefix, suffix))

    normalized_args = [to_arr(x)[:n_clips + 1] for x in (prefixes, EXPR_VARS, suffixes)]

    args = zip(*normalized_args)

    operators = operator * (n_clips - 1)

    return expr(clips, [expr_prefix, args, operators, expr_suffix], planes, **expr_kwargs)


def expr(
    clips: Sequence[vs.VideoNode], expr: StrArr, planes: SingleOrArrOpt[int], **expr_kwargs: Any
) -> vs.VideoNode:
    firstclip = clips[0]
    assert firstclip.format

    planes = normalise_planes(firstclip, planes)

    expr_array = list[SupportsString](flatten(expr))  # type: ignore

    expr_array_filtered = filter(lambda x: x is not None and x != '', expr_array)

    expr_string = ' '.join([str(x).strip() for x in expr_array_filtered])

    return expr_func(
        clips, norm_expr_planes(firstclip, expr_string, planes), **expr_kwargs
    )
