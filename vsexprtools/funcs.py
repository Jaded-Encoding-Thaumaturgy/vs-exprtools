from __future__ import annotations

from math import ceil
from typing import Any, Iterable, Literal, Sequence

from vstools import (
    EXPR_VARS, PlanesT, StrArr, StrArrOpt, StrList, SupportsString, VideoFormatT, core, get_video_format, to_arr, vs
)

from .exprop import ExprOp
from .util import aka_expr_available, norm_expr_planes

__all__ = [
    'expr_func', 'combine', 'norm_expr'
]


def expr_func(
    clips: vs.VideoNode | Sequence[vs.VideoNode], expr: str | Sequence[str],
    format: VideoFormatT | None = None, opt: bool | None = None, boundary: bool = False,
    force_akarin: Literal[False] | str = False
) -> vs.VideoNode:
    if not aka_expr_available and force_akarin:
        raise RuntimeError(
            f'{force_akarin}: This function only works with akarin-plugin!\n'
            'Download it from https://github.com/AkarinVS/vapoursynth-plugin'
        )

    fmt = None if format is None else get_video_format(format).id

    if aka_expr_available and opt is None:
        opt = all([
            clip.format and clip.format.sample_type == vs.INTEGER
            for clip in (clips if isinstance(clips, Sequence) else [clips])
        ])

    try:
        if aka_expr_available:
            return core.akarin.Expr(clips, expr, fmt, opt, boundary)

        return core.std.Expr(clips, expr, fmt)
    except BaseException as e:
        raise RuntimeError(
            'There was an error when evaluating the expression:\n' + (
                '' if aka_expr_available else 'You might need akarin-plugin, and are missing it.'
            )

        ) from e


def _combine_norm__ix(ffix: StrArrOpt, n_clips: int) -> list[SupportsString]:
    if ffix is None:
        return [''] * n_clips

    ffix = [ffix] if (type(ffix) in {str, tuple}) else list(ffix)  # type: ignore

    return ffix * max(1, ceil(n_clips / len(ffix)))


def combine(
    clips: Sequence[vs.VideoNode], operator: ExprOp = ExprOp.MAX, suffix: StrArrOpt = None, prefix: StrArrOpt = None,
    expr_suffix: StrArrOpt = None, expr_prefix: StrArrOpt = None, planes: PlanesT = None, **expr_kwargs: Any
) -> vs.VideoNode:
    n_clips = len(clips)

    prefixes, suffixes = (_combine_norm__ix(x, n_clips) for x in (prefix, suffix))

    normalized_args = [to_arr(x)[:n_clips + 1] for x in (prefixes, EXPR_VARS, suffixes)]

    args = zip(*normalized_args)

    operators = operator * (n_clips - 1)

    return norm_expr(clips, [expr_prefix, args, operators, expr_suffix], planes, **expr_kwargs)


def norm_expr(
    clips: vs.VideoNode | Iterable[vs.VideoNode], expr: str | StrArr | tuple[str | StrArr, ...],
    planes: PlanesT = None, format: VideoFormatT | None = None, opt: bool | None = None,
    boundary: bool = False, force_akarin: Literal[False] | str = False, **kwargs: Any
) -> vs.VideoNode:
    if isinstance(clips, vs.VideoNode):
        clips = [clips]
    else:
        clips = list(clips)

    if isinstance(expr, str):
        nexpr = tuple([[expr]])
    elif not isinstance(expr, tuple):
        nexpr = tuple([to_arr(expr)])  # type: ignore
    else:
        nexpr = tuple([to_arr(x) for x in expr])  # type: ignore

    normalized_exprs = [StrList(plane_expr).to_str() for plane_expr in nexpr]

    normalized_expr = norm_expr_planes(clips[0], normalized_exprs, planes, **kwargs)

    return expr_func(clips, normalized_expr, format, opt, boundary, force_akarin)
