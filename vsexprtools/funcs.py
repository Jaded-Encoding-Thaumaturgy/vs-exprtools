from __future__ import annotations

from math import ceil
from typing import Any, Iterable, Literal, Sequence

from vstools import (
    FuncExceptT, PlanesT, StrArr, StrArrOpt, StrList, SupportsString, VideoFormatT, core,
    flatten, get_video_format, to_arr, vs
)

from .exprop import ExprOp
from .util import ExprVars, aka_expr_available, bitdepth_aware_tokenize_expr, norm_expr_planes

__all__ = [
    'expr_func', 'combine', 'norm_expr',

    'average_merge', 'weighted_merge'
]


def expr_func(
    clips: vs.VideoNode | Sequence[vs.VideoNode], expr: str | Sequence[str],
    format: VideoFormatT | None = None, opt: bool | None = None, boundary: bool = False,
    force_akarin: Literal[False] | FuncExceptT = False
) -> vs.VideoNode:
    func = force_akarin or expr_func
    over_clips = len(clips) > 26

    if not aka_expr_available:
        if force_akarin or over_clips:
            raise ExprVars._get_akarin_err('This function only works with akarin plugin!')(func=func)
    elif over_clips and b'src26' not in vs.core.akarin.Version()['expr_features']:  # type: ignore
        raise ExprVars._get_akarin_err('You need at least v0.96 of akarin plugin!')(func=func)

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

    ffix = [ffix] if isinstance(ffix, (str, tuple)) else list(ffix)  # type: ignore

    return ffix * max(1, ceil(n_clips / len(ffix)))  # type: ignore


def combine(
    clips: Sequence[vs.VideoNode], operator: ExprOp = ExprOp.MAX, suffix: StrArrOpt = None, prefix: StrArrOpt = None,
    expr_suffix: StrArrOpt = None, expr_prefix: StrArrOpt = None, planes: PlanesT = None, **expr_kwargs: Any
) -> vs.VideoNode:
    n_clips = len(clips)

    prefixes, suffixes = (_combine_norm__ix(x, n_clips) for x in (prefix, suffix))

    args = zip(prefixes, ExprVars(n_clips), suffixes)

    operators = operator * max(n_clips - 1, 1)

    return norm_expr(clips, [expr_prefix, args, operators, expr_suffix], planes, **expr_kwargs)


def norm_expr(
    clips: vs.VideoNode | Iterable[vs.VideoNode], expr: str | StrArr | tuple[str | StrArr, ...],
    planes: PlanesT = None, format: VideoFormatT | None = None, opt: bool | None = None,
    boundary: bool = False, force_akarin: Literal[False] | str = False, **kwargs: Any
) -> vs.VideoNode:
    if isinstance(clips, vs.VideoNode):
        clips = [clips]  # type: ignore
    else:
        clips = list(clips)  # type: ignore

    if isinstance(expr, str):
        nexpr = tuple([[expr]])
    elif not isinstance(expr, tuple):
        nexpr = tuple([to_arr(expr)])  # type: ignore
    else:
        nexpr = tuple([to_arr(x) for x in expr])  # type: ignore

    normalized_exprs = [StrList(plane_expr).to_str() for plane_expr in nexpr]

    normalized_expr = norm_expr_planes(clips[0], normalized_exprs, planes, **kwargs)

    tokenized_expr = [
        bitdepth_aware_tokenize_expr(clips, e, bool(is_chroma))
        for is_chroma, e in enumerate(normalized_expr)
    ]

    return expr_func(clips, tokenized_expr, format, opt, boundary, force_akarin)


def average_merge(*clips: Iterable[vs.VideoNode] | vs.VideoNode) -> vs.VideoNode:
    flat_clips = list[vs.VideoNode](flatten(clips))  # type: ignore

    length = len(flat_clips)

    return combine(flat_clips, ExprOp.ADD, zip(((1 / length, ) * length), ExprOp.MUL))


def weighted_merge(*weighted_clips: Iterable[tuple[vs.VideoNode, float]] | tuple[vs.VideoNode, float]) -> vs.VideoNode:
    flat_clips = []

    for clip in weighted_clips:
        if isinstance(clip, tuple):
            flat_clips.append(clip)
        else:
            flat_clips.extend(list(clip))

    clips, weights = zip(*flat_clips)

    return combine(clips, ExprOp.ADD, zip(weights, ExprOp.MUL), expr_suffix=[sum(weights), ExprOp.DIV])
