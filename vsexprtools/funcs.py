from __future__ import annotations

from functools import partial, wraps
from math import ceil
from typing import Any, Callable, Iterable, Literal, Sequence, cast, overload

from vstools import (
    F_VD, CustomRuntimeError, CustomValueError, FuncExceptT, HoldsVideoFormatT, P, PlanesT,
    ProcessVariableResClip, StrArr, StrArrOpt, StrList, SupportsString, VideoFormatT,
    VideoNodeIterable, check_variable_format, core, flatten_vnodes, get_video_format, normalize_seq,
    to_arr, vs
)

from .exprop import ExprList, ExprOp, ExprOpBase, ExprToken, TupleExprList
from .util import ExprVars, bitdepth_aware_tokenize_expr, complexpr_available, norm_expr_planes

__all__ = [
    'expr_func', 'combine',

    'limiter_expr', 'limiter_output',

    'norm_expr',

    'average_merge', 'weighted_merge'
]


def expr_func(
    clips: vs.VideoNode | Sequence[vs.VideoNode], expr: str | Sequence[str],
    format: HoldsVideoFormatT | VideoFormatT | None = None, opt: bool | None = None, boundary: bool = True,
    force_akarin: Literal[False] | FuncExceptT = False, func: FuncExceptT | None = None
) -> vs.VideoNode:
    func = func or force_akarin or expr_func
    clips = list(clips) if isinstance(clips, Sequence) else [clips]
    over_clips = len(clips) > 26

    if not complexpr_available:
        if force_akarin or over_clips:
            raise ExprVars._get_akarin_err('This function only works with akarin plugin!')(func=func)
    elif over_clips and b'src26' not in vs.core.akarin.Version()['expr_features']:  # type: ignore
        raise ExprVars._get_akarin_err('You need at least v0.96 of akarin plugin!')(func=func)

    fmt = None if format is None else get_video_format(format).id

    got_var_res = False

    for clip in clips:
        check_variable_format(clip, func)
        got_var_res = got_var_res or (0 in (clip.width, clip.height))

    if complexpr_available and opt is None:
        opt = all(clip.format.sample_type == vs.INTEGER for clip in clips)  # type: ignore

    if complexpr_available:
        func_impl = partial(core.akarin.Expr, expr=expr, format=fmt, opt=opt, boundary=boundary)
    else:
        func_impl = partial(core.std.Expr, expr=expr, format=fmt)

    if got_var_res:
        if len(clips) == 1:
            return ProcessVariableResClip.from_func(clips[0], func_impl, None, clips[0].format)

        raise CustomValueError('You can run only one var res clip!')

    try:
        return func_impl(clips)
    except Exception:
        raise CustomRuntimeError(
            'There was an error when evaluating the expression:\n' + (
                '' if complexpr_available else 'You might need akarin-plugin, and are missing it.'
            ), func, f'\n{expr}\n'
        )


def _combine_norm__ix(ffix: StrArrOpt, n_clips: int) -> list[SupportsString]:
    if ffix is None:
        return [''] * n_clips

    ffix = [ffix] if isinstance(ffix, (str, tuple)) else list(ffix)  # type: ignore

    return ffix * max(1, ceil(n_clips / len(ffix)))  # type: ignore


def combine(
    clips: VideoNodeIterable, operator: ExprOpBase = ExprOp.MAX, suffix: StrArrOpt = None,
    prefix: StrArrOpt = None, expr_suffix: StrArrOpt = None, expr_prefix: StrArrOpt = None,
    planes: PlanesT = None, split_planes: bool = False, **kwargs: Any
) -> vs.VideoNode:
    clips = flatten_vnodes(clips, split_planes=split_planes)

    n_clips = len(clips)

    prefixes, suffixes = (_combine_norm__ix(x, n_clips) for x in (prefix, suffix))

    args = zip(prefixes, ExprVars(n_clips), suffixes)

    has_op = (n_clips >= operator.n_op) or any(x is not None for x in (suffix, prefix, expr_suffix, expr_prefix))

    operators = operator * max(n_clips - 1, int(has_op))

    return norm_expr(clips, [expr_prefix, args, operators, expr_suffix], planes, **kwargs)


def limiter_expr(
    clip: vs.VideoNode,
    min_val: float | ExprToken | Sequence[float | ExprToken] = ExprToken.RangeMin,
    max_val: float | ExprToken | Sequence[float | ExprToken] = ExprToken.RangeMax,
    planes: PlanesT = None,
    func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Limits the pixel values to the range (min_val, max_val)

    :param clip:        Clip to process.
    :param min_val:     Lower bound. Defaults to the lowest allowed value for the input.
                        Can be specified for each plane individually.
    :param max_val:     Upper bound. Defaults to the highest allowed value for the input.
                        Can be specified for each plane individually.
    :param planes:      Specifies which planes will be processed.
    """
    func = func or limiter_expr

    assert check_variable_format(clip, func)

    min_val = normalize_seq(min_val)
    max_val = normalize_seq(max_val)

    if all([
        clip.format.sample_type == vs.INTEGER,
        (mi == ExprToken.RangeMin for mi in min_val),
        (ma == ExprToken.RangeMax for ma in max_val),
    ]):
        return clip

    expr = list[ExprList]()

    for i, (mi, ma) in enumerate(zip(min_val, max_val)):
        if mi is ExprToken.RangeMin and i >= 1:
            mi = mi.get_value(clip, chroma=True)

        if ma is ExprToken.RangeMax and i >= 1:
            ma = ma.get_value(clip, chroma=True)

        expr.append(ExprOp.clamp(mi, ma, "x"))

    return norm_expr(clip, tuple(expr), planes, func=func)


@overload
def limiter_output(func: F_VD, /) -> F_VD:
    ...


@overload
def limiter_output(
    *,
    min_val: float | ExprToken | Sequence[float | ExprToken] = ExprToken.RangeMin,
    max_val: float | ExprToken | Sequence[float | ExprToken] = ExprToken.RangeMax,
    planes: PlanesT = None,
) -> Callable[[F_VD], F_VD]:
    ...


def limiter_output(
    func: F_VD | None = None,
    /,
    *,
    min_val: float | ExprToken | Sequence[float | ExprToken] = ExprToken.RangeMin,
    max_val: float | ExprToken | Sequence[float | ExprToken] = ExprToken.RangeMax,
    planes: PlanesT = None,
) -> F_VD | Callable[[F_VD], F_VD]:
    """Decorator implementation of `limiter_expr`."""
    if func is None:
        return cast(Callable[[F_VD], F_VD], partial(limiter_output, min_val=min_val, max_val=max_val, planes=planes))

    @wraps(func)
    def _wrapper_limiter(*args: P.args, **kwargs: P.kwargs) -> vs.VideoNode:
        return limiter_expr(func(*args, **kwargs), min_val, max_val, planes, func)

    return cast(F_VD, _wrapper_limiter)


def norm_expr(
    clips: VideoNodeIterable,
    expr: str | StrArr | ExprList | tuple[str | StrArr | ExprList, ...] | TupleExprList,
    planes: PlanesT = None, format: HoldsVideoFormatT | VideoFormatT | None = None,
    opt: bool | None = None, boundary: bool = True,
    force_akarin: Literal[False] | FuncExceptT = False, func: FuncExceptT | None = None,
    split_planes: bool = False,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Evaluates an expression per pixel.

    :param clips:           Input clip(s).
    :param expr:            Expression to be evaluated.
                            A single str will be processed for all planes.
                            A list will be concatenated to form a single expr for all planes.
                            A tuple of these types will allow specification of different expr for each planes.
                            A TupleExprList will make a norm_expr call for each expression within this tuple.
    :param planes:          Plane to process, defaults to all.
    :param format:          Output format, defaults to the first clip format.
    :param opt:             Forces integer evaluation as much as possible.
    :param boundary:        Specifies the default boundary condition for relative pixel accesses:
                            - 0 means clamped
                            - 1 means mirrored
    :param split_planes:    Splits the VideoNodes into their individual planes.
    :return:                Evaluated clip.
    """
    clips = flatten_vnodes(clips, split_planes=split_planes)

    if isinstance(expr, str):
        nexpr = tuple([[expr]])
    elif isinstance(expr, tuple):
        if isinstance(expr, TupleExprList):
            if len(expr) < 1:
                raise CustomRuntimeError(
                    "When passing a TupleExprList you need at least one expr in it!", func, expr
                )

            nclips: list[vs.VideoNode] | vs.VideoNode = clips

            for e in expr:
                nclips = norm_expr(
                    nclips, e, planes, format, opt, boundary, force_akarin, func, split_planes, **kwargs
                )

            return cast(vs.VideoNode, nclips)
        else:
            nexpr = tuple([to_arr(x) for x in expr])  # type: ignore[arg-type]
    else:
        nexpr = tuple([to_arr(expr)])  # type: ignore[arg-type]

    normalized_exprs = [StrList(plane_expr).to_str() for plane_expr in nexpr]

    normalized_expr = norm_expr_planes(clips[0], normalized_exprs, planes, **kwargs)

    tokenized_expr = [
        bitdepth_aware_tokenize_expr(clips, e, bool(is_chroma))
        for is_chroma, e in enumerate(normalized_expr)
    ]

    return expr_func(clips, tokenized_expr, format, opt, boundary, force_akarin, func)


def average_merge(*clips: VideoNodeIterable) -> vs.VideoNode:
    flat_clips = flatten_vnodes(clips)

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
