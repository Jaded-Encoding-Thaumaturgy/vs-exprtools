from __future__ import annotations

from typing import Any, Sequence

from vstools import (
    EXPR_VARS, ColorRange, PlanesT, core, get_lowest_value, get_neutral_value, get_peak_value, normalize_planes,
    normalize_seq, to_arr, vs, CustomIndexError, FuncExceptT
)

__all__ = [
    # VS variables
    'EXPR_VARS', 'aka_expr_available',
    # Expr helpers
    'bitdepth_aware_tokenize_expr',
    # VS helpers
    'norm_expr_planes'
]


try:
    aka_expr_available = bool(core.akarin.Expr)
except AttributeError:
    aka_expr_available = False


def bitdepth_aware_tokenize_expr(
    clips: Sequence[vs.VideoNode], expr: str, chroma: bool, func: FuncExceptT | None = None
) -> str:
    func = func or bitdepth_aware_tokenize_expr

    if not expr or len(expr) < 4:
        return expr

    replaces = [
        ('ymin', lambda clip, _: get_lowest_value(clip, False, ColorRange.LIMITED)),
        ('cmin', lambda clip, _: get_lowest_value(clip, True, ColorRange.LIMITED)),
        ('ymax', lambda clip, _: get_peak_value(clip, False, ColorRange.LIMITED)),
        ('cmax', lambda clip, _: get_peak_value(clip, True, ColorRange.LIMITED)),
        ('range_half', lambda clip, _: get_neutral_value(clip, chroma)),
        ('range_size', lambda clip, _: (val := get_peak_value(clip)) + (1 - (val <= 1.0))),
        ('range_min', lambda clip, _: get_lowest_value(clip, chroma)),
        ('yrange_min', lambda clip, _: get_lowest_value(clip, False)),
        ('crange_min', lambda clip, _: get_lowest_value(clip, True)),
        ('range_max', lambda clip, _: get_peak_value(clip, chroma)),
        ('yrange_max', lambda clip, _: get_peak_value(clip, False)),
        ('crange_max', lambda clip, _: get_peak_value(clip, True)),
        ('range_in_min', lambda clip, crange: get_lowest_value(clip, chroma, crange)),
        ('yrange_in_min', lambda clip, crange: get_lowest_value(clip, False, crange)),
        ('crange_in_min', lambda clip, crange: get_lowest_value(clip, True, crange)),
        ('range_in_max', lambda clip, crange: get_peak_value(clip, chroma, crange)),
        ('yrange_in_max', lambda clip, crange: get_peak_value(clip, False, crange)),
        ('crange_in_max', lambda clip, crange: get_peak_value(clip, True, crange)),
    ]

    clips = list(clips)
    ranges = [ColorRange.from_video(c, func=func) for c in clips]

    mapped_clips = list(reversed(list(zip(['', *EXPR_VARS], clips[:1] + clips, ranges[:1] + ranges))))

    for mkey, function in replaces:
        if mkey in expr:
            for key, clip, crange in [
                (f'{mkey}_{k} ' if k else f'{mkey} ', clip, crange)
                for k, clip, crange in mapped_clips
            ]:
                expr = expr.replace(key, str(function(clip, crange) * 1.0) + ' ')

        if mkey in expr:
            raise CustomIndexError('Parsing error or not enough clips passed!', func, reason=expr)

    print(expr)

    return expr


def norm_expr_planes(
    clip: vs.VideoNode, expr: str | list[str], planes: PlanesT = None, **kwargs: Any
) -> list[str]:
    assert clip.format

    expr_array = normalize_seq(to_arr(expr), clip.format.num_planes)  # type: ignore

    planes = normalize_planes(clip, planes)

    string_args = [(key, normalize_seq(value)) for key, value in kwargs.items()]

    return [
        exp.format(**{key: value[i] for key, value in string_args})
        if i in planes else '' for i, exp in enumerate(expr_array, 0)
    ]
