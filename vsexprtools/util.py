from __future__ import annotations

from typing import Any


from vstools import PlanesT, core, normalize_planes, normalize_seq, to_arr, EXPR_VARS, vs

__all__ = [
    # VS variables
    'EXPR_VARS', 'aka_expr_available',
    # VS helpers
    'norm_expr_planes'
]


try:
    aka_expr_available = bool(core.akarin.Expr)
except AttributeError:
    aka_expr_available = False


def norm_expr_planes(
    clip: vs.VideoNode, expr: str | list[str], planes: PlanesT = None, **kwargs: Any
) -> list[str]:
    assert clip.format

    expr_array = normalize_seq(to_arr(expr), clip.format.num_planes)

    planes = normalize_planes(clip, planes)

    string_args = [(key, normalize_seq(value)) for key, value in kwargs.items()]

    return [
        exp.format(**{key: value[i] for key, value in string_args})
        if i in planes else '' for i, exp in enumerate(expr_array, 0)
    ]
