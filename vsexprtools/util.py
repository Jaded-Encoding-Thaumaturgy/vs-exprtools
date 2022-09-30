from __future__ import annotations

import string
from typing import Any, List

import vapoursynth as vs
from vstools import PlanesT, core, normalize_planes, normalize_seq, to_arr

__all__ = [
    # VS variables
    'EXPR_VARS', 'aka_expr_available',
    # VS helpers
    'norm_expr_planes'
]

EXPR_VARS = (alph := list(string.ascii_lowercase))[(idx := alph.index('x')):] + alph[:idx]

try:
    aka_expr_available = bool(core.akarin.Expr)
except AttributeError:
    aka_expr_available = False


def norm_expr_planes(
    clip: vs.VideoNode, expr: str | List[str], planes: PlanesT = None, **kwargs: Any
) -> List[str]:
    assert clip.format

    expr_array = normalize_seq(to_arr(expr), clip.format.num_planes)

    planes = normalize_planes(clip, planes)

    string_args = [(key, normalize_seq(value)) for key, value in kwargs.items()]

    return [
        exp.format(**{key: value[i] for key, value in string_args})
        if i in planes else '' for i, exp in enumerate(expr_array, 0)
    ]
