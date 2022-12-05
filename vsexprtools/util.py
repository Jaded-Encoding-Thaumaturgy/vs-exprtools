from __future__ import annotations

from itertools import count
from typing import Any, Iterable, Iterator, Sequence, SupportsIndex, overload

from vstools import (
    EXPR_VARS, MISSING, ColorRange, CustomIndexError, CustomNotImplementedError, CustomRuntimeError, FuncExceptT,
    MissingT, PlanesT, classproperty, core, get_lowest_value, get_neutral_value, get_peak_value, normalize_planes,
    normalize_seq, to_arr, vs
)

__all__ = [
    # VS variables
    'EXPR_VARS', 'aka_expr_available',
    # Expr helpers
    'ExprVars', 'bitdepth_aware_tokenize_expr',
    # VS helpers
    'norm_expr_planes'
]


try:
    aka_expr_available = bool(core.akarin.Expr)
except AttributeError:
    aka_expr_available = False


class _ExprVars(Iterable[str]):
    @overload
    def __init__(self, stop: SupportsIndex, /, *, akarin: bool | None = None) -> None:
        ...

    @overload
    def __init__(
        self, start: SupportsIndex, stop: SupportsIndex, step: SupportsIndex = 1, /, *, akarin: bool | None = None
    ) -> None:
        ...

    def __init__(
        self, start_stop: SupportsIndex, stop: SupportsIndex | MissingT = MISSING, step: SupportsIndex = 1,
        /, *, akarin: bool | None = None
    ) -> None:
        if stop is MISSING:
            self.start = 0
            self.stop = start_stop.__index__()
        else:
            self.start = start_stop.__index__()
            self.stop = stop.__index__()

        self.step = step.__index__()

        if self.start < 0:
            raise CustomIndexError('"start" must be bigger or equal than 0!')
        elif self.stop <= self.start:
            raise CustomIndexError('"stop" must be bigger than "start"!')

        self.akarin = self._check_akarin(self.stop, akarin)

        self.curr = self.start

    @overload
    def __call__(self, stop: SupportsIndex, /, *, akarin: bool | None = None) -> _ExprVars:
        ...

    @overload
    def __call__(
        self, start: SupportsIndex, stop: SupportsIndex, step: SupportsIndex = 1, /, *, akarin: bool | None = None
    ) -> _ExprVars:
        ...

    def __call__(
        self, start_stop: SupportsIndex, stop: SupportsIndex | MissingT = MISSING, step: SupportsIndex = 1,
        /, *, akarin: bool | None = None
    ) -> _ExprVars:
        return ExprVars(start_stop, stop, step, akarin=akarin)  # type: ignore

    def __iter__(self) -> Iterator[str]:
        indices = range(self.start, self.stop, self.step)

        if self.akarin:
            return (f'src{x}' for x in indices)

        return (EXPR_VARS[x] for x in indices)

    def __next__(self) -> str:
        if self.curr >= self.stop:
            raise StopIteration

        var = f'src{self.curr}' if self.akarin else EXPR_VARS[self.curr]

        self.curr += self.step

        return var

    @classmethod
    def _check_akarin(cls, stop: SupportsIndex, akarin: bool | None = None) -> bool:
        stop = stop.__index__()

        if akarin is None:
            akarin = stop > 26

        if akarin and not aka_expr_available:
            raise cls._get_akarin_err(
                'You are trying to get more than 26 variables or srcX vars, you need akarin plugin!'
            )

        return akarin

    @classmethod
    def get_var(cls, value: SupportsIndex, akarin: bool | None = None) -> str:
        value = value.__index__()

        if value < 0:
            raise CustomIndexError('"value" should be bigger than 0!')

        akarin = cls._check_akarin(value + 1, akarin)

        return f'src{value}' if akarin else EXPR_VARS[value]

    @classmethod
    def _get_akarin_err(cls, message: str) -> CustomRuntimeError:
        return CustomRuntimeError(f'{message}\nDownload it from https://github.com/AkarinVS/vapoursynth-plugin')

    @overload
    def __class_getitem__(cls, index: SupportsIndex | tuple[SupportsIndex, bool], /) -> str:
        ...

    @overload
    def __class_getitem__(cls, slice: slice | tuple[slice, bool], /) -> list[str]:
        ...

    def __class_getitem__(
        cls, idx_slice: SupportsIndex | slice | tuple[SupportsIndex | slice, bool], /,
    ) -> str | list[str]:
        if isinstance(idx_slice, tuple):
            idx_slice, akarin = idx_slice
        else:
            akarin = None

        if isinstance(idx_slice, slice):
            return list(ExprVars(idx_slice.start, idx_slice.stop, idx_slice.step))
        elif isinstance(idx_slice, SupportsIndex):
            return ExprVars.get_var(idx_slice.__index__(), akarin)

        raise CustomNotImplementedError

    @overload
    def __getitem__(self, index: SupportsIndex | tuple[SupportsIndex, bool], /) -> str:
        ...

    @overload
    def __getitem__(self, slice: slice | tuple[slice, bool], /) -> list[str]:
        ...

    def __getitem__(  # type: ignore
        self, idx_slice: SupportsIndex | slice | tuple[SupportsIndex | slice, bool], /,
    ) -> str | list[str]:
        ...

    def __str__(self) -> str:
        return ' '.join(iter(self))

    @classproperty
    def cycle(cls) -> Iterator[str]:
        for x in count():
            yield cls.get_var(x)


ExprVars: _ExprVars = _ExprVars  # type: ignore


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
