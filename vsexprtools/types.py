from __future__ import annotations

from abc import abstractmethod
from typing import (
    Any, Callable, Iterable, List, Protocol, Sequence, SupportsFloat, SupportsIndex, TypeAlias, TypeVar, Union,
    overload, runtime_checkable
)

import vapoursynth as vs

__all__ = [
    'F', 'T', 'R', 'T_contra', 'Self',

    'SingleOrArr', 'SingleOrArrOpt',
    'StrArr', 'StrArrOpt',

    'PlanesT', 'FrameRange',

    'SupportsFloatOrIndex', 'ByteData',

    'SupportsAllComparisons',
    'SupportsTrunc', 'SupportsString',
    'SupportsDunderLT', 'SupportsDunderGT',
    'SupportsDunderLE', 'SupportsDunderGE',
    'SupportsRichComparison', 'SupportsRichComparisonT',

    'VSFunction', 'ComparatorFunc',

    'StrList'
]

T = TypeVar('T')
R = TypeVar('R')

F = TypeVar('F', bound=Callable[..., Any])
T_contra = TypeVar("T_contra", contravariant=True)

PlanesT = Union[int, Sequence[int], None]
FrameRange = Union[int, tuple[int, int], list[int]]

Self = TypeVar("Self")

SingleOrArr = Union[T, List[T]]
SingleOrArrOpt = Union[SingleOrArr[T], None]

SupportsFloatOrIndex: TypeAlias = SupportsFloat | SupportsIndex
ByteData: TypeAlias = SupportsFloat | SupportsIndex | str | bytes | bytearray | memoryview


@runtime_checkable
class SupportsTrunc(Protocol):
    def __trunc__(self) -> int:
        pass


@runtime_checkable
class SupportsString(Protocol):
    @abstractmethod
    def __str__(self) -> str:
        pass


@runtime_checkable
class SupportsDunderLT(Protocol[T_contra]):
    def __lt__(self, __other: T_contra) -> bool:
        pass


@runtime_checkable
class SupportsDunderGT(Protocol[T_contra]):
    def __gt__(self, __other: T_contra) -> bool:
        pass


@runtime_checkable
class SupportsDunderLE(Protocol[T_contra]):
    def __le__(self, __other: T_contra) -> bool:
        pass


@runtime_checkable
class SupportsDunderGE(Protocol[T_contra]):
    def __ge__(self, __other: T_contra) -> bool:
        pass


@runtime_checkable
class SupportsAllComparisons(
    SupportsDunderLT[Any], SupportsDunderGT[Any], SupportsDunderLE[Any], SupportsDunderGE[Any], Protocol
):
    pass


SupportsRichComparison: TypeAlias = SupportsDunderLT[Any] | SupportsDunderGT[Any]
SupportsRichComparisonT = TypeVar("SupportsRichComparisonT", bound=SupportsRichComparison)


class StrList(List[SupportsString]):
    @property
    def string(self) -> str:
        pass

    @string.getter
    def string(self) -> str:
        return self.to_str()

    def to_str(self) -> str:
        return str(self)

    def __str__(self) -> str:
        from .util import flatten

        return ' '.join(map(str, flatten(self)))


StrArr = SingleOrArr[SupportsString]
StrArrOpt = SingleOrArrOpt[SupportsString]


class VSFunction(Protocol):
    def __call__(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
        ...


_T = TypeVar('_T')
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')


class ComparatorFunc(Protocol):
    @overload
    def __call__(
        self, __arg1: SupportsRichComparisonT, __arg2: SupportsRichComparisonT,
        *_args: SupportsRichComparisonT, key: None = ...
    ) -> SupportsRichComparisonT:
        ...

    @overload
    def __call__(self, __arg1: _T, __arg2: _T, *_args: _T, key: Callable[[_T], SupportsRichComparison]) -> _T:
        ...

    @overload
    def __call__(self, __iterable: Iterable[SupportsRichComparisonT], *, key: None = ...) -> SupportsRichComparisonT:
        ...

    @overload
    def __call__(self, __iterable: Iterable[_T], *, key: Callable[[_T], SupportsRichComparison]) -> _T:
        ...

    @overload
    def __call__(
        self, __iterable: Iterable[SupportsRichComparisonT], *, key: None = ..., default: _T
    ) -> SupportsRichComparisonT | _T:
        ...

    @overload
    def __call__(
        self, __iterable: Iterable[_T1], *, key: Callable[[_T1], SupportsRichComparison], default: _T2
    ) -> _T1 | _T2:
        ...
