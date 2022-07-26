from __future__ import annotations

from abc import abstractmethod
from typing import (
    Any, Callable, List, Protocol, SupportsFloat, SupportsIndex, TypeAlias, TypeVar, Union, runtime_checkable
)

import vapoursynth as vs

F = TypeVar('F', bound=Callable[..., Any])

T = TypeVar('T')
R = TypeVar('R')

T_contra = TypeVar("T_contra", contravariant=True)

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

        return ' '.join(map(str, flatten(self)))  # type: ignore


StrArr = SingleOrArr[SupportsString]
StrArrOpt = SingleOrArrOpt[SupportsString]


class VSFunction(Protocol):
    def __call__(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
        ...
