import builtins
from functools import update_wrapper
from types import FunctionType
from typing import Any, Callable

from .operators import ExprOperators
from .variables import ExprVar


def copy_func(f: Callable[..., Any]) -> FunctionType:
    try:
        g = FunctionType(
            f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__
        )
        g = update_wrapper(g, f)
        g.__kwdefaults__ = f.__kwdefaults__
        return g
    except BaseException:  # for builtins
        return f  # type: ignore


global_builtins = _originals = {
    'min': copy_func(builtins.min),
    'max': copy_func(builtins.max)
}


def _expr_min(*args: Any, **kwargs: Any) -> Any:
    if not any(isinstance(x, ExprVar) for x in args):
        return _originals['min'](*args, **kwargs)

    var = args[0]
    for arg in args[1:]:
        var = ExprOperators.MIN(var, arg)

    return var


def _expr_max(*args: Any, **kwargs: Any) -> Any:
    if not any(isinstance(x, ExprVar) for x in args):
        return _originals['max'](*args, **kwargs)

    var = args[0]
    for arg in args[1:]:
        var = ExprOperators.MAX(var, arg)

    return var


global_builtins_expr = {
    'min': _expr_min,
    'max': _expr_max
}
