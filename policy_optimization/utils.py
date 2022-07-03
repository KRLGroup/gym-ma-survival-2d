from typing import TypeVar, Any, Union, Tuple, Callable, List, Dict, Optional, no_type_check

_T = TypeVar('_T')


def repeat(f, n, *args, **kwargs):
    return [f(*args, **kwargs) for _ in range(n)]


def callmethod(name):
    def call(obj, *args, **kwargs):
        return getattr(obj, name)(*args, **kwargs)
    return call


def subdict(d, without=[]):
    return {k: d[k] for k in d.keys() if k not in without}


# The "structure" of the arguments is taken from the 'struct_arg', which is the last by default.
def recursive_apply(f: Callable, *args: Any, default: Optional[Any] = None, struct_arg: int = -1) -> Any:
    if len(args) == 0:
        return default
    if isinstance(args[struct_arg], list):
        return [recursive_apply(f, *arg) for arg in zip(*args)]
    if isinstance(args[struct_arg], dict):
        return {k: recursive_apply(f, *[arg[k] for arg in args])
                for k in args[struct_arg].keys()}
    else:
        return f(*args)
