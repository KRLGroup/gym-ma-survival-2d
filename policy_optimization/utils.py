from typing import TypeVar, Any, Union, Tuple, Callable, List, Dict, Optional, no_type_check

_T = TypeVar('_T')


def repeat(f, n, *args, **kwargs):
    return [f(*args, **kwargs) for _ in range(n)]


def callmethod(name):
    def call(obj, *args, **kwargs):
        return getattr(obj, name)(*args, **kwargs)
    return call


def ensure(cond):
    assert cond


def subdict(d, without=[]):
    return {k: d[k] for k in d.keys() if k not in without}


# The "structure" of the arguments is taken from the 'struct_arg', which is the last by default.
def recursive_apply(f: Callable, *args: Any, default: Optional[Any] = None, struct_arg: int = -1, verbose: bool = False) -> Any:
    if len(args) == 0:
        return default
    if isinstance(args[struct_arg], list):
        if verbose:
            print('list')
        return [recursive_apply(f, *arg, verbose=verbose) for arg in zip(*args)]
    if isinstance(args[struct_arg], dict):
        if verbose:
            print(f'dict({args[struct_arg].keys()})')
        return {k: recursive_apply(f, *[arg[k] for arg in args], verbose=verbose)
                for k in args[struct_arg].keys()}
    else:
        if verbose:
            print('apply')
        return f(*args)
