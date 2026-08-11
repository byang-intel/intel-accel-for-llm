# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ctypes
import contextlib
from contextlib import AbstractContextManager
from typing import Optional, Callable, Union
from functools import wraps

from ..envs import envs


def _resolve_strategy():
    profile_mode = envs.IAXL_PROFILE_MODE

    if profile_mode == "full":
        from torch.autograd.profiler import record_function

        return record_function, "full"

    elif profile_mode == "nvtx":
        import nvtx

        return nvtx.annotate, "nvtx"

    else:
        return contextlib.nullcontext, "disabled"


_REAL_PROFILER_FUNC, _STRATEGY = _resolve_strategy()


_PROFILING_ACTIVE = False


_PROFILER_FUNC = contextlib.nullcontext


def is_profiling_enabled() -> bool:
    return _STRATEGY != "disabled"


def is_profiling_active() -> bool:
    return _STRATEGY != "disabled" and _PROFILING_ACTIVE


def resume_profiling():
    global _PROFILING_ACTIVE, _PROFILER_FUNC
    if _STRATEGY == "disabled":
        return
    _PROFILING_ACTIVE = True
    _PROFILER_FUNC = _REAL_PROFILER_FUNC


def pause_profiling():
    global _PROFILING_ACTIVE, _PROFILER_FUNC
    _PROFILING_ACTIVE = False
    _PROFILER_FUNC = contextlib.nullcontext


start_profiling = resume_profiling
stop_profiling = pause_profiling


def set_thread_name(name: str):
    name = name[:15]

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_NAME = 15
        libc.prctl(PR_SET_NAME, name.encode("utf-8"), 0, 0, 0)
    except Exception:
        pass

    if _STRATEGY == "nvtx":
        try:
            import nvtx

            nvtx.mark(f"Thread: {name}")
        except Exception:
            pass


class _NullCrossScopeProfile:
    __slots__ = ()

    @property
    def active(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return ""

    def start(self):
        pass

    def end(self):
        pass

    def __repr__(self):
        return "CrossScopeProfile(disabled)"


_NULL_CROSS_SCOPE = _NullCrossScopeProfile()


class _ActiveCrossScopeProfile:
    __slots__ = ("_name", "_ctx", "_active")

    def __init__(self, name: str):
        self._name = name
        self._ctx: Optional[AbstractContextManager] = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def name(self) -> str:
        return self._name

    def start(self):
        if self._active:
            raise RuntimeError(
                f"CrossScopeProfile '{self._name}' is already active. "
                f"Call end() before starting again."
            )

        ctx = _PROFILER_FUNC(self._name)
        ctx.__enter__()
        self._ctx = ctx
        self._active = True

    def end(self):
        if not self._active:
            return
        self._active = False
        ctx = self._ctx
        self._ctx = None
        if ctx is not None:
            ctx.__exit__(None, None, None)

    def __del__(self):
        if self._active:
            try:
                self.end()
            except Exception:
                pass

    def __repr__(self):
        state = "active" if self._active else "inactive"
        return f"CrossScopeProfile('{self._name}', {state})"


def profile_scope(name: str) -> AbstractContextManager:
    return _PROFILER_FUNC(name)


def profile_func(name_suffix: Callable[..., str] = None):
    def decorator(func: Callable) -> Callable:
        if _STRATEGY == "disabled":
            return func

        func_name = func.__name__

        if name_suffix is not None:

            @wraps(func)
            def wrapper_with_suffix(*args, **kwargs):
                try:
                    suffix = name_suffix(*args, **kwargs)
                    profile_name = f"{func_name}{suffix}"
                except Exception:
                    profile_name = func_name

                with _PROFILER_FUNC(profile_name):
                    return func(*args, **kwargs)

            return wrapper_with_suffix
        else:

            @wraps(func)
            def wrapper_simple(*args, **kwargs):
                with _PROFILER_FUNC(func_name):
                    return func(*args, **kwargs)

            return wrapper_simple

    return decorator


def profile_cross_scope(
    name: str,
) -> Union[_NullCrossScopeProfile, _ActiveCrossScopeProfile]:
    if _STRATEGY == "disabled":
        return _NULL_CROSS_SCOPE
    return _ActiveCrossScopeProfile(name)
