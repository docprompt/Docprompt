import asyncio
import sys
from functools import partial, update_wrapper, wraps
from typing import Callable, Optional, Set, Tuple, Type

if sys.version_info >= (3, 9):
    to_thread = asyncio.to_thread
else:

    def to_thread(func, /, *args, **kwargs):
        @wraps(func)
        async def wrapper():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # If there's no running event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            pfunc = partial(func, *args, **kwargs)
            return await loop.run_in_executor(None, pfunc)

        return wrapper()


def get_closest_attr(cls: Type, attr_name: str) -> Tuple[Type, Optional[Callable], int]:
    closest_cls = cls
    attr = getattr(cls.__dict__, attr_name, None)
    depth = 0

    if attr and hasattr(attr, "_original"):
        attr = None
    elif attr:
        return (cls, attr, 0)

    for idx, base in enumerate(cls.__mro__, start=1):
        if not attr and attr_name in base.__dict__:
            if not hasattr(base.__dict__[attr_name], "_original"):
                closest_cls = base
                attr = base.__dict__[attr_name]
                depth = idx

        if attr:
            break

    return (closest_cls, attr, depth)


def validate_method(cls, name: str, method: Callable, expected_async: bool):
    if method is None:
        return None
    is_async = asyncio.iscoroutinefunction(method)
    if is_async != expected_async:
        return f"Method '{name}' in {cls.__name__} should be {'async' if expected_async else 'sync'}, but it's {'async' if is_async else 'sync'}"

    return None


def apply_dual_methods_to_cls(cls: Type, method_group: Tuple[str, str]):
    errors = []

    sync_name, async_name = method_group

    sync_trace = get_closest_attr(cls, sync_name)
    async_trace = get_closest_attr(cls, async_name)

    sync_cls, sync_method, sync_depth = sync_trace
    async_cls, async_method, async_depth = async_trace

    if sync_method:
        sync_error = validate_method(cls, sync_name, sync_method, False)
        if sync_error:
            errors.append(sync_error)

    if async_method:
        async_error = validate_method(cls, async_name, async_method, True)
        if async_error:
            errors.append(async_error)

    if (
        sync_method is None
        and async_method is None
        and not getattr(getattr(cls, "Meta", None), "abstract", False)
    ):
        return [
            f"{cls.__name__} must implement at least one of these methods: {sync_name}, {async_name}"
        ]

    if sync_cls is cls and async_cls is cls and sync_method and async_method:
        return errors  # Both methods are already in the same class

    if async_cls is cls and async_method:

        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_method(*args, **kwargs))

        update_wrapper(sync_wrapper, async_method)

        sync_wrapper._original = async_method

        setattr(cls, sync_name, sync_wrapper)
    elif sync_cls is cls and sync_method:

        async def async_wrapper(*args, **kwargs):
            if hasattr(sync_method, "__func__"):
                return await to_thread(sync_method, *args, **kwargs)
            return await to_thread(sync_method, *args, **kwargs)

        update_wrapper(async_wrapper, sync_method)

        async_wrapper._original = sync_method

        setattr(cls, async_name, async_wrapper)
    else:
        if async_depth < sync_depth:

            def sync_wrapper(*args, **kwargs):
                return asyncio.run(async_method(*args, **kwargs))

            update_wrapper(sync_wrapper, async_method)

            sync_wrapper._original = async_method

            setattr(cls, sync_name, sync_wrapper)
        else:

            async def async_wrapper(*args, **kwargs):
                return await to_thread(sync_method, *args, **kwargs)

            update_wrapper(async_wrapper, sync_method)

            async_wrapper._original = sync_method

            setattr(cls, async_name, async_wrapper)

    return errors


def get_flexible_method_configs(cls: Type) -> Set[Tuple[str, str]]:
    all = set()
    for base in cls.__mro__:
        all.update(getattr(base, "__flexible_methods__", set()))

    return all


def flexible_methods(*method_groups: Tuple[str, str]):
    def decorator(cls: Type):
        if not hasattr(cls, "__flexible_methods__"):
            setattr(cls, "__flexible_methods__", set())

        for base in cls.__bases__:
            if hasattr(base, "__flexible_methods__"):
                cls.__flexible_methods__.update(base.__flexible_methods__)

        cls.__flexible_methods__.update(method_groups)

        def apply_flexible_methods(cls: Type):
            errors = []

            for group in get_flexible_method_configs(cls):
                if len(group) != 2:
                    errors.append(
                        f"Invalid method group {group}. Each group must be a tuple of exactly two method names."
                    )
                    continue

                errors.extend(apply_dual_methods_to_cls(cls, group))

            if errors:
                raise TypeError("\n".join(errors))

        apply_flexible_methods(cls)

        original_init_subclass = cls.__init_subclass__

        @classmethod
        def new_init_subclass(cls, **kwargs):
            original_init_subclass(**kwargs)
            apply_flexible_methods(cls)

        cls.__init_subclass__ = new_init_subclass

        return cls

    return decorator
