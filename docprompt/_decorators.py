import asyncio
from functools import wraps
from typing import Callable, Tuple, Type

from docprompt.utils.async_utils import to_thread


def flexible_methods(*method_groups: Tuple[str, str]):
    def decorator(cls: Type):
        def get_method(cls: Type, name: str) -> Callable:
            return cls.__dict__.get(name)

        def validate_method(name: str, method: Callable, expected_async: bool):
            if method is None:
                return
            is_async = asyncio.iscoroutinefunction(method)
            if is_async != expected_async:
                return f"Method '{name}' in {cls.__name__} should be {'async' if expected_async else 'sync'}, but it's {'async' if is_async else 'sync'}"
            return None

        def apply_flexible_methods(cls: Type):
            errors = []

            is_abstract = getattr(getattr(cls, "Meta", None), "abstract", False)

            for group in method_groups:
                if len(group) != 2:
                    errors.append(
                        f"Invalid method group {group}. Each group must be a tuple of exactly two method names."
                    )
                    continue

                sync_name, async_name = group
                sync_method = cls.__dict__.get(sync_name)
                async_method = cls.__dict__.get(async_name)

                sync_error = validate_method(sync_name, sync_method, False)
                if sync_error:
                    errors.append(sync_error)

                async_error = validate_method(async_name, async_method, True)
                if async_error:
                    errors.append(async_error)

                if not sync_method and not async_method and not is_abstract:
                    errors.append(
                        f"{cls.__name__} must implement at least one of these methods: {sync_name}, {async_name}"
                    )

                if sync_method and not async_method:

                    @wraps(sync_method)
                    async def async_wrapper(*args, **kwargs):
                        return await to_thread(sync_method, *args, **kwargs)

                    setattr(cls, async_name, async_wrapper)

                elif async_method and not sync_method:

                    @wraps(async_method)
                    def sync_wrapper(*args, **kwargs):
                        return asyncio.run(async_method(*args, **kwargs))

                    setattr(cls, sync_name, sync_wrapper)

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
