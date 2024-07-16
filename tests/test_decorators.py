import asyncio

import pytest

# Assuming the flexible_methods decorator is imported from your module
from docprompt._decorators import flexible_methods


@pytest.fixture(scope="function")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Fixture for running coroutines
@pytest.fixture(scope="function")
def run_async(event_loop):
    def _run_async(coro):
        return event_loop.run_until_complete(coro)

    return _run_async


# Test classes
@flexible_methods(("sync_method", "async_method"))
class BaseClass:
    def sync_method(self):
        return "base_sync"


class ChildWithSync(BaseClass):
    def sync_method(self):
        return "child_sync"


class ChildWithAsync(BaseClass):
    async def async_method(self):
        return "child_async"


class ChildWithBoth(BaseClass):
    def sync_method(self):
        return "child_both_sync"

    async def async_method(self):
        return "child_both_async"


# Successful cases
def test_base_class(run_async):
    base = BaseClass()
    assert base.sync_method() == "base_sync"
    assert run_async(base.async_method()) == "base_sync"


def test_child_with_sync(run_async):
    child = ChildWithSync()
    assert child.sync_method() == "child_sync"
    assert run_async(child.async_method()) == "child_sync"


def test_child_with_async(run_async):
    child = ChildWithAsync()
    assert child.sync_method() == "child_async"
    assert run_async(child.async_method()) == "child_async"


def test_child_with_both(run_async):
    child = ChildWithBoth()
    assert child.sync_method() == "child_both_sync"
    assert run_async(child.async_method()) == "child_both_async"


# Failure modes
def test_wrong_sync_async():
    with pytest.raises(TypeError) as excinfo:

        @flexible_methods(("method", "method_async"))
        class ErrorClass:
            async def method(self):
                return "wrong_sync"

            def method_async(self):
                return "wrong_async"

    error_message = str(excinfo.value)
    assert (
        "Method 'method' in ErrorClass should be sync, but it's async" in error_message
    )
    assert (
        "Method 'method_async' in ErrorClass should be async, but it's sync"
        in error_message
    )


def test_no_implementation():
    with pytest.raises(TypeError) as excinfo:

        @flexible_methods(("new_method", "new_method_async"))
        class ErrorClass:
            pass

    assert (
        "ErrorClass must implement at least one of these methods: new_method, new_method_async"
        in str(excinfo.value)
    )


def test_invalid_group():
    with pytest.raises(TypeError) as excinfo:

        @flexible_methods(("single_method",))
        class ErrorClass:
            pass

    assert (
        "Invalid method group ('single_method',). Each group must be a tuple of exactly two method names."
        in str(excinfo.value)
    )


def test_error_in_subclass():
    @flexible_methods(("method", "method_async"))
    class BaseClass:
        def method(self):
            return "base_sync"

    with pytest.raises(TypeError) as excinfo:

        class ErrorSubclass(BaseClass):
            async def method(self):
                return "wrong_sync"

        assert "Method 'method' in ErrorSubclass should be sync, but it's async" in str(
            excinfo.value
        )


# Advanced async/sync wrapping tests
def test_sync_to_async_wrapping(run_async):
    @flexible_methods(("sync_method", "async_method"))
    class SyncOnly:
        def sync_method(self):
            return "sync_result"

    obj = SyncOnly()
    assert obj.sync_method() == "sync_result"
    assert asyncio.iscoroutinefunction(obj.async_method)
    assert run_async(obj.async_method()) == "sync_result"


def test_async_to_sync_wrapping(event_loop):
    @flexible_methods(("sync_method", "async_method"))
    class AsyncOnly:
        async def async_method(self):
            await asyncio.sleep(0.1)
            return "async_result"

    obj = AsyncOnly()
    assert not asyncio.iscoroutinefunction(obj.sync_method)
    assert obj.sync_method() == "async_result"
    assert event_loop.run_until_complete(obj.async_method()) == "async_result"


def test_async_wrapper_preserves_sync_behavior(run_async):
    @flexible_methods(("sync_method", "async_method"))
    class SyncWithSideEffect:
        def __init__(self):
            self.called = False

        def sync_method(self):
            self.called = True
            return "sync_with_side_effect"

    obj = SyncWithSideEffect()
    assert run_async(obj.async_method()) == "sync_with_side_effect"
    assert obj.called, "Sync method side effect should be preserved in async wrapper"


def test_sync_wrapper_preserves_async_behavior(event_loop):
    @flexible_methods(("sync_method", "async_method"))
    class AsyncWithSideEffect:
        def __init__(self):
            self.called = False

        async def async_method(self):
            await asyncio.sleep(0.1)
            self.called = True
            return "async_with_side_effect"

    obj = AsyncWithSideEffect()
    assert obj.sync_method() == "async_with_side_effect"
    assert obj.called, "Async method side effect should be preserved in sync wrapper"


def test_multiple_method_groups(run_async):
    @flexible_methods(("method1", "method1_async"), ("method2", "method2_async"))
    class MultiGroupClass:
        def method1(self):
            return "sync1"

        async def method2_async(self):
            return "async2"

    obj = MultiGroupClass()
    assert obj.method1() == "sync1"
    assert obj.method2() == "async2"
    # assert run_async(obj.method1_async()) == "sync1"
    # assert run_async(obj.method2_async()) == "async2"


def test_inheritance_and_overriding(run_async):
    @flexible_methods(("method", "method_async"))
    class Base:
        def method(self):
            return "base"

    class Child1(Base):
        async def method_async(self):
            return "child1_async"

    class Child2(Base):
        def method(self):
            return "child2_sync"

    base = Base()
    child1 = Child1()
    child2 = Child2()

    assert base.method() == "base"
    assert run_async(base.method_async()) == "base"

    assert child1.method() == "child1_async"
    assert run_async(child1.method_async()) == "child1_async"

    assert child2.method() == "child2_sync"
    assert run_async(child2.method_async()) == "child2_sync"


def test_abstract_base_class():
    from abc import ABC, abstractmethod

    @flexible_methods(("abstract_method", "abstract_method_async"))
    class AbstractBase(ABC):
        @abstractmethod
        def abstract_method(self):
            pass

    class ConcreteSync(AbstractBase):
        def abstract_method(self):
            return "concrete_sync"

    class ConcreteAsync(AbstractBase):
        async def abstract_method_async(self):
            return "concrete_async"

    with pytest.raises(TypeError):
        AbstractBase()

    sync_instance = ConcreteSync()
    assert sync_instance.abstract_method() == "concrete_sync"
    assert asyncio.run(sync_instance.abstract_method_async()) == "concrete_sync"

    async_instance = ConcreteAsync()
    assert async_instance.abstract_method() == "concrete_async"
    assert asyncio.run(async_instance.abstract_method_async()) == "concrete_async"


def test_multiple_inheritance():
    @flexible_methods(("method1", "method1_async"))
    class Base1:
        def method1(self):
            return "base1"

    @flexible_methods(("method2", "method2_async"))
    class Base2:
        async def method2_async(self):
            return "base2"

    class Child(Base1, Base2):
        async def method1_async(self):
            return "child1"

        def method2(self):
            return "child2"

    child = Child()
    assert child.method1() == "child1"
    assert asyncio.run(child.method1_async()) == "child1"
    assert child.method2() == "child2"
    assert asyncio.run(child.method2_async()) == "child2"

    # Test that Base1 and Base2 methods are not affected
    base1 = Base1()
    base2 = Base2()
    assert base1.method1() == "base1"
    assert asyncio.run(base1.method1_async()) == "base1"
    assert asyncio.run(base2.method2_async()) == "base2"
    assert base2.method2() == "base2"


def test_preserve_signature_and_docstring(run_async):
    @flexible_methods(("method", "method_async"))
    class PreserveMetadata:
        def method(self, arg1: int, arg2: str = "default") -> str:
            """This is a test method."""
            return f"{arg1} {arg2}"

    instance = PreserveMetadata()
    assert instance.method.__doc__ == "This is a test method."
    assert instance.method.__annotations__ == {"arg1": int, "arg2": str, "return": str}
    assert instance.method_async.__doc__ == "This is a test method."
    assert instance.method_async.__annotations__ == {
        "arg1": int,
        "arg2": str,
        "return": str,
    }

    assert instance.method(1, "test") == "1 test"
    assert run_async(instance.method_async(2, "async")) == "2 async"


@pytest.mark.skip(reason="Not implemented yet")
def test_static_methods():
    @flexible_methods(
        ("class_method", "class_method_async"), ("static_method", "static_method_async")
    )
    class MethodTypes:
        @classmethod
        def class_method(cls):
            return f"class {cls.__name__}"

        @staticmethod
        def static_method():
            return "static"

    assert MethodTypes.static_method() == "static"
    assert asyncio.run(MethodTypes.static_method_async()) == "static"

    instance = MethodTypes()
    assert instance.static_method() == "static"
    assert asyncio.run(instance.static_method_async()) == "static"
