"""The test suite for the base task provider seeks to ensure that all of the
builtin functionality of the BaseTaskProvider is proeprly implemented.
"""

import pytest

from docprompt.tasks.base import AbstractTaskProvider


class TestAbstractTaskProviderBaseFunctionliaty:
    """
    Test that the BaseTaskProvider interface provides the correct expected basic
    functionality to be inherited by all subclasses.

    This includes:
        - the model validaton asserts `name` and `capabilities` are required
        - the intialization of the model properly sets invoke kwargs
        - the `ainvoke` method calling the `_ainvoke` method
        - the `invoke` method calling the `_invoke` method
    """

    def test_model_validator_raises_error_on_missing_name(self):
        class BadTaskProvider(AbstractTaskProvider):
            capabilities = []

        with pytest.raises(ValueError):
            BadTaskProvider.validate_class_vars({})

    def test_model_validator_raises_error_on_missing_capabilities(self):
        class BadTaskProvider(AbstractTaskProvider):
            name = "BadTaskProvider"

        with pytest.raises(ValueError):
            BadTaskProvider.validate_class_vars({})

    def test_model_validator_raises_error_on_empty_capabilities(self):
        class BadTaskProvider(AbstractTaskProvider):
            name = "BadTaskProvider"
            capabilities = []

        with pytest.raises(ValueError):
            BadTaskProvider.validate_class_vars({})

    def test_init_no_invoke_kwargs(self):
        class TestTaskProvider(AbstractTaskProvider):
            name = "TestTaskProvider"
            capabilities = ["test"]

        provider = TestTaskProvider()

        assert provider._default_invoke_kwargs == {}

    def test_init_with_invoke_kwargs(self):
        class TestTaskProvider(AbstractTaskProvider):
            name = "TestTaskProvider"
            capabilities = ["test"]

        kwargs = {"test": "test"}
        provider = TestTaskProvider(invoke_kwargs=kwargs)

        assert provider._default_invoke_kwargs == kwargs

    def test_init_with_fields_and_invoke_kwargs(self):
        class TestTaskProvider(AbstractTaskProvider):
            name = "TestTaskProvider"
            capabilities = ["test"]

            foo: str

        kwargs = {"test": "test"}
        provider = TestTaskProvider(foo="bar", invoke_kwargs=kwargs)

        assert provider._default_invoke_kwargs == kwargs
        assert provider.foo == "bar"

    @pytest.mark.asyncio
    async def test_ainvoke_calls__ainvoke(self):
        class TestTaskProvider(AbstractTaskProvider):
            name = "TestTaskProvider"
            capabilities = ["test"]

            async def _ainvoke(self, input, config=None, **kwargs):
                return input

        provider = TestTaskProvider()

        assert await provider.ainvoke([1, 2, 3]) == [1, 2, 3]

    def test_invoke_calls__invoke(self):
        class TestTaskProvider(AbstractTaskProvider):
            name = "TestTaskProvider"
            capabilities = ["test"]

            def _invoke(self, input, config=None, **kwargs):
                return input

        provider = TestTaskProvider()

        assert provider.invoke([1, 2, 3]) == [1, 2, 3]
