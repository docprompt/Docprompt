"""Test the Base Metadata class to ensure that all of it's complex behavior is working correctly."""

from typing import Optional
from unittest.mock import MagicMock

import pytest
from pydantic import Field, ValidationError

from docprompt.schema.pipeline.metadata import BaseMetadata, TaskResultsDescriptor


class TestTaskResultDescriptor:
    """Ensure that the TaskResultsDescriptor class is working correctly."""

    @pytest.fixture
    def obj(self):
        """This `obj` object mocks the metadata class."""
        obj = MagicMock()
        obj.extra = {}
        return obj

    def test_get(self, obj):
        """Ensure that the __get__ method is working correctly."""
        descriptor = TaskResultsDescriptor()

        # Test when _results doesn't exist
        result = descriptor.__get__(obj)
        assert result == {}
        assert obj.extra["_results"] == {}

        # Test when _results already exists
        obj.extra["_results"] = {"existing": "data"}
        result = descriptor.__get__(obj)
        assert result == {"existing": "data"}

        # Test when obj is None
        assert descriptor.__get__(None) == descriptor

    def test_set(self, obj):
        """Ensure that attempting to set the task results raises an AttributeError."""
        descriptor = TaskResultsDescriptor()

        with pytest.raises(AttributeError):
            descriptor.__set__(obj, "value")

    def test_delete(self, obj):
        """Ensure that the __delete__ method is working correctly."""
        descriptor = TaskResultsDescriptor()
        obj.extra["_results"] = {"some": "data"}

        descriptor.__delete__(obj)
        assert obj.extra["_results"] == {}


class TestBaseMetadataAsDictLike:
    """
    Test that when we do not sub-class the BaseMetadata class, it behaves
    like a dictionary.
    """

    @pytest.fixture
    def metadata(self):
        return BaseMetadata(foo="bar", baz=42)

    def test_getitem(self, metadata):
        assert metadata["foo"] == "bar"
        assert metadata["baz"] == 42

    def test_setitem(self, metadata):
        metadata["new_key"] = "new_value"
        assert metadata["new_key"] == "new_value"

    def test_delitem(self, metadata):
        del metadata["foo"]
        with pytest.raises(KeyError):
            _ = metadata["foo"]

    def test_len(self, metadata):
        assert len(metadata) == 2

    def test_iter(self, metadata):
        assert set(metadata) == {"foo", "baz"}

    def test_contains(self, metadata):
        assert "foo" in metadata
        assert "nonexistent" not in metadata

    def test_get(self, metadata):
        assert metadata.get("foo") == "bar"
        assert metadata.get("nonexistent", "default") == "default"

    def test_update(self, metadata):
        metadata.update({"new_key": "new_value", "foo": "updated"})
        assert metadata["new_key"] == "new_value"
        assert metadata["foo"] == "updated"

    def test_owner_property(self):
        owner = MagicMock()
        metadata = BaseMetadata.from_owner(owner, foo="bar")
        assert metadata.owner == owner

    def test_task_results(self, metadata):
        assert isinstance(metadata.task_results, dict)
        metadata.task_results["task1"] = {"result": "success"}
        assert metadata.task_results["task1"] == {"result": "success"}

    def test_task_results_blocks_set(self, metadata):
        metadata.task_results["task1"] = {"result": "success"}
        before = metadata.task_results
        with pytest.raises(AttributeError):
            metadata.task_results = {}
        assert before is metadata.task_results

    def test_task_results_allows_del(self, metadata):
        metadata.task_results["task1"] = {"result": "success"}
        del metadata.task_results
        assert metadata.task_results == {}

    def test_repr(self, metadata):
        assert repr(metadata) == '{"foo": "bar", "baz": 42}'

    def test_attribute_access(self, metadata):
        assert metadata.foo == "bar"
        assert metadata.baz == 42

        metadata.new_attr = "new_value"
        assert metadata.new_attr == "new_value"
        assert metadata["new_attr"] == "new_value"

    def test_is_field_typed(self, metadata):
        assert not metadata._is_field_typed()

    def test_from_owner(self):
        owner = MagicMock()
        metadata = BaseMetadata.from_owner(owner, foo="bar")
        assert metadata.owner == owner
        assert metadata["foo"] == "bar"


class CustomMetadata(BaseMetadata):
    """A custom metadata model for testing purposes."""

    field1: str = Field(...)
    field2: int = Field(...)
    optional_field: Optional[float] = Field(None)


class TestCustomMetadata:
    """Test the behavior of a subclassed BaseMetadata model."""

    @pytest.fixture
    def custom_metadata(self):
        return CustomMetadata(
            field1="test", field2=42, extra={"extra_field": "extra_value"}
        )

    def test_initialization(self, custom_metadata):
        assert custom_metadata.field1 == "test"
        assert custom_metadata.field2 == 42
        assert custom_metadata.optional_field is None
        assert custom_metadata.extra == {"extra_field": "extra_value"}

    def test_attribute_access_and_modification(self, custom_metadata):
        assert custom_metadata.field1 == "test"
        custom_metadata.field1 = "modified"
        assert custom_metadata.field1 == "modified"

        custom_metadata.optional_field = 3.14
        assert custom_metadata.optional_field == 3.14

    def test_dictionary_like_operations(self, custom_metadata):
        with pytest.raises(AttributeError):
            _ = custom_metadata["field1"]

        with pytest.raises(AttributeError):
            custom_metadata["new_field"] = "value"

        with pytest.raises(AttributeError):
            del custom_metadata["field1"]

    def test_extra_fields(self, custom_metadata):
        assert custom_metadata.extra["extra_field"] == "extra_value"
        custom_metadata.extra["new_extra"] = "new_value"
        assert custom_metadata.extra["new_extra"] == "new_value"

    def test_validation(self):
        with pytest.raises(ValidationError):
            CustomMetadata(field1=42, field2="not an int")

    def test_is_field_typed(self, custom_metadata):
        assert custom_metadata._is_field_typed()

    def test_repr(self, custom_metadata):
        assert "field1='test'" in repr(custom_metadata)
        assert "field2=42" in repr(custom_metadata)
        assert "optional_field=None" in repr(custom_metadata)

    def test_iteration_not_allowed(self, custom_metadata):
        with pytest.raises(AttributeError):
            list(custom_metadata)

    def test_len_not_allowed(self, custom_metadata):
        with pytest.raises(AttributeError):
            len(custom_metadata)

    def test_owner_property(self):
        owner = MagicMock()
        metadata = CustomMetadata.from_owner(owner, field1="test", field2=42)
        assert metadata.owner == owner
        assert metadata.field1 == "test"
        assert metadata.field2 == 42

    def test_task_results(self, custom_metadata):
        assert isinstance(custom_metadata.task_results, dict)
        custom_metadata.task_results["task1"] = {"result": "success"}
        assert custom_metadata.task_results["task1"] == {"result": "success"}

    def test_task_results_blocks_set(self, custom_metadata):
        custom_metadata.task_results["task1"] = {"result": "success"}
        before = custom_metadata.task_results
        with pytest.raises(AttributeError):
            custom_metadata.task_results = {}
        assert before is custom_metadata.task_results

    def test_task_results_allows_del(self, custom_metadata):
        custom_metadata.task_results["task1"] = {"result": "success"}
        del custom_metadata.task_results
        assert custom_metadata.task_results == {}
