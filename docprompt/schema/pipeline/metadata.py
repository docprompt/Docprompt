"""The metadata class is utilized for defining a basic, yet flexible interface for metadata attached to various fields.

In essence, this allows for developers to choose to either create their metadtata
in an unstructured manner (i.e. a dictionary), or to sub class the base metadata class in order to
create a more strictly typed metadata model for their page and document nodes.
"""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Dict, Generic, TypeVar, Union

from pydantic import BaseModel, Field, PrivateAttr, model_validator

if TYPE_CHECKING:
    from docprompt.schema.pipeline.node import DocumentNode, PageNode
    from docprompt.tasks.base import BaseResult


TBaseTaskResult = TypeVar("TBaseTaskResult", bound="BaseResult")
TMetadataOwner = TypeVar("TMetadataOwner", bound=Union["DocumentNode", "PageNode"])


class TaskResultsDescriptor:
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if "_results" not in obj.extra:
            obj.extra["_results"] = {}
        return obj.extra["_results"]

    def __set__(self, obj, value):
        raise AttributeError(
            "Can't set task_results directly. Use task_results['key'] = value instead."
        )

    def __delete__(self, obj):
        obj.extra["_results"] = {}


class BaseMetadata(BaseModel, MutableMapping, Generic[TMetadataOwner]):
    """
    The base metadata class is utilized for defining a basic yet flexible interface
    for metadata attached to various fields.

    The metadata class can be used in two ways:
        1. As a dictionary-like object, where metadata is stored in the `extra` field.
        2. As a sub-classed model, where metadata is stored in the fields of the model.

    When used out of the box, the metadata class will adobpt dictionary-like behavior. You
    may easily access different fields of the metadata as if it were a dictionary:
    ```python
    # Instantiate it with any kwargs you like
    metadata = BaseMetadata(foo-'bar', cow='moo')

    metadata["foo"]  # "bar"
    metadata["cow"]  # "moo"

    # Update the value of the key
    metadata["foo"] = "fighters"

    # Set new key-value pairs
    metadata['sheep'] = 'baa'
    ```

    Otherwise, you may sub-class the metadata class in order to create a more strictly typed
    metadata model. This is useful when you want to enforce a specific structure for your metadata.

    ```python
    class CustomMetadata(BaseMetadata):
        foo: str
        cow: str

    # Instantiate it with the required fields
    metadata = CustomMetadata(foo='bar', cow='moo')

    metadata.foo  # "bar"
    metadata.cow  # "moo"

    # Update the value of the key
    metadata.foo = "fighters"

    # Use the extra field to store dynamic metadata
    metadata.extra['sheep'] = 'baa'
    ```

    Additionally, the task results descriptor allows for controlled and easy access to the task results
    of various tasks that are run on the parent node.
    """

    extra: Dict[str, Any] = Field(..., default_factory=dict, repr=False)

    _task_results: TaskResultsDescriptor = PrivateAttr(
        default_factory=TaskResultsDescriptor
    )

    _owner: TMetadataOwner = PrivateAttr()

    @property
    def task_results(self) -> TaskResultsDescriptor:
        return self._task_results

    @property
    def owner(self) -> TMetadataOwner:
        """Return the owner of the metadata.

        NOTE: We avoid using a standard property here, due to conflicts with the custom
        __getattr__ implementation.
        """
        return self._owner

    def set_owner(self, owner: TMetadataOwner) -> None:
        """Return the owner of the metadata.

        NOTE: We avoid using a standard setter here, due to conflicts with the custom
        __setattr__ implementation.
        """
        self._owner = owner

    @classmethod
    def from_owner(cls, owner: TMetadataOwner, **data) -> BaseMetadata:
        """Create a new instance of the metadata class with the owner set."""
        metadata = cls(**data)
        metadata.set_owner(owner)
        return metadata

    @model_validator(mode="before")
    @classmethod
    def validate_data_fields_from_annotations(cls, data: Any) -> Any:
        """Validate the data fields from the annotations."""

        # We want to make sure that we combine the `extra` metdata along with any
        # other specific fields that are defined in the metadata.
        extra = data.pop("extra", {})
        data = {**data, **extra}

        # If the model has been sub-classed, then all of our fields must be
        # validated by the pydantic model.
        if cls._is_field_typed():
            return data

        # Otherwise, we are using our mock-dict implentation, so we store our
        # metadata in the `extra` field.
        return {"extra": data}

    @classmethod
    def _is_field_typed(cls):
        """
        Check if the metadata model is field typed.

        This is used to determine if the metadata model is a dictionary-like model,
        or a more strictly typed model.
        """
        if set(["extra"]) != set(cls.model_fields.keys()):
            return True

        return False

    def __repr__(self):
        """
        Provide a string representation of the metadata.

        This only works for the base metadata model. If sub-classed, this will raise an error,
        unless overridden, as BaseModel's do not have a __repr__ method.
        """
        if self._is_field_typed():
            return super().__repr__()

        # Otherwise, we are deailing with dictornary-like metadata
        return json.dumps(self.extra)

    def __getitem__(self, name):
        """
        Provide dictionary functionlaity to the metadata class.

        This only works for the base metadata model. If sub-classed, this will raise an error,
        unless overridden, as BaseModel's do not have an __getitem__ method.
        """
        if not self._is_field_typed():
            return self.extra[name]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setitem__(self, name, value):
        """
        Provide dictionary functionality to the metadata class.

        This only works for the base metadata model. If sub-classed, this will raise an error,
        unless overridden, as BaseModel's do not have an __setitem__ method.
        """
        if not self._is_field_typed():
            self.extra[name] = value
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __delitem__(self, name):
        """
        Provide dictionary functionality to the metadata class.

        This only works for the base metadata model. If sub-classed, this will raise an error,
        unless overridden, as BaseModel's do not have an __delitem__ method.
        """
        if not self._is_field_typed():
            del self.extra[name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __iter__(self):
        """
        Iterate over the keys in the metadata.

        This only works for the base metadata model. If sub-classed, this will raise an error,
        unless overridden, as BaseModel's do not have an __iter__ method.
        """
        if self._is_field_typed():
            raise AttributeError(f"'{self.__class__.__name__}' object is not iterable")

        return iter(self.extra)

    def __len__(self):
        """
        Get the number of keys in the metadata.

        This only works for the base metadata model. If sub-classed, this will raise an error,
        unless overridden, as BaseModel's do not have a __len__ method.
        """
        if self._is_field_typed():
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '__len__'"
            )

        return len(self.extra)

    def __getattr__(self, name):
        """
        Allow for getting of attributes on the metadata class.

        The attributes are retrieved through the following heirarchy:
            - If the model is sub-classed, it will be retrieved as normal.
            - Otherwise, if the attribute is private, it will be retrieved as normal.
            - Finally, if we are getting a public attribute on the base metadata class,
                we use the extra field.
            - If the key is not set in the `extra` dictionary, we resort back to just
            trying to get the field.
                - This is when we grab the `owner` or `task_result` attribuite.
        """
        if self._is_field_typed():
            return super().__getattr__(name)

        if name.startswith("_"):
            return super().__getattr__(name)

        # Attempt to retreieve the attr from the `extra` field
        try:
            return self.extra.get(name)

        except KeyError:
            # This is for grabbing properties on the base metadata class
            return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Allow for setting of attributes on the metadata class.

        The attributes are set through the following heirarchy:
            - If the model is sub-classed, it will be set as normal.
            - Otherwise, if the attribute is private, it will be set as normal.
            - Finally, if we are setting a public attribute on the base metadata class,
                we use the extra field.
        """
        if self._is_field_typed():
            return super().__setattr__(name, value)

        # We want to avoid setting any private attributes in the extra
        # dictionary
        if name.startswith("_"):
            return super().__setattr__(name, value)

        self.extra[name] = value
