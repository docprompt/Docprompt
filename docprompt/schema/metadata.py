"""The metadata class is utilized for defining a basic, yet flexible interface for metadata attached to various fields.

In essence, this allows for developers to choose to either create their metadtata
in an unstructured manner (i.e. a dictionary), or to sub class the base metadata class in order to
create a more strictly typed metadata model for their page and document nodes.
"""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, Field, PrivateAttr, model_validator

if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode
DocumentNode = TypeVar("DocumentNode", bound="DocumentNode")


class BaseMetadata(BaseModel, MutableMapping):
    """The base metadata class is utilized for defining a basic yet flexible interface
    for metadata attached to various fields.
    """

    extra: dict[str, Any] = Field(..., default_factory=dict, repr=False)

    _document: DocumentNode = (
        PrivateAttr()
    )  # TODO: Implement automated setting of this value

    @property
    def owner(self) -> DocumentNode:
        """Return the owner of the metadata."""
        return self._document

    @classmethod
    def from_owner(cls, owner: DocumentNode, **data: Any) -> BaseMetadata:
        """Create a new instance of the metadata class with the owner set."""
        metadata = cls(**data)
        metadata._document = owner
        return metadata

    @model_validator(mode="before")
    @classmethod
    def validate_data_fields_from_annotations(cls, data: Any) -> Any:
        """Validate the data fields from the annotations."""

        # We want to make sure that we combine the `extra` metdata along with any
        # other specific fields that are defined in the metadata.
        extra = data.pop("extra", {})
        data = {**data, **extra}

        if cls._is_field_typed():
            # Make sure we extract everything out of the `extra` key, if we
            # are dealing with a typed field.
            return data

        # Otherwise, we are dealing with a dictionary-like metadata, and we
        # want to dump everything under the `extra` key.
        extra = data.pop("extra", {})
        data = {**data, **extra}
        return {"extra": data}

    @classmethod
    def _is_field_typed(cls):
        if set(["extra"]) != set(cls.model_fields.keys()):
            return True

        return False

    def __repr__(self):
        if self._is_field_typed():
            return super().__repr__()

        # Otherwise, we are deailing with dictornary-like metadata
        return json.dumps(self.extra)

    def __getitem__(self, name):
        """Provide dictionary functionlaity to the metadata class."""
        if not self._is_field_typed():
            return self.extra[name]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setitem__(self, name, value):
        if not self._is_field_typed():
            self.extra[name] = value
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __delitem__(self, name):
        if not self._is_field_typed():
            del self.extra[name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __iter__(self):
        if self._is_field_typed():
            raise AttributeError(f"'{self.__class__.__name__}' object is not iterable")

        return iter(self.extra)

    def __len__(self):
        if self._is_field_typed():
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '__len__'"
            )

        return len(self.extra)

    def __getattr__(self, name):
        if self._is_field_typed():
            return super().__getattr__(name)
        else:
            return self.extra.get(name)

    def __setattr__(self, name: str, value) -> None:
        if self._is_field_typed():
            return super().__setattr__(name, value)
        else:
            # We want to avoid setting any private attributes in the extra
            # dictionary
            if name.startswith("_"):
                return super().__setattr__(name, value)

            self.extra[name] = value


class Metadata(BaseMetadata):
    """Concrete metadata test."""

    title: str = Field(..., title="The title of the document.")
    description: str = Field(..., title="The description of the document.")
