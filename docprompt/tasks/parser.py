"""The base output parser that seeks to mimic the langhain implementation."""

from abc import abstractmethod
from typing import TypeVar

from pydantic import BaseModel
from typing_extensions import Generic

TTaskInput = TypeVar("TTaskInput", bound=BaseModel)
TTaskOutput = TypeVar("TTaskOutput", bound=BaseModel)


class BaseOutputParser(BaseModel, Generic[TTaskInput, TTaskOutput]):
    """The output parser for the page classification system."""

    @abstractmethod
    def from_task_input(
        cls, task_input: TTaskInput
    ) -> "BaseOutputParser[TTaskInput, TTaskOutput]":
        """Create an output parser from the task input."""

    @abstractmethod
    def parse(self, text: str) -> TTaskOutput:
        """Parse the results of the classification task."""
