"""
The core primatives for any language model interfacing. Docprompt uses these for the prompt garden, but
supports free conversion to and from these types from other libaries.
"""

from typing import List, Literal, Union, Optional
from pydantic import BaseModel, model_validator


class OpenAIImageURL(BaseModel):
    url: str


class OpenAIComplexContent(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[OpenAIImageURL] = None

    @model_validator(mode="after")
    def validate_content(cls, v):
        if v.type == "text" and v.text is None:
            raise ValueError("Text content must be provided when type is 'text'")
        if v.type == "image_url" and v.image_url is None:
            raise ValueError(
                "Image URL content must be provided when type is 'image_url'"
            )

        if v.text is not None and v.image_url is not None:
            raise ValueError("Only one of text or image_url can be provided")

        return v


class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[OpenAIComplexContent]]

    def to_langchain_message(self):
        try:
            from langchain.schema import SystemMessage, HumanMessage, AIMessage
        except ImportError:
            raise ImportError(
                "Could not import langchain.schema. Install with `docprompt[langchain]`"
            )

        role_mapping = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage,
        }

        dumped = self.model_dump(mode="json", exclude_unset=True, exclude_none=True)

        return role_mapping[self.role](content=dumped["content"])

    def to_openai(self):
        return self.model_dump(mode="json", exclude_unset=True, exclude_none=True)

    def to_llamaindex_chat_message(self):
        try:
            from llama_index.core.base.llms.types import ChatMessage, MessageRole
        except ImportError:
            raise ImportError(
                "Could not import llama_index.core. Install with `docprompt[llamaindex]`"
            )

        role_mapping = {
            "system": MessageRole.SYSTEM,
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
        }

        dumped = self.model_dump(mode="json", exclude_unset=True, exclude_none=True)

        return ChatMessage.from_str(
            content=dumped["content"], role=role_mapping[self.role]
        )
