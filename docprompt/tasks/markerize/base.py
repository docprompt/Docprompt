from typing import Optional

from pydantic import BaseModel

from docprompt.tasks.base import (
    BasePageResult,
    ProviderAgnosticOAI,
    SupportsDirectInvocation,
    SupportsOpenAIMessages,
    SupportsParsing,
    SupportsTaskConfig,
)


class MarkerizeResult(BasePageResult):
    task_name = "markerize"
    raw_markdown: str


class MarkerizeConfig(BaseModel):
    system_prompt: Optional[str] = None
    human_prompt: Optional[str] = None


class BaseLLMMarkerizeProvider(
    ProviderAgnosticOAI,
    SupportsTaskConfig[MarkerizeConfig],
    SupportsOpenAIMessages[bytes],
    SupportsDirectInvocation[bytes, MarkerizeConfig, MarkerizeResult],
    SupportsParsing[MarkerizeResult],
):
    class Meta:
        abstract = True
