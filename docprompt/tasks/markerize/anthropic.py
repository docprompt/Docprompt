from typing import Iterable, List, Optional

from bs4 import BeautifulSoup
from pydantic import Field

from docprompt.tasks.message import OpenAIComplexContent, OpenAIImageURL, OpenAIMessage
from docprompt.utils import inference

from .base import BaseMarkerizeProvider, MarkerizeResult

_HUMAN_MESSAGE_PROMPT = """
Convert the image into markdown, preserving the overall layout and style of the page. \
Use the appropriate headings for different sections. Preserve bolded and italicized text. \
Include ALL the text on the page.

You ALWAYS respond by wrapping the markdown in <md> </md> tags.
""".strip()


def ensure_single_root(xml_data: str) -> str:
    """Ensure the XML data has a single root element."""
    if not xml_data.strip().startswith("<root>") and not xml_data.strip().endswith(
        "</root>"
    ):
        return f"<root>{xml_data}</root>"
    return xml_data


def _parse_result(raw_markdown: str) -> Optional[str]:
    raw_markdown = ensure_single_root(raw_markdown)
    soup = BeautifulSoup(raw_markdown, "html.parser")

    md = soup.find("md")

    return md.text.strip() if md else ""  # TODO Fix bad extractions


def _prepare_messages(
    document_images: Iterable[bytes],
    start: Optional[int] = None,
    stop: Optional[int] = None,
):
    messages = []

    for image_bytes in document_images:
        messages.append(
            [
                OpenAIMessage(
                    role="user",
                    content=[
                        OpenAIComplexContent(
                            type="image_url",
                            image_url=OpenAIImageURL(url=image_bytes),
                        ),
                        OpenAIComplexContent(type="text", text=_HUMAN_MESSAGE_PROMPT),
                    ],
                ),
            ]
        )

    return messages


class AnthropicMarkerizeProvider(BaseMarkerizeProvider):
    name = "anthropic"

    anthropic_model_name: str = Field("claude-3-haiku-20240307")

    async def _ainvoke(
        self, input: Iterable[bytes], config: Optional[None] = None, **kwargs
    ) -> List[MarkerizeResult]:
        messages = _prepare_messages(input)

        model_name = kwargs.pop("model_name", self.anthropic_model_name)
        completions = await inference.run_batch_inference_anthropic(
            model_name, messages, **kwargs
        )

        return [
            MarkerizeResult(raw_markdown=_parse_result(x), provider_name=self.name)
            for x in completions
        ]
