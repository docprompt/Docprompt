from typing import Dict, Optional

from bs4 import BeautifulSoup

from docprompt.schema.document import Document
from docprompt.tasks.message import OpenAIComplexContent, OpenAIImageURL, OpenAIMessage
from docprompt.utils import inference

from .base import BaseMarkerizeProvider, MarkerizeResult

_HUMAN_MESSAGE_PROMPT = """
Convert the image into markdown, preserving the overall layout and style of the page. \
Use the appropriate headings for different sections. Preserve bolded and italicized text. \
Include ALL the text on the page.

You ALWAYS respond by wrapping the markdown in <md> </md> tags.
""".strip()


def _parse_result(raw_markdown: str) -> Optional[str]:
    soup = BeautifulSoup(raw_markdown, "xml")

    md = soup.find("md")

    return md.text.strip() if md else ""  # TODO Fix bad extractions


async def _prepare_message(
    document: Document, start: Optional[int] = None, stop: Optional[int] = None
):
    messages = []

    for page_number in range(start or 1, (stop or len(document)) + 1):
        rastered_page = document.rasterize_page_to_data_uri(page_number)

        messages.append(
            [
                OpenAIMessage(
                    role="user",
                    content=[
                        OpenAIComplexContent(
                            type="image_url",
                            image_url=OpenAIImageURL(url=rastered_page),
                        ),
                        OpenAIComplexContent(type="text", text=_HUMAN_MESSAGE_PROMPT),
                    ],
                ),
            ]
        )

    return messages


class AnthropicMarkerizeProvider(BaseMarkerizeProvider):
    name = "anthropic"

    async def aprocess_document_pages(
        self,
        document: Document,
        task_input: Optional[None] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        model_name: str = "claude-3-haiku-20240307",
        **kwargs,
    ) -> Dict[int, MarkerizeResult]:
        messages = await _prepare_message(document, start=start, stop=stop)
        completions = await inference.run_batch_inference_anthropic(
            model_name, messages, **kwargs
        )

        parsed = [_parse_result(x) for x in completions]

        return {
            i: MarkerizeResult(provider_name=self.name, raw_markdown=x)
            for i, x in zip(range(start or 1, (stop or len(document)) + 1), parsed)
        }
