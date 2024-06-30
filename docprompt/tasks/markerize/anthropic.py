from typing import Iterable, List, Optional

from bs4 import BeautifulSoup

from docprompt.schema.pipeline.node.document import DocumentNode
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


async def _prepare_messages(
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

    def _invoke(
        self, input: Iterable[bytes], config: Optional[None] = None
    ) -> List[MarkerizeResult]:
        messages = _prepare_messages(input)

        completions = inference.run_batch_inference_anthropic(messages)

        return [_parse_result(x) for x in completions]

    def process_document_node(
        self,
        document_node: "DocumentNode",
        task_config: Optional[None] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ):
        raster_bytes = []
        for page_number in range(start or 1, (stop or len(document_node)) + 1):
            image_bytes = document_node.page_nodes[
                page_number - 1
            ].rasterizer.rasterize("default")
            raster_bytes.append(image_bytes)

        results = self._invoke(raster_bytes, config=task_config, **kwargs)

        return {
            i: MarkerizeResult(provider_name=self.name, raw_markdown=x)
            for i, x in zip(
                range(start or 1, (stop or len(document_node)) + 1), results
            )
        }
