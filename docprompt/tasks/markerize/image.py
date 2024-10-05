import asyncio
import base64
from typing import ClassVar, Iterable, List, Optional

from bs4 import BeautifulSoup

from docprompt.tasks.message import OpenAIComplexContent, OpenAIImageURL, OpenAIMessage

from .base import BaseLLMMarkerizeProvider, MarkerizeConfig, MarkerizeResult

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


def _image_bytes_to_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _prepare_messages(
    document_images: Iterable[bytes],
    config: MarkerizeConfig,
):
    messages = []

    human_message = config.human_prompt or _HUMAN_MESSAGE_PROMPT

    for image_bytes in document_images:
        messages.append(
            [
                OpenAIMessage(
                    role="user",
                    content=[
                        OpenAIComplexContent(
                            type="image_url",
                            image_url=OpenAIImageURL(
                                url=_image_bytes_to_url(image_bytes)
                            ),
                        ),
                        OpenAIComplexContent(type="text", text=human_message),
                    ],
                ).model_dump()
            ]
        )

    return messages


class GenericMarkerizeProvider(BaseLLMMarkerizeProvider):
    name: ClassVar[str] = "markerize"

    def model_post_init(self, __context):
        self.task_config = self.task_config or self._get_default_markerize_config()

        if not self.async_callable:

            async def async_callable(messages):
                result = await asyncio.to_thread(self.sync_callable, messages)

                return result

            self.async_callable = async_callable

    def _get_default_markerize_config(self):
        return MarkerizeConfig(
            human_prompt=_HUMAN_MESSAGE_PROMPT,
        )

    async def ainvoke(
        self, input: Iterable[bytes], config: Optional[MarkerizeConfig] = None, **kwargs
    ) -> List[MarkerizeResult]:
        config = config or self._get_default_markerize_config()

        messages = self.get_openai_messages(input, config=config)

        coroutines = [self.async_callable(x) for x in messages]

        result = await asyncio.gather(*coroutines)

        final = []

        for x in result:
            text = x["choices"][0]["message"]["content"]
            parsed = self.parse(text)
            final.append(parsed)

        return final

    def invoke(
        self, input: Iterable[bytes], config: Optional[MarkerizeConfig] = None, **kwargs
    ) -> List[MarkerizeResult]:
        config = config or self._get_default_markerize_config()

        result = asyncio.run(self.ainvoke(input, config, **kwargs))

        return result

    def get_openai_messages(
        self, input: Iterable[bytes], config: Optional[MarkerizeConfig] = None, **kwargs
    ):
        return _prepare_messages(
            input, config or self.task_config or self._get_default_markerize_config()
        )

    def parse(self, response: str):
        return MarkerizeResult(
            raw_markdown=_parse_result(response), provider_name=self.name
        )

    async def aparse(self, response: str, **kwargs) -> MarkerizeResult:
        return self.parse(response)
