from typing import Optional

from docprompt.schema.pipeline.node.document import DocumentNode
from docprompt.tasks.base import AbstractPageTaskProvider, BasePageResult

from ..capabilities import PageLevelCapabilities


class MarkerizeResult(BasePageResult):
    task_name = "markerize"
    raw_markdown: str


class BaseMarkerizeProvider(AbstractPageTaskProvider[bytes, None, MarkerizeResult]):
    capabilities = [PageLevelCapabilities.PAGE_MARKERIZATION]

    class Meta:
        abstract = True

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

        # TODO: This is a somewhat dangerous way of requiring these kwargs to be drilled
        # through, potentially a decorator solution to be had here
        kwargs = {**self._default_invoke_kwargs, **kwargs}
        results = self._invoke(raster_bytes, config=task_config, **kwargs)

        return {
            i: res
            for i, res in zip(
                range(start or 1, (stop or len(document_node)) + 1), results
            )
        }
