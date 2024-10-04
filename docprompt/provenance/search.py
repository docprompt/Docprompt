from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

from docprompt.schema.layout import NormBBox, TextBlock
from docprompt.tasks.ocr.result import OcrPageResult

from .source import PageTextLocation, ProvenanceSource
from .util import (
    construct_valid_rtree_tuple,
    create_tantivy_document_wise_block_index,
    insert_generator,
    preprocess_query_text,
    refine_block_to_word_level,
)

try:
    import tantivy
    from rtree.index import Index as RTreeIndex
except ImportError:
    raise ImportError(
        "Could not import tantivy and/or rtree. Install with `docprompt[search]`"
    )


if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode


SearchBestModes = Literal["shortest_text", "longest_text", "highest_score"]
BlockGranularity = Literal["word", "line", "block"]
DocumentProvenanceGeoMap = Dict[int, Dict[BlockGranularity, RTreeIndex]]


@dataclass
class DocumentProvenanceLocator:
    document_name: str
    search_index: "tantivy.Index"
    block_mapping: Dict[int, OcrPageResult] = field(repr=False)
    geo_index: DocumentProvenanceGeoMap = field(repr=False)

    @classmethod
    def from_document_node(cls, document_node: "DocumentNode"):
        index = create_tantivy_document_wise_block_index()
        block_mapping_dict = {}
        geo_index_dict: DocumentProvenanceGeoMap = {}

        writer = index.writer()

        for page_node in document_node.page_nodes:
            ocr_result = page_node.ocr_results

            if ocr_result is None:
                raise ValueError(
                    "Page {} does not have OCR results".format(page_node.page_number)
                )

            for idx, text_block in enumerate(ocr_result.block_level_blocks):
                writer.add_document(
                    tantivy.Document(
                        page_number=page_node.page_number,
                        block_type=text_block.type,
                        block_page_idx=idx,
                        content=text_block.text,
                    )
                )

            for granularity in ["word", "line", "block"]:
                text_blocks = getattr(ocr_result, f"{granularity}_level_blocks", [])

                bounding_boxes = [text_block.bounding_box for text_block in text_blocks]

                if bounding_boxes:
                    r_tree = RTreeIndex(
                        insert_generator(bounding_boxes), fill_factor=0.9
                    )
                else:
                    r_tree = RTreeIndex()

                if page_node.page_number not in geo_index_dict:
                    geo_index_dict[page_node.page_number] = {}

                geo_index_dict[page_node.page_number][granularity] = r_tree  # type: ignore

            block_mapping_dict[page_node.page_number] = ocr_result

        writer.commit()
        index.reload()

        return cls(
            document_name=document_node.document.name,
            search_index=index,
            block_mapping=block_mapping_dict,
            geo_index=geo_index_dict,
        )

    def _construct_tantivy_query(
        self, query: str, page_number: Optional[int] = None
    ) -> tantivy.Query:
        query = preprocess_query_text(query)

        if page_number is None:
            return self.search_index.parse_query(f'content:"{query}"')
        else:
            return self.search_index.parse_query(
                f'(page_number:{page_number}) AND content:"{query}"'
            )

    def get_k_nearest_blocks(
        self,
        bbox: NormBBox,
        page_number: int,
        k: int,
        granularity: BlockGranularity = "block",
    ) -> List[TextBlock]:
        """
        Get the k nearest text blocks to a given bounding box
        """
        search_tuple = construct_valid_rtree_tuple(bbox)

        word_level_bbox_indices = list(
            self.geo_index[page_number][granularity].nearest(
                search_tuple, num_results=k
            )
        )

        block_mapping = self.block_mapping[page_number]

        nearest_blocks = [
            getattr(block_mapping, granularity + "s")[idx]
            for idx in word_level_bbox_indices
        ]

        nearest_blocks.sort(key=lambda x: (x.bounding_box.top, x.bounding_box.x0))

        return [x for x in nearest_blocks if x.bounding_box != bbox]

    def get_overlapping_blocks(
        self, bbox: NormBBox, page_number: int, granularity: BlockGranularity = "block"
    ) -> List[TextBlock]:
        """
        Get the text blocks that overlap with a given bounding box
        """
        search_tuple = construct_valid_rtree_tuple(bbox)

        bbox_indices = list(
            self.geo_index[page_number][granularity].intersection(search_tuple)
        )

        block_mapping = self.block_mapping[page_number]

        overlapping_blocks = [
            getattr(block_mapping, f"{granularity}_level_blocks")[idx]
            for idx in bbox_indices
        ]

        overlapping_blocks.sort(key=lambda x: (x.bounding_box.top, x.bounding_box.x0))

        return [x for x in overlapping_blocks if x.bounding_box != bbox]

    def search_raw(self, raw_query: str) -> List[str]:
        """
        Search for a piece of text using a raw query

        Args:
            query: The text to search for
            page_number: The page number to search on
        """
        parsed_query = self.search_index.parse_query(raw_query)

        searcher = self.search_index.searcher()

        search_results = searcher.search(parsed_query, limit=100)

        results = []

        for score, doc_address in search_results.hits:
            doc = searcher.doc(doc_address)

            result_page_number = doc["page_number"][0]
            result_block_page_idx = doc["block_page_idx"][0]
            block_mapping = self.block_mapping[result_page_number]

            source_block: TextBlock = block_mapping.block_level_blocks[
                result_block_page_idx
            ]

            results.append(source_block.text)

        return results

    def refine_query_to_word_level(
        self, query: str, page_number: int, enclosing_block: TextBlock
    ):
        """
        Refine a query to the word level
        """
        search_tuple = construct_valid_rtree_tuple(enclosing_block.bounding_box)

        word_level_bbox_indices = list(
            self.geo_index[page_number]["word"].intersection(search_tuple)
        )
        word_level_blocks_in_original_bbox = [
            self.block_mapping[page_number].word_level_blocks[idx]
            for idx in word_level_bbox_indices
        ]

        refine_result = refine_block_to_word_level(
            source_block=enclosing_block,
            intersecting_word_level_blocks=word_level_blocks_in_original_bbox,
            query=query,
        )

        return refine_result

    def search(
        self,
        query: str,
        page_number: Optional[int] = None,
        *,
        refine_to_word: bool = True,
        require_exact_match: bool = True,
    ) -> List[ProvenanceSource]:
        """
        Search for a piece of text in the document and return the source of it

        Args:
            query: The text to search for
            page_number: The page number to search on
            refine_to_word: Whether to refine the search to the word level
            require_exact_match: Whether to require null results if `refine_to_word` is True and no exact match is found
        """
        search_query = self._construct_tantivy_query(query, page_number)

        searcher = self.search_index.searcher()

        search_results = searcher.search(search_query, limit=100)

        results = []

        for score, doc_address in search_results.hits:
            doc = searcher.doc(doc_address)

            result_page_number = doc["page_number"][0]
            result_block_page_idx = doc["block_page_idx"][0]
            block_mapping = self.block_mapping[result_page_number]

            source_block: TextBlock = block_mapping.block_level_blocks[
                result_block_page_idx
            ]

            source_blocks = [source_block]
            principal_block = source_block

            if refine_to_word:
                refine_result = self.refine_query_to_word_level(
                    query=query,
                    page_number=result_page_number,
                    enclosing_block=source_block,
                )

                if refine_result is not None:
                    principal_block, source_blocks = refine_result
                elif require_exact_match:
                    continue

            source = ProvenanceSource(
                document_name=self.document_name,
                page_number=result_page_number,
                text_location=PageTextLocation(
                    source_blocks=source_blocks,
                    text=query,
                    score=score,
                    granularity="block",
                    merged_source_block=principal_block,
                ),
            )
            results.append(source)

        results.sort(key=lambda x: x.page_number)

        return results

    def search_n_best(
        self, query: str, n: int = 3, mode: SearchBestModes = "shortest_text"
    ) -> List[ProvenanceSource]:
        results = self.search(query)

        if not results:
            return []

        if mode == "shortest_text":
            score_func = lambda x: len(x.source_block.text)  # noqa: E731
        elif mode == "longest_text":
            score_func = lambda x: -len(x[0].source_block.text)  # noqa: E731
        elif mode == "highest_score":
            score_func = lambda x: x[1]  # noqa: E731
        else:
            raise ValueError(f"Unknown mode {mode}")

        results.sort(key=score_func)

        return results[:n]
