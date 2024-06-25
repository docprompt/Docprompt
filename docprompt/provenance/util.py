import re
from collections import defaultdict
from typing import Any, Iterable, List, Optional

from rapidfuzz import fuzz
from rapidfuzz.utils import default_process

from docprompt.schema.layout import NormBBox, TextBlock

try:
    import tantivy
except ImportError:
    raise ImportError("Could not import tantivy. Install with `docprompt[search]`")

try:
    import networkx
except ImportError:
    raise ImportError("Could not import networkx. Install with `docprompt[search]`")


_prefix_regexs = [
    re.compile(r"^\d+\.\s+"),
    re.compile(r"^\d+\.\d+\s+"),
    re.compile(r"^\*+\s+"),
    re.compile(r"^-+\s+"),
]


def preprocess_query_text(text: str) -> str:
    """
    Improve matching ability by applying some preprocessing to the query text.
    """
    for regex in _prefix_regexs:
        text = regex.sub("", text)

    text = text.strip()

    text = text.replace('"', "")

    return text


def word_tokenize(text: str) -> List[str]:
    """
    Tokenize a string into words.
    """
    return re.split(r"\s+", text)


def create_tantivy_document_wise_block_index():
    schema_builder = tantivy.SchemaBuilder()

    schema_builder.add_integer_field(
        "page_number", stored=True, indexed=True, fast=True
    )
    schema_builder.add_text_field("block_type", stored=True)
    schema_builder.add_integer_field("block_page_idx", stored=True)
    schema_builder.add_text_field("content", stored=True)

    schema = schema_builder.build()

    index = tantivy.Index(schema=schema)

    return index


def construct_valid_rtree_tuple(bbox: NormBBox):
    # For some reason sometimes the bounding box is invalid (top > bottom, x0 > x1
    # This function is to ensure that the bounding box is valid for the rtree index

    true_top = min(bbox.top, bbox.bottom)
    true_bottom = max(bbox.top, bbox.bottom)

    true_x0 = min(bbox.x0, bbox.x1)
    true_x1 = max(bbox.x0, bbox.x1)

    return (true_x0, true_top, true_x1, true_bottom)


def insert_generator(bboxes: List[NormBBox], data: Optional[Iterable[Any]] = None):
    """
    Make an iterator that yields tuples of (id, bbox, data) for insertion into an RTree index
    which improves performance massively.
    """
    data = data or [None] * len(bboxes)

    for idx, (bbox, data_item) in enumerate(zip(bboxes, data)):
        yield (idx, construct_valid_rtree_tuple(bbox), data_item)


def refine_block_to_word_level(
    source_block: TextBlock,
    intersecting_word_level_blocks: List[TextBlock],
    query: str,
):
    """
    Create a new text block by merging the intersecting word level blocks that
    match the query.

    """
    intersecting_word_level_blocks.sort(
        key=lambda x: (x.bounding_box.top, x.bounding_box.x0)
    )

    tokenized_query = word_tokenize(query)

    if len(tokenized_query) == 1:
        fuzzified = default_process(tokenized_query[0])
        for word_level_block in intersecting_word_level_blocks:
            if fuzz.ratio(fuzzified, default_process(word_level_block.text)) > 87.5:
                return word_level_block, [word_level_block]
    else:
        fuzzified_word_level_texts = [
            default_process(word_level_block.text)
            for word_level_block in intersecting_word_level_blocks
        ]

        # Populate the block mapping
        token_block_mapping = defaultdict(set)

        first_word = tokenized_query[0]
        last_word = tokenized_query[-1]

        for token in tokenized_query:
            fuzzified_token = default_process(token)
            for i, word_level_block in enumerate(intersecting_word_level_blocks):
                if fuzz.ratio(fuzzified_token, fuzzified_word_level_texts[i]) > 87.5:
                    token_block_mapping[token].add(i)

        graph = networkx.DiGraph()
        prev = tokenized_query[0]

        for i in token_block_mapping[prev]:
            graph.add_node(i)

        for token in tokenized_query[1:]:
            for prev_block in token_block_mapping[prev]:
                for block in sorted(token_block_mapping[token]):
                    if block > prev_block:
                        weight = (
                            (block - prev_block) ** 2
                        )  # Square the distance to penalize large jumps, which encourages reading order
                        graph.add_edge(prev_block, block, weight=weight)

            prev = token

        # Get every combination of first and last word
        first_word_blocks = token_block_mapping[first_word]
        last_word_blocks = token_block_mapping[last_word]

        combinations = sorted(
            [(x, y) for x in first_word_blocks for y in last_word_blocks if x < y],
            key=lambda x: abs(x[1] - x[0]),
        )

        for start, end in combinations:
            try:
                path = networkx.shortest_path(graph, start, end, weight="weight")
            except networkx.NetworkXNoPath:
                continue
            except Exception:
                continue

            matching_blocks = [intersecting_word_level_blocks[i] for i in path]

            merged_bbox = NormBBox.combine(
                *[word_level_block.bounding_box for word_level_block in matching_blocks]
            )

            merged_text = ""

            for word_level_block in matching_blocks:
                merged_text += word_level_block.text
                if not word_level_block.text.endswith(" "):
                    merged_text += " "  # Ensure there is a space between words

            return (
                TextBlock(
                    text=merged_text,
                    type="block",
                    bounding_box=merged_bbox,
                    metadata=source_block.metadata,
                ),
                matching_blocks,
            )
