from collections import defaultdict
from copy import deepcopy
from statistics import mean
from typing import List, Optional

from docprompt.schema.layout import TextBlock


def _normalize_block_edges(
    blocks: List[TextBlock], tolerance: float = 0.001, min_cluster_size=4
) -> List[TextBlock]:
    """
    A given word block may have an identical margin visually, but due to small
    variations in the OCR output, the x0 value may be slightly different.

    When building a LLM prompt, these small differences can balloon into large
    differences in the apparent structure of the text.

    Approach:

    1. Sort blocks by their x0 values, but keep track of their original indices.
    2. Cluster word blocks whose x0 values are within a given tolerance.
    3. Compute the average x0 value for each cluster.
    4. Update each word block in the cluster to have the average x0 value.
    """

    # First, create a deep copy of blocks so that we don't modify the original blocks
    blocks_copy = deepcopy(blocks)

    # 1. Sort blocks by their x0 values, but keep track of their original indices.
    indexed_blocks = [(i, block) for i, block in enumerate(blocks_copy)]
    indexed_blocks.sort(key=lambda x: x[1].bounding_box[0])

    # 2. Cluster word blocks whose x0 values are within a given tolerance.
    clusters = []
    cluster = [indexed_blocks[0]]
    for i in range(1, len(indexed_blocks)):
        prev_block = indexed_blocks[i - 1][1]
        curr_block = indexed_blocks[i][1]

        if abs(curr_block.bounding_box[0] - prev_block.bounding_box[0]) <= tolerance:
            cluster.append(indexed_blocks[i])
        else:
            clusters.append(cluster)
            cluster = [indexed_blocks[i]]
    clusters.append(cluster)  # Add the last cluster

    # 3. Compute the average x0 value for each cluster.
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            continue
        avg_x0 = sum([block[1].bounding_box[0] for block in cluster]) / len(cluster)

        # 4. Update each word block in the cluster to have the average x0 value.
        for _, block in cluster:
            block.bounding_box = (
                avg_x0,
                block.bounding_box[1],
                block.bounding_box[2],
                block.bounding_box[3],
            )

    # Return the updated list of blocks
    return blocks_copy


def word_line_clusters_from_line_blocks(
    word_blocks: List[TextBlock],
    line_blocks: List[TextBlock],
    *,
    min_line_confidence: float = 0.3,
) -> List[List[TextBlock]]:
    """
    Construct word line clusters, using line blocks as a reference point.

    Typically line blocks from an OCR provider are more accurate than our line clustering.

    If a word block has multiple line blocks intersecting it, we take the line block with the
    highest vertical overlap.
    """
    # Filter line blocks with confidence >= 0.3
    filtered_line_blocks = [
        (idx, line)
        for idx, line in enumerate(line_blocks)
        if line.confidence >= min_line_confidence
    ]

    # Dictionary to hold word blocks grouped by their intersecting line block
    line_to_words = defaultdict(list)

    # Iterate over each word block
    for word_block in word_blocks:
        best_overlap = 0
        best_line_idx = None

        # Find the line block with the highest vertical overlap
        for idx, line_block in filtered_line_blocks:
            overlap = min(
                word_block.bounding_box.bottom, line_block.bounding_box.bottom
            ) - max(word_block.bounding_box.top, line_block.bounding_box.top)
            if overlap > best_overlap:
                best_overlap = overlap
                best_line_idx = idx

        # If a best line was found, add the word to that line's group
        if best_line_idx is not None:
            line_to_words[best_line_idx].append(word_block)

    # Convert the dictionary to a list of lists for the output format
    return sorted(line_to_words.values(), key=lambda x: x[0].bounding_box.top)


def cluster_words_into_lines(
    word_blocks: List[TextBlock], minimum_y_overlap_threshold: float = 0.5
) -> List[List[TextBlock]]:
    """
    Cluster word blocks into lines using a greedy approach.

    Args:
    word_blocks (List[TextBlock]): List of word blocks to be clustered.
    minimum_y_overlap_threshold (float): Minimum vertical overlap required to consider two words in the same line.

    Returns:
    List[List[TextBlock]]: List of lines, where each line is a list of word blocks.
    """
    # Sort word blocks by their top coordinate first, then by their left coordinate
    sorted_words = sorted(
        word_blocks, key=lambda w: (w.bounding_box.top, w.bounding_box.left)
    )

    lines = []
    current_line = []

    for word in sorted_words:
        if not current_line:
            current_line.append(word)
        else:
            last_word = current_line[-1]

            # Calculate vertical overlap
            overlap = min(
                word.bounding_box.bottom, last_word.bounding_box.bottom
            ) - max(word.bounding_box.top, last_word.bounding_box.top)

            total_height = max(
                word.bounding_box.bottom, last_word.bounding_box.bottom
            ) - min(word.bounding_box.top, last_word.bounding_box.top)

            overlap_ratio = overlap / total_height

            if overlap_ratio >= minimum_y_overlap_threshold:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]

    # Add the last line if it's not empty
    if current_line:
        lines.append(current_line)

    # Sort words within each line by their left coordinate
    for line in lines:
        line.sort(key=lambda w: w.bounding_box.left)

    return lines


def _get_average_char_width(blocks: List[TextBlock]) -> float:
    """
    Returns the average character width in the text blocks
    """
    if not blocks:
        return 0.0

    widths = []

    for block in blocks:
        if block["type"] != "word":
            raise ValueError("Only words are supported")

        block_length = len(block["text"])
        block_width = block.bounding_box[2] - block.bounding_box[0]

        average_char_width = block_width / block_length

        widths.append(average_char_width)

    return sum(widths) / len(widths)


def _construct_word_line(
    word_blocks: List[TextBlock],
    x_density: Optional[float] = None,
    include_extra_space: bool = False,
    x_shift: float = 0.0,
) -> str:
    """
    Construct a line of text from a list of word blocks all
    assumed to be on the same line

    * max_line_char_count: the maximum number of characters allowed in the line
    * average_char_width: the average width of a character in the text
    * include_extra_space: whether to include extra space between words
    * x_shift: the amount to shift the initial word to the left
    """
    line = ""

    # Important for determining the number of spaces to add
    x_density = x_density or _get_average_char_width(word_blocks)

    min_spaces = 1 if include_extra_space else 0

    for idx, word in enumerate(word_blocks):
        cleaned_word = word["text"].replace("\n", " ")

        left = word.bounding_box.x0 - x_shift

        x_dist = left / x_density

        num_spaces = max(min(min_spaces, len(line)), round(x_dist) - len(line))

        line += " " * num_spaces + cleaned_word

    return line


def build_layout_aware_page_representation(
    word_blocks: List[TextBlock],
    regularize_left_margin: bool = False,
    x_density: float = 0.01,
    do_left_shift: bool = True,
    include_extra_space: bool = False,
    line_blocks: Optional[List[TextBlock]] = None,
):
    """
    Builds a layout-aware representation of the document text
    for use in a prompt in large language models.

    Requires the width and height of the image to be provided
    in order to normalize the bounding boxes.

    Optionally accepts line blocks. Some OCR providers give
    much better line segmentation than others then what can
    be derived with simple clustering of word blocks.

    * line_blocks: a list of line blocks. If provided, we instead build the line
    clusters on vertical union of word blocks and line blocks. This may lead
    to more accurate image representations
    """

    word_blocks = [
        block
        for block in word_blocks
        if block.direction == "UP" or block.direction is None
    ]

    if not word_blocks:
        return ""

    if regularize_left_margin:
        word_blocks = _normalize_block_edges(word_blocks)

    prompt_lines = []

    if line_blocks:
        line_clusters = word_line_clusters_from_line_blocks(word_blocks, line_blocks)
    else:
        line_clusters = cluster_words_into_lines(word_blocks)

    line_heights = []
    for line_words in line_clusters:
        if line_words:  # Check if the line_words list is not empty
            line_height = mean(
                [
                    abs(word.bounding_box.bottom - word.bounding_box.top)
                    for word in line_words
                ]
            )
            line_heights.append(line_height)
        else:
            line_heights.append(0.0)

    average_line_height = mean(line_heights) or 0.02

    # A page with sparser horizontal density will need more padding

    if do_left_shift:
        x_shift = min(block.bounding_box.x0 for block in word_blocks)
    else:
        x_shift = 0.0

    top_line = 0.0

    for i, line_blocks in enumerate(line_clusters):
        cluster_y = mean([x.bounding_box.top for x in line_blocks])

        # First compute the number of newlines to add
        distance_from_top = cluster_y - top_line
        # Update to consider y_density
        line_distance = int(round(distance_from_top / average_line_height))

        # Update to consider y_density
        num_newlines = 0 if i == 0 else max(min(line_distance, 4), 1)

        # Add the newlines
        prompt_lines.extend(["\n"] * num_newlines)

        # Construct the line
        word_line = _construct_word_line(
            line_blocks,
            x_density=x_density,
            include_extra_space=include_extra_space,
            x_shift=x_shift,
        )
        prompt_lines.append(word_line)

        # Update the top line
        top_line = mean([x.bounding_box.bottom for x in line_blocks])

    return "".join(prompt_lines)
