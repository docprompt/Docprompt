from functools import lru_cache

from more_itertools import chunked

from .schema import LayoutLMV3BaseInput

try:
    import numpy as np
    import transformers as trf
except ImportError:
    print("Please install torch and transformers and numpy to use the LayoutLMv3EmbeddingProvider")


@lru_cache(maxsize=1)
def get_layoutlmv3_processor(path: str = "microsoft/layoutlmv3-base", **kwargs):
    return trf.LayoutLMv3Processor.from_pretrained(path, apply_ocr=False, **kwargs)


@lru_cache(maxsize=1)
def get_layoutlmv3_model(path: str = "microsoft/layoutlmv3-base", **kwargs):
    return trf.LayoutLMv3Model.from_pretrained(path, **kwargs)


def get_layoutlmv3_hidden_states(lm_inputs: list[LayoutLMV3BaseInput], device="cuda", batch_size=20, **kwargs):
    processor = get_layoutlmv3_processor()
    model = get_layoutlmv3_model().to(device)

    output_embeddings = []

    for chunked_inputs in chunked(lm_inputs, batch_size):
        encoding = processor(
            [lm_input.image for lm_input in chunked_inputs],
            [lm_input.tokens for lm_input in chunked_inputs],
            boxes=[lm_input.bboxes for lm_input in chunked_inputs],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        import torch

        with torch.no_grad():
            outputs = model(**encoding)
            last_hidden_states = outputs.last_hidden_state

            attention_masks = encoding["attention_mask"]
            sequence_lengths = attention_masks.sum(dim=1)

            embeddings = []
            for i, seq_len in enumerate(sequence_lengths):
                # If sequence length is less than 2, use [CLS] embedding
                if seq_len <= 2:
                    embeddings.append(last_hidden_states[i, 0].cpu().numpy())
                else:
                    attention_mask = attention_masks[i]
                    attention_mask[0] = 0  # Remove [CLS] token
                    attention_mask[seq_len.item() - 1] = 0  # Remove [SEP] token

                    # This gets all the tokens, excluding padding
                    tokens = last_hidden_states[i, : attention_masks[i].size(0)][attention_masks[i].bool()]

                    # Simple mean pooling
                    embedding = torch.mean(tokens, axis=0).cpu().numpy()

                    embeddings.append(embedding)

            output_embeddings.extend(embeddings)

    return np.array(output_embeddings)
