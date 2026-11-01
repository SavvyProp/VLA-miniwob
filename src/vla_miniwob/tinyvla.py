"""Optional bridge to the pinned TinyVLA processor without reimplementing its VLM."""

import copy


class SingleViewProcessor:
    """Wrap an upstream LlavaPythiaProcess instance for full-history, single-view GUI data.

    Use with TinyVLADataset(..., processor=SingleViewProcessor(upstream, image_tokens)).
    image_tokens must be the actual post-projector visual token count for the
    selected image pipeline, not an assumed language-only context allowance.
    Requires the pinned TinyVLA packages in a separate training environment.
    """

    def __init__(self, upstream, image_tokens: int):
        if image_tokens < 1:
            raise ValueError("The actual multimodal image token count is required")
        self.upstream = upstream
        self.image_tokens = image_tokens

    def __call__(self, sample):
        from data_utils.processor import preprocess, preprocess_multimodal

        upstream = self.upstream
        source = upstream.datastruct_droid2llava(sample)
        conversations = preprocess_multimodal(
            copy.deepcopy([source["conversations"]]), upstream.data_args
        )
        processed = preprocess(conversations, upstream.tokenizer, has_image=True)
        ids = processed["input_ids"][0]
        # Upstream may truncate at model_max_length. Even a full-length truncated
        # result fails this stricter check once visual tokens are included.
        total = len(ids) + self.image_tokens - 1
        if total > upstream.tokenizer.model_max_length:
            raise ValueError(f"Full GUI context exceeds capacity ({total} multimodal tokens)")
        images = upstream.parse_image(sample["image"])
        if len(images) != 1:
            raise ValueError("GUI pilot requires exactly one current screenshot")
        return {
            "input_ids": ids,
            "labels": processed["labels"][0],
            "image": images[:1],
            "state": sample["state"],
            "action": sample["action"],
            "is_pad": sample["is_pad"],
        }
