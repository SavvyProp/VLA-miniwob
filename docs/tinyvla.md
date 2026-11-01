# TinyVLA-B handoff

The initial target is TinyVLA-B, initialized from `lesjie/Llava-Pythia-700M`.
The model and source revisions are pinned in `configs/tinyvla_b.json`. These
released weights are a VLM backbone, not a GUI policy. This repository supplies
training data and integration adapters; it has not run GPU fine-tuning.

## Audited upstream behavior

At TinyVLA revision `94f441827b45e4f76316ef6a0ae443736dc93a5d`:

- `train_tinyvla.py` defaults to a `droid_diffusion` head, action dimension 10,
  state dimension 7, and chunk length 16. The GUI recipe changes both dimensions
  to 3 and keeps H=16, K=1.
- `llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py` builds a
  conditional 1D U-Net and DDIM scheduler: 100 training timesteps, cosine beta
  schedule `squaredcos_cap_v2`, epsilon prediction, clipping enabled, and 10
  inference timesteps. Retain these native semantics for the initial comparison.
- Its epsilon loss multiplies by the padding mask then averages over **all**
  padded and valid positions. This gives short trajectories less loss weight.
  Use the included `masked_diffusion_loss()` to normalize by each example's
  valid action positions before averaging examples, as the PDF specifies.
- `data_utils/datasets.py` reads one `language_raw` per episode, makes its own
  train/validation split, and can shift real-robot actions by one timestep.
  The stock processor unconditionally indexes a second image view.
- The training script declares LoRA rank 64, alpha 256, `vit llm` LoRA modules,
  frozen backbone/vision tower, and a trainable multimodal projector. Do not infer
  actual trainable parameters from those flags: inspect `requires_grad` after
  model initialization and record names/counts.
- The stock custom sampler samples transition indices uniformly. Its
  `sample_weights` setup does not establish balanced GUI task/complexity sampling.
  Choose and record a balanced sampler before a comparative learning study.

## Use the exported data

```python
from vla_miniwob.dataset import ChunkDataset, TinyVLADataset
from vla_miniwob.tinyvla import SingleViewProcessor

train_chunks = ChunkDataset("exports/train", horizon=16)
dev_chunks = ChunkDataset("exports/dev", horizon=16)

# upstream_process is TinyVLA's LlavaPythiaProcess(data_args, tokenizer).
# actual_visual_tokens must be measured for the selected backbone/processor.
processor = SingleViewProcessor(upstream_process, image_tokens=actual_visual_tokens)
train_dataset = TinyVLADataset(train_chunks, processor=processor)
dev_dataset = TinyVLADataset(dev_chunks, processor=processor)
```

The adapters produce the upstream collator's `input_ids`, `labels`, `image`,
`state`, `action`, and `is_pad` keys. They keep a single screenshot view and the
full literal history. `SingleViewProcessor` reuses upstream language/image
preprocessing and fails when the combined image/text length exceeds capacity.
It must be used instead of the original two-camera forward method. Development
exports must reuse `exports/train/normalization.json`.

The base `ChunkDataset` works without Torch. To install the optional Torch-facing
bridge into this project's uv environment, use `uv sync --frozen --extra torch`.
For actual TinyVLA training, create a **separate uv environment** compatible with
the pinned upstream requirements, install the upstream `policy_heads` and
`llava-pythia` packages there, and install this repository as an editable package.
Do not merge incompatible model-library pins into the browser collection runtime.
The GPU environment has not been resolved or validated in this implementation.

## Training integration still required

Adapt the upstream training entry point to accept the two datasets above rather
than calling its random-splitting `load_data()`. Preserve the frozen split,
single-camera observations, and per-timestep language. Replace the masked loss
reduction, use action/state dimensions 3, and ensure the diffusion decoder and
necessary projections are trainable. The project does not provide a misleading
one-command training launcher before this integration is verified.

Before training, inspect the longest procedure/history after the actual
multimodal formatter and inspect screenshots after the actual vision processor.
The exported 720×540 image is readable at its original size; the stock 320-pixel
vision path may lose the small native text. The 2048-token dataclass default is
not an allowance for 2048 procedure tokens in addition to visual tokens.

For each run record the backbone revision, actual trainable names/counts, LoRA
settings, optimizer, learning rates, weighting/sampling policy, demonstration and
transition counts, chunk count, image pipeline, context limit, noise scheduler,
denoising steps, training seeds, and checkpoints. No joint-angle, end-effector,
or gripper conversion belongs in this GUI pipeline.

## Inference adapter

A future checkpoint adapter should provide the `Policy` protocol. Build the same
full context and single-view image preprocessing; standardize observable state;
sample exactly one native diffusion chunk; invert `Normalizer.actions()` with
`undo_actions()`; return the H×3 array. The harness handles clipping, thresholding,
K=1 execution, and the next observation. It must not see evaluator objects or
select chunks using a private success check. Disable the stock real-robot temporal
aggregation, which could average two target coordinates into an invalid click.

Sources: [TinyVLA repository](https://github.com/liyaxuanliyaxuan/TinyVLA/tree/94f441827b45e4f76316ef6a0ae443736dc93a5d),
[backbone](https://huggingface.co/lesjie/Llava-Pythia-700M/tree/9ac93b5370af5a0b6e66ffa2b60e0316489826d5).
