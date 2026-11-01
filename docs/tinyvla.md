# Train and evaluate TinyVLA-B

The trainer fine-tunes the pinned `lesjie/Llava-Pythia-700M` backbone using
TinyVLA's native conditional diffusion U-Net and DDIM scheduler. The released
backbone is not already a GUI policy. This integration has CPU regression tests;
full pretrained GPU training and its benchmark score have not been measured here.

Run commands from this repository's root on the machine containing your collected
examples. No OpenAI key is needed for student training or evaluation.

## Install the separate uv environment

```bash
bash scripts/setup_tinyvla.sh cuda
```

This installs pinned CUDA 12.4 PyTorch and checks native imports and GPU operation.
The first training command downloads the pinned backbone. Reserve disk space for
the runtime, backbone, exported images, and optimizer/model checkpoints. See
[runtime setup](tinyvla_setup.md) for CPU diagnostics and dependency details.

## Train from collected examples

```bash
uv run --project integrations/tinyvla --frozen --extra cuda \
  vla-benchmark train-tinyvla runs/astra-train runs/tinyvla-b --max-steps 2000
```

The source can also be an existing `exports/train` directory. A raw collection is
exported automatically into `runs/tinyvla-b/data/train`; only accepted successful
training episodes are used. The original collection is not modified. New exports
record task/complexity/order/wording metadata for balanced transition sampling.
Legacy exports without it use a recorded uniform-transition fallback; re-export
from raw collection to obtain balanced sampling with complete provenance.

The default [configuration](../configs/tinyvla_b.json) targets a single RTX 3090:
batch size 1, 8 accumulation steps (effective batch size 8), BF16 computation,
FP32 trainable weights/AdamW state, and nonreentrant gradient checkpointing. This
is a starting configuration, not a measured memory/performance guarantee. Both
language and vision transformers receive LoRA adapters (rank 64, alpha 256);
original backbone weights stay frozen. The projector and diffusion decoder train
in full. Adapter LR is `2e-4`, other trainable LR `2e-5`, with 100 warmup updates
followed by constant LR. A step means one optimizer update.

For a download, context and image-pipeline check without optimization:

```bash
uv run --project integrations/tinyvla --frozen --extra cuda \
  vla-benchmark train-tinyvla runs/astra-train runs/tinyvla-b --preflight-only

# Continue that verified preflight, or resume an interrupted training run:
uv run --project integrations/tinyvla --frozen --extra cuda \
  vla-benchmark train-tinyvla runs/astra-train runs/tinyvla-b \
  --max-steps 2000 --resume
```

Inspect `train-vision-*.png` in the output directory. These show the actual resized
and padded vision input, with display normalization undone. The pinned CLIP-L/14
path produces 576 patch tokens at 336×336. Small GUI labels in multi-panel tasks
can become difficult to read at this resolution; code cannot establish
readability automatically. `preflight.json` records actual token counts, longest
full context, and every trainable parameter name/count. All contexts are checked
against combined image/text capacity before optimization; history is never
silently truncated. Rollout context overflow counts as a model failure.

Checkpoints are published atomically every 250 updates and at the final step.
Resume restores model, optimizer, sampler, and random states from the latest
complete checkpoint; updates since that checkpoint are repeated. The same inputs,
code, runtime, device, precision, and settings are required. Only `--max-steps`
may increase. Use a new output directory for hyperparameter changes. A checkpoint
includes all trainable weights, architecture/tokenizer/processor, normalization,
checksums, provenance and optimizer/RNG state. Frozen pinned backbone weights
are loaded from Hugging Face/cache.

The directory contains:

- `training_run.json`: recipe, data/code/runtime fingerprints and sampler.
- `metrics.jsonl`: epsilon loss, gradient norm, LRs, timing, optional dev loss.
- `preflight.json` and vision previews: context and trainable-parameter audits.
- `checkpoint-0000250/`, etc.: complete inference and resume checkpoints.
- `latest.json`: convenience pointer; complete checkpoint directories are authoritative.

## Optional development demonstrations

Training works with training demonstrations alone. If you separately collected an
Astra reference on the manifest's **dev** split, use it to monitor held-out loss:

```bash
uv run --project integrations/tinyvla --frozen --extra cuda \
  vla-benchmark train-tinyvla runs/astra-train runs/tinyvla-b-with-dev \
  --dev-source runs/astra-dev --max-steps 2000
```

Only dev exports/runs are accepted. Dev normalization must exactly match training;
episode IDs, value seeds and layout seeds must be disjoint. Legacy exports lacking
seeds must be re-exported from raw runs. There is no automatic random split or use
of evaluation demonstrations. Dev loss is an imitation diagnostic, not autonomous
success; at most 32 sequential batches are checked by default and the count is
logged. Choose checkpoints using dev rollouts, then freeze the choice for eval.

## Autonomous browser evaluation

```bash
# Specify a checkpoint when comparing experiments.
uv run --project integrations/tinyvla --frozen --extra cuda \
  vla-benchmark evaluate-tinyvla runs/tinyvla-b/checkpoint-0002000 \
  configs/pilot.json runs/tinyvla-b-dev --split dev

uv run --project integrations/tinyvla --frozen --extra cuda \
  vla-benchmark evaluate-tinyvla runs/tinyvla-b/checkpoint-0002000 \
  configs/pilot.json runs/tinyvla-b-eval --split eval

uv run --frozen vla-benchmark report runs/tinyvla-b-eval
```

Passing `runs/tinyvla-b` as checkpoint resolves the latest complete checkpoint.
Evaluation runs **all** instances in the selected split by default (38 in the
pilot). `--max-episodes 2` bounds a smoke run. Repeat the same command/output to
resume unfinished evaluation; changing checkpoint or settings requires a new
evaluation directory. Chrome/ChromeDriver setup is the same as collection.

The adapter sees only the current screenshot, normalized observable state and
full procedure/history. It generates one H=16 chunk with 10 native DDIM steps,
inverts saved training normalization, and returns numerical mouse coordinates
and click values. The harness executes K=1 then observes again. There is no
target snapping, privileged completion signal, success-based resampling, robot
action conversion or temporal aggregation. Invalid predictions count as model
failures. Existing reports retain strict chronological success, task/complexity
breakdowns, uncertainty and incomplete-coverage handling.

For the generic plugin API, use `vla_miniwob.tinyvla_policy:TinyVLAPolicy` with a
JSON config containing `checkpoint`, optionally `device` and `dtype`.

## Implementation and validation

The integration reuses upstream VLM, LoRA layers, convolutional decoder and DDIM
scheduler. It replaces robot-specific data loading and two-camera processing.
Masked epsilon MSE is normalized by valid action positions separately per example.
Language padding is excluded from mean-pooled conditioning, and padded actions
cannot carry hidden target information. All adapter/projector/decoder tensors
are saved while frozen pinned weights are reused.

```bash
# CPU tests do not fetch pretrained backbone weights.
uv run --project integrations/tinyvla --frozen --extra cpu pytest
```

Tests exercise native upstream transformer/vision/U-Net code at reduced dimensions,
gradient flow, masks, context overflow, batch padding, exact checkpoint inference
round trips, optimizer/RNG resume, split leakage, normalization, policy action
conversion and checkpoint corruption. They establish implementation mechanics,
not task competence or GPU memory fit. The CPU command switches this optional
environment's Torch build; rerun CUDA setup before GPU training.

Sources: [pinned TinyVLA source](https://github.com/liyaxuanliyaxuan/TinyVLA/tree/94f441827b45e4f76316ef6a0ae443736dc93a5d),
[pinned backbone](https://huggingface.co/lesjie/Llava-Pythia-700M/tree/9ac93b5370af5a0b6e66ffa2b60e0316489826d5).
