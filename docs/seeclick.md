# SeeClick fine-tuning and evaluation

The integration loads the pretrained SeeClick GUI-grounding checkpoint and trains
LoRA adapters with Hugging Face Trainer and PEFT. It uses the original model's
vision encoder, image-token encoding, and generation implementation. It does not
replace SeeClick with Qwen2-VL or a randomly initialized language model.

## Runtime and model pins

| Component | Pin |
| --- | --- |
| Model | `cckevinn/SeeClick`, `b79618049920e209d15cda4d35004e32e8b485d9` |
| Tokenizer and font | `Qwen/Qwen-VL-Chat`, `f57cfbd358cb56b710d963669ad1bcfb44cdcdd8` |
| Python | 3.11 |
| PyTorch / torchvision | 2.5.1 / 0.20.1 |
| Transformers | 4.37.2 |
| PEFT / Accelerate | 0.10.0 / 0.27.2 |
| bitsandbytes, CUDA extra | 0.45.5 |

The model and tokenizer use `trust_remote_code=True` with explicit commit pins.
The upstream tokenizer's auxiliary font is also loaded from a pinned revision.
No model source or weights are vendored. The isolated [uv project](../integrations/seeclick/pyproject.toml)
and lockfile select compatible dependencies; use that environment for all training
and evaluation commands. Its CUDA and CPU extras are mutually exclusive.

The [SeeClick authors' implementation](https://github.com/njucckevin/SeeClick)
provides the model and action conventions. Our trainer reuses Transformers/PEFT,
with stricter data checks and a single-GPU configuration instead of their DeepSpeed
launch script. The attention implementation does not require FlashAttention.

## RTX 3090 configuration

[configs/seeclick_3090.json](../configs/seeclick_3090.json) defaults to:

- NF4 language projections with double quantization and BF16 compute.
- An unquantized, frozen vision encoder and frozen vocabulary matrices.
- Rank-16 LoRA, alpha 32, dropout 0.05, on language attention and MLP projections.
- Gradient checkpointing, batch size 1, gradient accumulation 16, max length 2,048.
- Three epochs, learning rate 1e-4, cosine schedule, AdamW, checkpoint every 100 steps.

Only language LoRA parameters train. This deliberately differs from adapting the
vision encoder in some SeeClick training recipes. The runtime checks for accidental
trainable vision/base parameters. It restores the frozen vision/vocabulary matrices
to compute precision after PEFT preparation to avoid an unnecessary FP32 copy.
Trainable LoRA parameters use FP32 to support mixed-precision optimization.

The integration supports one process and one visible GPU. To choose a different
physical card, set `CUDA_VISIBLE_DEVICES` and keep the config's device `cuda:0`.
Use `fp16` instead of `bf16` for hardware without BF16 support. Full training is not
tested on a real 3090 here; batch/context settings may need adjustment based on
measured memory. If reducing context length, preflight must still accept every
example; it will never silently drop history or labels.

The original weights total 19,313,870,336 bytes. Four-bit loading still downloads
these original shards before quantization. Allow 30 GB or more free disk space for
model cache, CUDA environment, adapter checkpoints, and dataset images. A CUDA
toolkit compiler is not required by the supplied wheels, but the system NVIDIA
driver must support their CUDA 12.4 runtime.

## Train from expert demonstrations

From the repository root:

```bash
uv sync --project integrations/seeclick --extra cuda --frozen
uv run --project integrations/seeclick --extra cuda --frozen \
  scriptwob train-seeclick runs/astra-train runs/seeclick \
  --config configs/seeclick_3090.json
```

`runs/astra-train` must be a scriptWoB training collection. The command freezes its
accepted demonstrations into `runs/seeclick/dataset/`, then tokenizes and trains.
New demonstrations added to the source collection later do not change this frozen
training snapshot. Use a new training output to train on more data.

An existing `--format seeclick` export is also accepted as the source. Its images
must remain at the absolute paths referenced in `train.json`; re-export after
moving it. Native Qwen-VL image paths must fit in 254 UTF-8 bytes. The loader checks
image integrity, episode/sample membership, training provenance, prompt schema,
and numerical click labels. Oracle data requires `--allow-oracle` and remains
explicitly labeled as privileged supervision.

Training uses the entire procedure, current screenshot, and previous literal
clicks. All system/user/image/header/padding tokens are masked with `-100`. Only
the assistant click JSON and its end token contribute to the loss. The original
tokenizer expands each image tag to its native 256-token image span. No hidden
evaluator state or frontier-model reasoning is used as supervision.

## Check the data before loading weights

```bash
uv run --project integrations/seeclick --extra cuda --frozen \
  scriptwob train-seeclick runs/astra-train runs/seeclick \
  --config configs/seeclick_3090.json --prepare-only
```

This downloads only the tokenizer/code/font, validates every sequence, and writes
`data_preflight.json` with example counts and token lengths. It reserves 64 output
tokens, matching default evaluation. Oversized inputs fail before model weights
load. This is not a GPU memory test. To train this prepared output, repeat the
command with `--resume` and omit `--prepare-only`.

For a bounded training smoke test, add `--max-steps 20`. This counts optimizer
updates, not examples; gradient accumulation applies. The trainer always writes
a final complete optimizer checkpoint even when fewer than 100 updates were run.

## Resume

```bash
uv run --project integrations/seeclick --extra cuda --frozen \
  scriptwob train-seeclick runs/astra-train runs/seeclick \
  --config configs/seeclick_3090.json --resume --max-steps 500
```

Resume restores LoRA weights, optimizer, scheduler, trainer progress, and RNG state
from the newest complete checkpoint. Data, model revision, parameters, code, and
runtime must match the saved fingerprint. The positive `max_steps` target may
increase; changes are recorded. Extending it changes the cosine scheduling horizon
and is not identical to requesting the longer run from the beginning.

Interrupted progress after the last checkpoint is lost. Partially written
checkpoints are ignored; an interrupted first optimization without any complete
checkpoint needs a new output directory. Failures before optimization starts,
such as tokenizer-download failures, can be retried with `--resume`. Dataset
validation errors that occur before a run is initialized should be corrected and
retried without `--resume`.

If you used `--max-steps`, retain that target on later resumes, or raise it. To
change context length, batch size, data, LoRA settings, or model revisions, start a
new training output. Evaluation of an intermediate checkpoint remains available
even if the training process stopped.

## Evaluate

```bash
uv run --project integrations/seeclick --extra cuda --frozen \
  scriptwob evaluate-seeclick runs/seeclick configs/seed_suite.json \
  runs/seeclick-dev --split dev

uv run --project integrations/seeclick --extra cuda --frozen \
  scriptwob evaluate-seeclick runs/seeclick configs/seed_suite.json \
  runs/seeclick-test --split test
```

Use development results to choose settings/checkpoints, then evaluate the chosen
checkpoint on test. To select an intermediate checkpoint, pass
`runs/seeclick/checkpoints/checkpoint-100` as the first argument. Passing a run
directory selects its completed `adapter/`, never an arbitrary latest checkpoint.

For a pretrained baseline:

```bash
uv run --project integrations/seeclick --extra cuda --frozen \
  scriptwob evaluate-seeclick base configs/seed_suite.json \
  runs/seeclick-base-dev --split dev
```

This baseline gets the same script/history prompt and JSON action contract. Its
grounding pretraining does not guarantee it can follow this format or execute
scripts without fine-tuning. There is no planner or oracle fallback.

Inference loads the model once and generates greedily from each new screenshot,
using the same ChatML prefix as training. It decodes only newly generated tokens.
Outputs must contain one normalized JSON click (`action_type=4`, two numerical
coordinates); invalid outputs fail the episode. No clipping or target snapping
is applied. Context overflow fails explicitly rather than removing history.

`--max-episodes N` limits task coverage; `--max-total-steps N` limits new inference
calls for this invocation. Evaluation allows one attempt per dev/test task. A
budget-cut episode remains incomplete/failed, so use a sufficiently large budget
for final scores. Repeating the command resumes remaining tasks; use a new output
for a fresh evaluation. Adapter content hashes bind resume to the evaluated weights.

By default, a trained adapter must be evaluated with its training manifest, using
that manifest's held-out split. `--allow-different-manifest` permits deliberate
transfer evaluation; first verify those tasks/structures were not training data.

## Outputs

```text
runs/seeclick/
  scriptwob_training.json     # pinned model/config, data identity, status, limits
  data_preflight.json         # validated sample lengths and counts
  train_results.json          # final training metrics and adapter path
  dataset/                   # frozen export, when training from a collection
  adapter/                   # completed LoRA adapter and immutable provenance
  checkpoints/checkpoint-N/   # adapter, optimizer, scheduler, RNG and trainer state

runs/seeclick-dev/
  report.json                # coverage, first-attempt success and per-level metrics
  run.json                   # evaluated checkpoint hashes/settings
  attempts/                  # screenshots, actions, outcomes and failure details
```

Saved adapters depend on the pinned pretrained base model; they are not merged
standalone model weights. No upload to Hugging Face or external experiment tracker
is performed. Reports are local files.

## Validation without pretrained weights

```bash
uv sync --project integrations/seeclick --extra cpu --frozen
uv run --project integrations/seeclick --extra cpu --frozen pytest -q
SCRIPTWOB_TEST_NATIVE_SEECLICK=1 \
  uv run --project integrations/seeclick --extra cpu --frozen \
  pytest tests/test_seeclick_runtime.py -q
```

The optional native test downloads only pinned model configuration/custom code,
then tests a tiny randomly initialized original Qwen-VL architecture. Tests cover
image forward/backward, language-only gradients, adapter reload/generation, loss
masking, and Trainer optimizer/RNG resume. These verify implementation mechanics,
not pretrained GUI accuracy or 4-bit CUDA memory fit. Full pretrained GPU training
and benchmark scores must be measured on the target machine.
