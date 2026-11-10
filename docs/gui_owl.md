# GUI-Owl fine-tuning and evaluation

This integration fine-tunes the released **GUI-Owl-1.5-Instruct 2B, 4B, and 8B**
checkpoints on existing scriptWoB demonstrations. They share the Qwen3-VL model
family and use its native multi-image processor and chat template. Training and
evaluation return one normalized click per observation, using the same action
contract as the SeeClick experiments.

The integration reuses Hugging Face Transformers, PEFT, and Trainer. It does not
run the upstream MobileAgent orchestration system or give the policy private
script state, DOM content, future frames, or a separate planner. This is a
scriptWoB adaptation of the pretrained GUI policy, not a reproduction of the
authors' published agent scores.

## Separate scripts for each model

Run these from the repository root, one model at a time:

| Model | Fetch pretrained checkpoint | Train | Evaluate dev and test |
| --- | --- | --- | --- |
| 2B | `./scripts/gui_owl/fetch_2b.sh` | `./scripts/gui_owl/train_2b.sh` | `./scripts/gui_owl/eval_2b.sh` |
| 4B | `./scripts/gui_owl/fetch_4b.sh` | `./scripts/gui_owl/train_4b.sh` | `./scripts/gui_owl/eval_4b.sh` |
| 8B | `./scripts/gui_owl/fetch_8b.sh` | `./scripts/gui_owl/train_8b.sh` | `./scripts/gui_owl/eval_8b.sh` |

Fetch downloads the exact revision in that size's config, including safetensors
and processor/tokenizer files. It uses a small uv-managed Hugging Face Hub
dependency and does not install PyTorch or require a GPU. Repeating a fetch
reuses the cache. Downloads honor `HF_HOME`, `HF_HUB_CACHE`, and the legacy
Transformers cache variables, using the same precedence as the model loader.
Use the same cache environment for fetching, training, and evaluation.

Training automatically synchronizes the frozen CUDA runtime through uv. It
defaults to `datasets/seeclick-800-code`, the matching
`configs/gui_owl_{2b,4b,8b}_3090.json`, and `runs/gui-owl-{2b,4b,8b}-code`.
Evaluation loads that training run's completed adapter, uses
`configs/suite_800.json`, and writes `runs/gui-owl-{size}-code-dev` and
`runs/gui-owl-{size}-code-test`. Each split gets a 10,000-call budget by default.
Evaluation stops if a split's command exits with an error; check `report.json`
for completion when using a smaller call budget.

The scripts also work when invoked from another directory: default paths remain
relative to the repository, and explicitly supplied paths remain relative to
your current directory. Every script supports `--help` without installing or
downloading anything. Custom paths and CLI options can be supplied as follows:

```bash
# Validate the default dataset, then train that prepared run.
./scripts/gui_owl/train_2b.sh --prepare-only
./scripts/gui_owl/train_2b.sh --resume

# Use another export and training output.
./scripts/gui_owl/train_4b.sh datasets/seeclick-800-code runs/gui-owl-4b-experiment

# Evaluate that run: checkpoint, manifest, output prefix, then CLI options.
./scripts/gui_owl/eval_4b.sh runs/gui-owl-4b-experiment configs/suite_800.json \
  runs/gui-owl-4b-experiment --max-total-steps 20000
```

Training does not resume automatically. Add `--resume` for an existing prepared
or interrupted run; add `--max-steps N` to extend its cumulative optimizer-step
target. Edit the matching configuration file to change training settings before
a new run; the launchers reserve `--config` to keep model sizes aligned.
Evaluation reserves `--split` because each script always evaluates both splits.
Use the underlying CLI commands below for a custom config or one split only.

## Install the isolated runtime

Run commands from the repository root on your training machine:

```bash
cd /home/cadmium/projects/VLA-miniwob
uv sync --project integrations/gui_owl --extra cuda --frozen
```

The separate [uv project](../integrations/gui_owl/pyproject.toml) keeps modern
Qwen3-VL dependencies separate from SeeClick's original Qwen-VL runtime.

| Component | Pin |
| --- | --- |
| Python | 3.11 |
| Transformers | 4.57.6 |
| PyTorch / torchvision | 2.6.0 / 0.21.0 |
| PEFT / Accelerate | 0.17.1 / 1.10.1 |
| bitsandbytes, CUDA extra | 0.48.1 |
| GUI-Owl 2B | `mPLUG/GUI-Owl-1.5-2B-Instruct` at `528ceaec795bbfbe6103bd79e03db849feadfb24` |
| GUI-Owl 4B | `mPLUG/GUI-Owl-1.5-4B-Instruct` at `3f061c2c562cc860c42bf32542a70e07a7ff4840` |
| GUI-Owl 8B | `mPLUG/GUI-Owl-1.5-8B-Instruct` at `06d5faecff74840bab2be2425e9c42667a5d04fc` |

Model weights and processor assets use the same immutable revision. Qwen3-VL is
implemented natively by Transformers; model repository Python is not executed.
The CUDA extra selects CUDA 12.4 wheels and needs a compatible NVIDIA driver. No
FlashAttention build or CUDA toolkit compiler is required. The CPU extra is for
small implementation checks, not practical full-model training; it is mutually
exclusive with the CUDA extra.

Four-bit loading downloads the original model shards before quantizing them.
Allow disk space for all three base checkpoints, the runtime, dataset images, and
optimizer checkpoints. Full pretrained training and the peak memory of these
configurations still need to be measured on the target RTX 3090.

## Reuse the existing 800 demonstrations

Use the existing `datasets/seeclick-800-code` directory if it already contains
`dataset.json`, `train.json`, and `images/`. GUI-Owl reads that export directly;
the name of the export format does not require a SeeClick model or environment.
It checks provenance, image hashes, instruction style, labels, and episode/step
continuity, then builds native Qwen3-VL inputs.

If only the original natural-language frozen export exists, convert it once:

```bash
uv run --frozen scriptwob reformat-seeclick \
  runs/seeclick-800/dataset configs/suite_800.json datasets/seeclick-800-code
```

Do not rerun that conversion when its output already exists. It preserves the
collected clicks and images and uses the original manifest to render code-like
instructions. No frontier API calls or new expert collection are needed.
`configs/suite_800.json` must be your original 800/100/100 manifest.

An accepted training collection is also a valid source. Pass its `train`
subdirectory and add `--prompt-style code`; the trainer freezes an export under
its output directory. Existing exports inherit their recorded prompt style.
Dev/test demonstrations cannot be used as training data.

## Screenshot history and memory

All three configurations use `history_images: 3`: up to **three previous
pre-action screenshots plus the current screenshot**, in chronological order.
The prompt also retains the full literal click history. Previous screenshots
are reconstructed from consecutive saved steps of the same episode during
training and retained from actual observations during evaluation. The history
is cleared at every episode reset. No later screenshot is included in a training
example.

Each image is labeled with its step and whether it is historical or current.
The model must choose its next click on the current page. Only its next-click
JSON and end-of-message token receive training loss; instructions, images,
history, and padding are masked. The model does not generate or receive a
reasoning trace.

To compare memory settings, copy all three configs, set `history_images` to `0`
for current-image-only runs, and use new training and evaluation output
directories. Keep the setting identical across training and evaluation. A
three-frame history window can still lose a distant branch-selection
observation; it is a controlled memory allowance, not a guarantee of sufficient
memory for every script.

## Train the three sizes

The configurations differ only in their pinned base model:

- [2B](../configs/gui_owl_2b_3090.json), [4B](../configs/gui_owl_4b_3090.json),
  [8B](../configs/gui_owl_8b_3090.json).
- NF4 language projections with double quantization and BF16 compute.
- Frozen vision encoder and language base weights; rank-16 language LoRA,
  alpha 32, dropout 0.05.
- Gradient checkpointing, batch size 1, gradient accumulation 16.
- 8,192 total tokens, reserving room for the next action; image processing uses
  the same 65,536–262,144 pixel limits for each size.
- Three epochs, learning rate 1e-4, cosine schedule, seed 2026.

First validate the 2B inputs without downloading model weights:

```bash
uv run --project integrations/gui_owl --extra cuda --frozen \
  scriptwob train-gui-owl datasets/seeclick-800-code runs/gui-owl-2b-code \
  --config configs/gui_owl_2b_3090.json --prepare-only
```

This downloads processor assets, checks sequence lengths and image histories,
and writes `data_preflight.json`. It is not a GPU memory test. Context overflow
fails explicitly; examples and screenshots are not silently truncated. Resume
that prepared output to train:

```bash
uv run --project integrations/gui_owl --extra cuda --frozen \
  scriptwob train-gui-owl datasets/seeclick-800-code runs/gui-owl-2b-code \
  --config configs/gui_owl_2b_3090.json --resume

for size in 4 8; do
  uv run --project integrations/gui_owl --extra cuda --frozen \
    scriptwob train-gui-owl datasets/seeclick-800-code "runs/gui-owl-${size}b-code" \
    --config "configs/gui_owl_${size}b_3090.json" || break
done
```

If you skipped preflight, omit `--resume` when starting the 2B run. Train one
size at a time on the 3090. Use `CUDA_VISIBLE_DEVICES` to select a different
physical GPU while keeping `device: "cuda:0"` in the config.

A bounded training check can use `--max-steps 20`; this counts optimizer updates,
not individual next-click samples. After inspecting its memory use, resume with
a larger positive cumulative `--max-steps` target. For a clean fixed-epoch
comparison, start fresh full-training outputs with the same epoch setting for
all sizes. Extending a completed run changes its cosine schedule and differs
from requesting that total training duration from the beginning.

Resume restores the newest complete optimizer/scheduler/RNG checkpoint. Data,
model, runtime, prompt/history settings, and implementation must match the saved
fingerprint. Change settings using a new output directory. An external export
is referenced in place; retain its images and metadata for later resume.

## Evaluate dev and test

After each training run completes:

```bash
for size in 2 4 8; do
  for split in dev test; do
    uv run --project integrations/gui_owl --extra cuda --frozen \
      scriptwob evaluate-gui-owl "runs/gui-owl-${size}b-code" \
      configs/suite_800.json "runs/gui-owl-${size}b-code-${split}" \
      --split "$split" --max-total-steps 10000 || break 2
  done
done
```

Evaluation reads the exact base revision, image limits, screenshot-history
window, prompt style, and context length from the checkpoint. It loads one
model, generates greedily, and accepts exactly one JSON click with normalized
coordinates. Invalid output fails the episode without snapping or an oracle
fallback. Use development scores to select settings and report the held-out
test scores for the selected experiment.

Passing a training directory selects its completed `adapter/`. To evaluate an
intermediate checkpoint, explicitly pass, for example,
`runs/gui-owl-2b-code/checkpoints/checkpoint-100`. An incomplete training run does
not silently select whichever checkpoint happens to exist.

The same manifest used for training supplies the dev/test tasks. Transfer to a
different manifest requires `--allow-different-manifest`. Each evaluation split
uses one attempt per episode, with no retries for failed policy decisions.
`--max-total-steps` bounds new policy calls; incomplete coverage is recorded in
the report. Repeating the same command resumes the remaining tasks. Use a new
output directory, or `--fresh`, for a different adapter or a fresh evaluation.

For an adapted zero-shot baseline, replace the checkpoint path with `base-2b`,
`base-4b`, or `base-8b`. Add `--prompt-style code` to match code-prompt training
and `--history-images 3` to make the memory setting explicit:

```bash
uv run --project integrations/gui_owl --extra cuda --frozen \
  scriptwob evaluate-gui-owl base-2b configs/suite_800.json \
  runs/gui-owl-2b-base-code-dev --split dev --prompt-style code \
  --history-images 3 --max-total-steps 10000
```

These baselines use scriptWoB's normalized-click JSON prompt and action contract.
They do not use GUI-Owl's original native tool prompt or MobileAgent execution
system, so they measure adaptation to this benchmark protocol. Invalid-format
outputs can fail even when the pretrained model could operate with its original
agent interface.

## Artifacts and interpretation

```text
runs/gui-owl-2b-code/
  scriptwob_training.json      # pinned configuration and dataset provenance
  data_preflight.json          # sample/image/token counts and context checks
  train_results.json          # final training metrics
  dataset/                    # only when input was a collection
  adapter/                    # completed LoRA adapter and immutable provenance
  checkpoints/checkpoint-N/    # adapter, optimizer, scheduler, RNG, trainer state

runs/gui-owl-2b-code-dev/
  report.json                 # coverage, success rates, per-level statistics
  run.json                    # checkpoint and inference settings
  attempts/                   # screenshots, actions, actual prompts, outcomes
```

Saved adapters depend on the pinned pretrained checkpoint; they are not merged
standalone weights. Nothing is uploaded to a model hub or experiment tracker.
The existing plotting and GIF tools consume these rollout artifacts.

For the capacity comparison, keep training exposure, prompt style, history,
image limits, decoding, and evaluation tasks fixed across sizes. Inspect both
overall task success and conditional-family results. The 8B model also has a
larger vision encoder, and pretraining differences remain a confound; the model
names are nominal size labels, not a controlled language-parameter-only sweep.
Report actual parameter counts from the loaded models alongside these labels.

## Upstream sources

- GUI-Owl [2B](https://huggingface.co/mPLUG/GUI-Owl-1.5-2B-Instruct),
  [4B](https://huggingface.co/mPLUG/GUI-Owl-1.5-4B-Instruct), and
  [8B](https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct) checkpoints.
- [Mobile-Agent-v3.5 / GUI-Owl paper](https://arxiv.org/abs/2602.16855) and
  [official repository](https://github.com/X-PLUG/MobileAgent).
- [Qwen3-VL native inference and multi-image examples](https://github.com/QwenLM/Qwen3-VL).
- [Transformers Qwen3-VL documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen3_vl).
