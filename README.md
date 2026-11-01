# VLA MiniWoB benchmark

An initial implementation of the supplied **CompWoB / MiniWoB++ / TinyVLA-B
research design v1.1**. Uses uv, pinned native MiniWoB++ browser tasks, numerical
mouse actions, a private strict-order evaluator, and GPT-6 Astra demonstrations.

The working pilot includes 7 constituent controls and 6 compositions with 2–7
components, forward/reversed **execution orders**, equivalent reverse-clause
wording, and paired UIs. Every family/order/wording is present in each split.
`configs/pilot.json` freezes 114 training, 38 development, and 38 evaluation
instances. These are task instances, not collected demonstrations.

This is an **adapted CompWoB-derived pilot**, not a reproduction of the original
50-task leaderboard. The compositions reuse unchanged native MiniWoB constituent
pages in independent panels. Layout, order semantics, completion handling, and
procedure assembly are explicit adaptations. Typing, dragging, scrolling,
page-transition tasks, and 8-component compositions are outside this release.

## Setup and first run

Install [uv](https://docs.astral.sh/uv/) and Chrome/Chromium, then:

```bash
uv sync --frozen
uv run --frozen vla-benchmark tasks
uv run --frozen vla-benchmark smoke
uv run --frozen pytest
uv run --frozen pytest --run-browser
```

Selenium Manager can download the matching ChromeDriver on first use. For a
managed/offline installation, set **both** `MINIWOB_CHROME_BINARY` and
`MINIWOB_CHROMEDRIVER`. `SE_CACHE_PATH` can direct Selenium's cache to a writable
directory. The environment serves installed MiniWoB assets on a loopback-only
ephemeral HTTP port. Smoke screenshots are written to `runs/smoke/`.

Create another immutable manifest if you want a larger dataset:

```bash
uv run --frozen vla-benchmark manifest configs/experiment.json \
  --train 100 --dev 10 --evaluation 20 --seed 2026
```

Counts are **per task/order/wording cell**. Repeated `--task TASK` options select
families. Values/internal native layout and the panel permutation have separate
seeds; paired order/wording variants share both. Splits have disjoint seeds.

## Generate examples with GPT-6 Astra

Set `OPENAI_API_KEY` in your shell; `.env` is not loaded automatically. Then:

```bash
uv run --frozen vla-benchmark collect configs/pilot.json runs/astra-train \
  --max-episodes 10 --attempts 2
uv run --frozen vla-benchmark report runs/astra-train
uv run --frozen vla-benchmark export runs/astra-train exports/train --horizon 16
```

The default model is exactly `gpt-6-astra`, using the Responses API, high-detail
PNG image inputs, and a structured numerical action schema. A call requests one
primitive, executes it in Chrome, and observes again. It does not invent example
trajectories without running them. See the official [model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs).

`--max-episodes` bounds **new attempts per invocation**, including failures;
`--attempts` bounds total collection attempts per instance, including its first
attempt. With the default 64-step
budget this is at most 640 logical action requests for 10 attempts, plus SDK
transport retries. API calls incur usage charges. Repeating the same command
resumes without overwriting existing attempts. You can increase `--attempts` on
resume: accepted instances are skipped, and exhausted failures become eligible
for the additional attempts. The increase is recorded in `run.json` under
`resume_history`. Continue using that new limit on later invocations; decreasing
it is rejected. Changing the manifest, policy, or runtime requires a new run
directory. Source changes also require a new directory unless an exact reviewed
compatibility entry exists; this resume fix includes one for the initial release.
Only validated success is exported; rejected and interrupted attempts remain auditable.

To process the full 114-instance pilot with up to three attempts per instance:

```bash
uv run --frozen vla-benchmark collect configs/pilot.json runs/astra-train \
  --max-episodes 342 --attempts 3
```

The command stops when it has processed all eligible instances, even if fewer
than 342 new attempts are needed. Existing attempts count toward each instance's
limit. A run that exhausts three attempts on an instance may still have a shortfall.

Collection is restricted to training instances. For a separate, single-attempt
frontier reference or development check:

```bash
uv run --frozen vla-benchmark evaluate configs/pilot.json runs/astra-dev \
  --policy astra --split dev --max-episodes 38
uv run --frozen vla-benchmark evaluate configs/pilot.json runs/astra-eval \
  --policy astra --max-episodes 38
uv run --frozen vla-benchmark export runs/astra-dev exports/dev --split dev \
  --normalization exports/train/normalization.json
```

All model attempts count for evaluation, including invalid actions and refusals.
Infrastructure failures are retained, counted separately, and excluded from the
success denominator. Reports flag incomplete coverage and withhold the complete
benchmark score until every declared evaluation instance has a usable outcome.
Collection acceptance rates are never reported as zero-shot benchmark scores.

## Trainable data and policy integration

```python
from vla_miniwob.dataset import ChunkDataset

dataset = ChunkDataset("exports/train", horizon=16)
sample = dataset[0]
# image: RGB uint8 H×W×3; raw_lang: full procedure + past literal actions
# state: float32 [3]; action: float32 [16,3]; is_pad: bool [16]
```

HDF5 exports contain raw executed actions, cursor/button state, current RGB
screenshots, the procedure, and per-timestep procedure/history text. Chunks are
built within a single episode; padding is masked. Only successful **training**
episodes fit normalization. Evaluator goals, DOM, order ledger, rewards, and
future screenshots are excluded from policy samples and HDF5 files.

TinyVLA-B training and checkpoint evaluation use a separate pinned uv runtime:

```bash
bash scripts/setup_tinyvla.sh cuda
uv run --project integrations/tinyvla --frozen --extra cuda \
  vla-benchmark train-tinyvla runs/astra-train runs/tinyvla-b --max-steps 2000
uv run --project integrations/tinyvla --frozen --extra cuda \
  vla-benchmark evaluate-tinyvla runs/tinyvla-b configs/pilot.json \
  runs/tinyvla-b-eval --split eval
```

Training automatically exports accepted demonstrations, checks full context,
records vision previews, trains native diffusion/LoRA/projector parameters and
saves resumable checkpoints. Defaults target a single RTX 3090. Add `--resume`
with the same settings to continue training. Evaluation runs every instance in
the selected split by default and writes the existing benchmark reports.
See [training, resume and evaluation details](docs/tinyvla.md) and the
[configuration](configs/tinyvla_b.json). Native model and training mechanics are
CPU-tested; full pretrained GPU training, memory fit, and GUI scores remain to
be measured on your machine.

Other VLA models implement the small [Policy protocol](src/vla_miniwob/contracts.py):
`reset(seed)`, `predict(observation) -> H×3`, and `provenance()`. Predictions must
be **denormalized** numerical GUI actions. Return one sampled chunk; the harness
executes the fixed prefix, default K=1, then observes again.

```bash
uv run --frozen vla-benchmark evaluate configs/pilot.json runs/student \
  --policy my_package.policy:create --policy-config configs/my_model.json
```

The user-supplied factory receives that JSON as keyword arguments. Keep policies
in importable packages in the active uv environment. Include checkpoint hashes,
training configuration, sampler settings, and `prediction_horizon` in provenance.
The `random` policy is an execution control, not a pretrained VLA baseline.

## What is recorded and validated

- `run.json`: model/prompt/settings, source/runtime fingerprints, H/K, and protocol.
- `manifest.json`: frozen instance seeds, families, orders, and wording conditions.
- `episodes/*/episode.json`, PNGs, `transitions.jsonl`: procedure, pre-action and
  settled observations, requested and executed primitives, and execution feedback.
- `episodes/*/attempt.json`, `private.json`: identity, provider usage/response IDs,
  semantic events, native rewards, first order violation, and rejection reason.
- `report.json`: coverage shortfalls, accepted data counts, per-task/complexity
  success, final-state versus order-valid success, and clustered uncertainty when
  enough independent UI instances are available.

The model sees only its full procedure, current screenshot, cursor/button state,
and literal past actions/feedback. No correct-next-step information is supplied.
Native task completion is absorbing; premature meaningful interaction with a
later component permanently invalidates chronology. Checkbox partial credit is
not full success. A successful final state cannot erase an earlier violation.

Saved primitives can be replayed and revalidated:

```bash
uv run --frozen vla-benchmark replay runs/astra-train/episodes/EPISODE \
  configs/pilot.json runs/replay-check
```

See [the protocol](docs/protocol.md) for precise semantics and known limitations,
and [upstream sources](configs/upstream_sources.json) for revisions and licenses.
Live Astra collection requires your API key; local fixture tests use clearly
labeled test policies and are not teacher demonstrations or benchmark results.
