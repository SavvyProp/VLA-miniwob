# scriptWoB

A generated GUI benchmark for executing natural-language procedures: sequences,
conditionals, counted repetition, while/until loops, iteration over records, and
nested control flow. A policy reads the procedure and screenshot, then predicts
**one click at a time**. A private interpreter checks execution order, branches,
iteration counts, and stopping.

This repository provides task generation, a Chrome environment, frontier-expert
collection, SeeClick fine-tuning, checkpoint evaluation, and portable exports. It replaces the earlier
TinyVLA benchmark; existing run directories are not migrated.

The implementation reuses pinned [MiniWoB++](https://github.com/Farama-Foundation/miniwob-plusplus)
browser infrastructure and core assets. Script pages and grammar are new: these
are custom MiniWoB++ tasks, not its unchanged official task suite or leaderboard.
No pretrained weights or GPU libraries are required for collection.

## Tasks and difficulty

There are **22 families** across six structural levels:

| Level | Structure | Example |
| --- | --- | --- |
| 0 | Atomic action | Press Maple, then Done. |
| 1 | Sequence | Press Maple, toggle Cedar, then Done. |
| 2 | Conditional | If Cedar is checked, press Maple; otherwise press Amber. |
| 3 | Repetition and stopping | Increase a counter while below 4; advance until the current record is red. |
| 4 | Branches with loops | For each record, approve if red; otherwise reject; then advance. |
| 5 | Nested control flow | While a counter is below 3, repeat a conditional twice, then increase the counter. |

Controls include buttons, checkboxes, counters, status displays, and record queues.
Future record details are revealed by Next. Current records, visited markers, counters, a
neutral action count, and literal click history support progress tracking. There
are no explicit recall/delayed-memory families; this does not imply every procedure
is solvable from a single screenshot without its action history.

The generator samples a constrained program grammar and renders unambiguous
language with bracketed blocks. It varies composition, labels, values, predicates,
repetitions, queue contents, and layout. Every instance must pass its reference
executor. The frontier model supplies demonstrations; free-form LLM paraphrasing
of instructions is not enabled.

Levels describe structure, not guaranteed empirical difficulty. Each instance
also records program size, nesting depths, predicate complexity, reference click
count, executed branches/loops, and scene size. Zero-iteration cases are intentional.
Compare performance within action-length bands as well as by level. See the
[exact protocol](docs/protocol.md).

## Install and inspect

Install [uv](https://docs.astral.sh/uv/) and Chrome/Chromium, then:

```bash
uv sync --frozen
uv run --frozen scriptwob doctor
uv run --frozen scriptwob catalog
uv run --frozen scriptwob inspect configs/seed_suite.json
uv run --frozen scriptwob validate configs/seed_suite.json
uv run --frozen scriptwob preview configs/seed_suite.json runs/preview --index 4
```

`preview` writes `screen.png`, `instruction.txt`, and public metadata. Add `--headed`
to show Chrome. Selenium Manager can obtain a matching ChromeDriver. For a managed
installation set both `MINIWOB_CHROME_BINARY` and `MINIWOB_CHROMEDRIVER`; use
`SE_CACHE_PATH` for a writable driver cache. The environment serves local assets
on a temporary loopback-only HTTP port. Linux is tested; collection uses POSIX locks.

Checked-in starter manifests contain **task instances**, not demonstrations:

| Manifest | Train | Dev | Test | Purpose |
| --- | ---: | ---: | ---: | --- |
| `configs/seed_suite.json` | 132 | 36 | 36 | All six levels, new task instances |
| `configs/structure_suite.json` | 120 | 24 | 24 | Levels 2–5, disjoint normalized structures |

Generate a larger manifest before collecting:

```bash
uv run --frozen scriptwob generate configs/train_large.json \
  --train 6000 --dev 600 --test 600 --seed 2026
uv run --frozen scriptwob generate configs/structure_large.json \
  --train 3000 --dev 300 --test 300 --split-strategy structure --seed 2026
```

Counts are totals per split, balanced across selected levels. `--levels 2,3,4,5`
restricts generation. A seed split groups exact tasks into stable split buckets,
so different seeds cannot put an identical instruction/world/layout in train and
test. Repeated tasks within a split remain possible, particularly for atomic
families; `inspect` reports unique task counts. A structure
split holds out normalized AST shapes, ignoring labels, targets, literal values,
and repeat counts. It tests composition within this grammar, not novel language or
unlimited semantic diversity. A structural split can hold out an entire family or
operator; it is not a controlled test of new combinations of only trained operators.
Levels 0–1 are excluded from structure splitting
because shape diversity is small. Train on the **same manifest's training split**;
mixing in other training sets can invalidate a structural holdout.

## Collect frontier demonstrations

Set `OPENAI_API_KEY` in your shell. Alternatively, copy `.env.example` to `.env`,
fill it in locally, and use `uv run --env-file .env`. The CLI does not load `.env`
automatically. Collect training demonstrations and held-out expert rollouts with
one command, using a new output directory:

```bash
uv run --env-file .env --frozen scriptwob collect \
  configs/seed_suite.json runs/astra-all \
  --all-splits --attempts 3 --max-total-steps 10000
uv run --frozen scriptwob report runs/astra-all
```

The command processes train, dev, then test into `runs/astra-all/train`,
`runs/astra-all/dev`, and `runs/astra-all/test`. The root `report.json` combines
progress and refreshes between splits; each split's own `report.json` refreshes
after every attempt. Training allows up to three
attempts per task, stopping after success; dev/test allow exactly one attempt.
Empty splits are explicitly reported as skipped. Use only the `train` directory
for fine-tuning or export. Dev/test expert rollouts measure the teacher; SeeClick
evaluation executes those held-out tasks itself.

Add `--fresh` to create a unique output root on every invocation. The printed
root replaces `runs/astra-all` in later commands: train from `<printed-root>/train`
and read held-out results in `<printed-root>/dev` and `<printed-root>/test`.

The default is `gpt-6-astra`, medium reasoning effort, using OpenAI
[Responses structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
Each request includes the full procedure, current PNG, and literal past clicks.
It returns normalized coordinates; Chrome executes the click before the next
request. No DOM, private program, next-action target, reward, or correctness
annotation reaches the teacher. Model reasoning is not a training target.

With `--all-splits`, `--max-episodes N` selects the first N instances **per split**,
including previously attempted ones. Omit it to select every instance. The
`--max-total-steps` budget is shared across all three splits and bounds new model
predictions in this invocation, including failures; default 10,000. If it runs out,
later tasks or entire splits can remain unattempted. Each prediction allows up to
three additional HTTP retries using the OpenAI SDK's backoff. Set
`--transport-retries 0` to disable these retries, or choose a value from 0 to 3.
SDK retries happen before the GUI click, so they do not replay an executed action.
They are separate from the per-task `--attempts` limit. Request attempts can therefore
exceed `--max-total-steps` (up to four SDK request attempts per prediction by default).
These limits do not cap dollar costs. Retry counts are logged; token usage from
earlier failed requests may be unknown, even if the final request succeeds.

Without `--all-splits`, collection still targets only train. For an optional,
bounded training pilot:

```bash
uv run --env-file .env --frozen scriptwob collect \
  configs/seed_suite.json runs/astra-train \
  --max-episodes 5 --attempts 3 --max-total-steps 100
```

To expand that pilot to all training tasks:

```bash
uv run --env-file .env --frozen scriptwob collect \
  configs/seed_suite.json runs/astra-train \
  --attempts 3 --max-total-steps 10000
```

To start from the beginning, add `--fresh`:

```bash
uv run --env-file .env --frozen scriptwob collect \
  configs/seed_suite.json runs/astra-all \
  --all-splits --fresh --attempts 3 --max-total-steps 10000
```

`--fresh` creates a uniquely named sibling directory such as
`runs/astra-all-20261103T120000Z-abcd1234`, leaving any existing run untouched.
The actual path is printed at startup and saved as `output_dir` in `report.json`.
For `--all-splits`, it creates one fresh root containing all three splits.
Use that root for reporting and its `train` subdirectory for export and training.
Each invocation with `--fresh` starts over; it also works with `evaluate` and
`evaluate-seeclick`.
A new, unused output path starts from the beginning without needing this flag.

Without `--fresh`, repeating a command resumes and skips successes. Increase
`--max-episodes` and `--attempts` as needed; the request budget can change between invocations. An
attempt interrupted by a budget limit or crash is retained as unsuccessful; its
next retry starts afresh. A changed manifest, policy/model/prompt, implementation,
or headless setting requires a new run directory. Limit changes are recorded.
Provider/infrastructure errors stop collection rather than consuming every task.
Temporary API connection failures are retried within the current prediction
first. After those retries are exhausted, the run stops and retains its results.
To continue a stopped run, use its exact output directory and omit `--fresh`.
Keep the original source code and teacher settings: changing the implementation
or retry configuration requires a new run and cannot resume an older fingerprint.

Only successful training trajectories are accepted. Failures remain auditable.
Reports distinguish first-attempt performance, success after retries, coverage,
infrastructure failures, and known provider usage. Reaching the retry cap does
not mean every task has an accepted demonstration.

## Export for fine-tuning

```bash
uv run --frozen scriptwob export runs/astra-all/train exports/messages
uv run --frozen scriptwob export runs/astra-all/train exports/seeclick --format seeclick
```

These paths use the combined collection above. Substitute the printed root if
you used `--fresh`; for a train-only collection, pass its output directory directly.

Each accepted trajectory produces one example per executed click, including Done.
The exporter verifies manifest identity, supervision, transition continuity, and
screenshot hashes. Only pre-action images, full instructions, literal history,
and next executed clicks enter model samples. Dev/test runs cannot become training
exports. No ASTs, oracle targets, future screenshots, or evaluator verdicts enter
the model messages.

- **`messages`**: portable `train.jsonl` with relative image paths, user/assistant
  messages, and `{"x": ..., "y": ...}` targets. Adapt it to the chosen image
  processor/chat template and mask the prompt from the training loss.
- **`seeclick`**: `train.json` in the SeeClick/Qwen-VL conversation format, `<img>`
  paths, and `{"action_type": 4, "click_point": [x, y]}` targets. Image paths are
  absolute for its loader; re-export after moving the dataset.

The SeeClick export follows its [conversation loader](https://github.com/njucckevin/SeeClick/blob/main/finetune/finetune.py)
and [MiniWoB preprocessing](https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/miniwob_process.py).
It uses valid JSON arrays and full history, so prompting is adapted rather than an
exact reproduction. The SeeClick trainer below uses these same prompts and masks
all instruction/history/image tokens from the loss. It rejects examples that exceed
the context window, including room for the next action. Check processed-image
legibility too: smaller image resolutions can make GUI text hard to read,
confounding visual grounding with procedural reasoning.

Exports include copied `images/`, `dataset.json` with provenance and counts, and
`episodes.json` with episode IDs. The source run remains the audit trail.

## Fine-tune and evaluate SeeClick

Training starts from the pretrained **`cckevinn/SeeClick`** GUI-grounding checkpoint,
using pinned original Qwen-VL code and tokenizer. The default configuration targets
one RTX 3090: 4-bit NF4 language weights, BF16 compute, language-only LoRA, frozen
vision encoder, batch size 1, gradient accumulation 16, and 2,048 tokens. It is a
starting configuration; full-checkpoint GPU memory use still needs verification
on your machine.

Run from the repository root. The separate uv environment keeps the older
Qwen-VL dependencies out of the benchmark environment:

```bash
uv sync --project integrations/seeclick --extra cuda --frozen

uv run --project integrations/seeclick --extra cuda --frozen \
  scriptwob train-seeclick runs/astra-all/train runs/seeclick \
  --config configs/seeclick_3090.json

uv run --project integrations/seeclick --extra cuda --frozen \
  scriptwob evaluate-seeclick runs/seeclick configs/seed_suite.json \
  runs/seeclick-dev --split dev
```

The training command automatically exports accepted training demonstrations into
`runs/seeclick/dataset/`. You can instead pass an existing SeeClick export directory
as its first argument. Downloads require about **19.3 GB for the original model**,
even with 4-bit loading; allow at least 30 GB free for model/runtime/checkpoint caches.
The CUDA extra uses PyTorch CUDA 12.4 wheels and needs a compatible NVIDIA driver.

Add `--resume` to continue an interrupted run with its optimizer/RNG state. Add
`--max-steps 20` for a short initial run; later resume with a larger positive
`--max-steps` target. Use `base` in place of `runs/seeclick` to evaluate the original
pretrained SeeClick without your adapter. Both evaluations use greedy single-click
predictions and one attempt per dev/test episode.

Final adapters are in `runs/seeclick/adapter/`; resumable checkpoints are under
`runs/seeclick/checkpoints/`. Evaluation results are in the selected output's
`report.json`, with screenshots and action traces under `attempts/`.
See [SeeClick setup, preflight, resume, and limitations](docs/seeclick.md).

## Evaluate a reference or student

```bash
uv run --env-file .env --frozen scriptwob evaluate \
  configs/seed_suite.json runs/astra-dev --split dev --max-total-steps 1000
```

Student plugins implement the [Policy interface](src/scriptwob/contracts.py):
`reset(seed)`, `predict(Observation) -> ClickAction`, and `provenance() -> dict`.
Observations contain a PIL RGB image, instruction, and literal executed-action
history. Use `format_context()` to reproduce exported context, and convert model
output into `ClickAction(x=..., y=...)`. Include checkpoint/config hashes in provenance.

```bash
uv run --frozen scriptwob evaluate configs/seed_suite.json runs/student-test \
  --split test --policy-factory my_policy:create \
  --policy-config configs/my_student.json
```

The importable factory receives JSON configuration as keyword arguments. Evaluation
permits one attempt per episode; training retries are never evaluation scores.
Use a new directory for a new evaluation. Plugins are trusted code, not sandboxed.

## Artifacts and verification

```text
runs/astra-all/train/        # dev/ and test/ have the same rollout structure
  run.json                  # fingerprint and limit history
  manifest.json             # private frozen programs and worlds
  report.json               # coverage, first-attempt/retry metrics, usage
  attempts/<id-hash>-001/
    episode.json            # public instruction and rollout metadata
    transitions.jsonl       # requested/executed clicks and PNG hashes
    000000.png ...          # observations before/after clicks
    attempt.json            # attempt identity
    private.json            # verdict, reason, evaluator state
    provider_calls.json     # response IDs, model identity, usage
    progress.json           # crash recovery state
```

Keep `manifest.json` and `private.json` out of model inputs.

```bash
uv run --frozen pytest
uv run --frozen pytest --run-browser
uv run --frozen ruff check src tests
```

For an end-to-end check without an API key:

```bash
uv run --frozen scriptwob collect configs/seed_suite.json runs/oracle-smoke \
  --teacher oracle --max-episodes 6 --attempts 1 --max-total-steps 100
uv run --frozen scriptwob export runs/oracle-smoke exports/oracle-smoke --allow-oracle
```

Oracle trajectories use the private interpreter and DOM, and are labeled
`privileged_oracle`. They are reference supervision, not frontier demonstrations;
export requires the explicit flag. Mock API and oracle browser tests do not
establish frontier-model accuracy.
