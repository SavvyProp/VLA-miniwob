# Pilot protocol: independent-panels-v1

This implements the initial environment, validation, collection, and data stages
of the supplied research design v1.1 dated 31 October 2026. The PDF is design input;
it does not imply an implementation, collected dataset, or experimental result.

## Task adaptation and reuse

MiniWoB++ is pinned in `pyproject.toml` and `uv.lock`. Its Gymnasium environment,
Chrome/Selenium primitives, screenshot code, task RNG, assets, and constituent
goal functions are reused. The wrapper serves those installed assets unchanged.
The seven constituent tasks are button, link, dialog, checkboxes, radio option,
widget, and button sequence. Widget tasks require clicking even when the widget
is a text input; this is not text entry.

The six composition definitions follow the named CompWoB source files listed in
`FAMILIES` in `tasks.py`, pinned to the Google Research revision in
`configs/upstream_sources.json`. We inspected the legacy HTML and integration
instructions, then replaced the assembly with independent native-task panels.
This changes the original shared page layout, distractors, reward wiring, and
some linguistic phrasing. It allows genuine reversed semantic orders without
revealing the next component by layout. It is not a drop-in port of the original
CompWoB HTML or a result comparable to its 50-task leaderboard.

All selected components are independent. The permitted order set is forward and
reverse, or just forward for singleton tasks. `reverse_clauses` changes phrasing
while preserving the selected semantic order. Orders are enumerated conditions,
not sampled arbitrary permutations; the recorded order seed is an identity seed.
The values seed also controls geometry internal to each unchanged native task.
The separate layout seed only shuffles the outer panels. Stable SHA-256-derived
seeds are independent of Python's process-randomized hash function.

## Observation/action contract

The screenshot is 720×540 RGB, device scale factor 1, top-left origin, no policy
crop or resizing. Seven 160×210 task surfaces fit in a four-column grid.
The screenshot shows each native subtask instruction and static panel labels.
The complete procedure is additionally supplied as text at every decision.
Panel labels and positions do not indicate progress. Policies never receive
native DOM/fields, raw reward, event labels, the required-order array, or goals.

The action `(x, y, c)` is finite, with x/y clipped to [0,1], mapped to
`floor(x*(width-1)+0.5)` and `floor(y*(height-1)+0.5)`. `c >= 0.5` means one full
left-button press/release after moving; otherwise move only. The button is
released after every primitive. Consecutive clicks remain consecutive clicks.
There is no nearest-target snapping, semantic repair, typing, or held button.
The stored executed vector uses the actual quantized pixel location.

The 100 ms settling delay is fixed in the manifest. A primitive counts once,
including its movement and press/release. The initial budget is 64 primitives;
success on primitive 64 is allowed. Native wall-clock task timers are disabled
so model/API latency does not consume a task deadline. Rendering/browser errors
are infrastructure failures. This timer change is part of the adaptation.

State is normalized cursor x/y plus a released-button indicator. Feedback says
only that execution occurred and whether coordinates were clipped. The history
logs executed numerical actions and that feedback, in chronological order.
No summaries, semantic progress, or oracle labels enter it. Full procedure and
full history are retained; context overflow fails explicitly.

## Goals, chronology, and corrections

Interactive-control clicks (including labels/widgets) generate a private progress
event before native handlers execute. Movement and blank-background clicks are
neutral. Partial work on a later component is already an order violation, even
if it does not finish that component. The first violation remains in the ledger.

A native terminal callback with raw reward exactly 1 completes a component.
Positive fractional checkbox reward is failure, not completion. Native errors
terminate the composition. Corrections are possible before native submission or
termination, e.g. toggling a mistaken checkbox while still working in the current
component. Completed components are absorbing: their handlers ignore subsequent
clicks, without adding a progress badge or changing the layout. There is no
reopening/undo of a completed component in this version.

The composition terminates when every component completes or one fails. Its
native aggregate reward is stored separately from constituent raw rewards and
the strict verdict. Full success requires all goals, no order violation, valid
termination within the primitive budget, and no policy/execution error. A
misordered composition can have native aggregate reward 1 and benchmark success
0. This behavior is tested in a real browser.

## Collection and evaluation

The teacher takes the exact permitted policy observations and outputs structured
mouse primitives through GPT-6 Astra. There are no macros to expand, hidden DOM
assists, task demonstrations in the prompt, or oracle retries within a decision.
Transport retry settings are recorded; provider-side sampling is not claimed to
be seedable. Recorded rollout seeds are passed to student policies and label
teacher attempts. Response model IDs and usage are retained when available;
unreturned usage during provider errors cannot be reconstructed.

Teacher H=1 is recorded separately from the H=16 training-chunk target. Student
adapters declare their H in provenance. K is fixed by the manifest, initially 1.
There is one sampled chunk per decision and no temporal averaging or private
validator selection among candidate samples.

Collection retries an instance up to its fixed attempt limit and accepts only
complete valid trajectories. All failures are retained. Evaluation attempts each
declared instance once. Infrastructure failures are reported separately and
excluded; model failures remain in the denominator. An interrupted attempt is
retained rather than silently overwritten. Resume fingerprints include code,
runtime, manifest, and policy. Use one writer per run directory; concurrent
collection/sharding is not implemented.

Reports average within order/wording cells, then equally across observed cells
within a task, then equally across tasks. Complete benchmark scores require full
planned coverage. Partial scores are labeled observed. Intervals resample task
templates and then UI seed clusters, keeping paired variants together. Intervals
are omitted when any task has fewer than two independent UI clusters. The small
checked-in pilot is for integration, not precise scientific confidence intervals.

## Data boundary

Each primitive has its pre-action image/state, requested vector, decoded command,
literal execution result, settled next image/state, and terminal flag. The latter
observation is stored for analysis but excluded from the policy sample for that
action. Evaluator/identity files are separate from public episode data.

HDF5 contains only current policy inputs and raw executed numerical targets.
`language_per_timestep` includes past actions only; `language_raw` is the original
procedure. `sim=True` prevents upstream robotics loaders from shifting targets
back one timestep. The supplied `ChunkDataset` is the supported loader: H targets
stay within one episode, the terminal action is valid, and all later positions
are padded/masked. Training-only empirical min/max maps actions to [-1,1];
constant channels use physical [0,1] bounds. State uses training mean/std with a
0.01 floor. Statistics are reused unchanged for dev/eval and inference.

## Limits remaining before a research release

- No live teacher data or fine-tuned policy is bundled. Tests include unit fixtures
  and a privileged browser test oracle, explicitly excluded from policy baselines.
- Native random layouts can overlap controls. The executor does not fix them.
  Coverage and failure inspection must identify genuinely inaccessible instances
  and freeze a handling policy before a reported collection run.
- TinyVLA's image resizing may make a multi-panel screen unreadable. Audit the
  actual backbone processor and visual-token budget before training; cropping,
  tiling, higher resolution, or smaller family subsets change the condition.
- A frozen Chrome binary/font environment is still needed for cross-machine
  pixel reproducibility. Browser versions are recorded, not downloaded/pinned by
  this repository. A fixed settle delay does not guarantee arbitrary future
  animated tasks are stable; only the declared pilot tasks are supported.
- Full training orchestration, balanced experimental mixtures, learning curves,
  training checkpoints, richer actions, and additional original CompWoB tasks
  remain follow-up work.
