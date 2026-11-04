# scriptWoB v1 protocol

## Program semantics

The manifest contains typed program ASTs, initial worlds, layout seeds, canonical
instructions, complexity vectors, splits, and action budgets. Generation is
deterministic under the pinned implementation and seed. Provider sampling is not:
rollout seeds are bookkeeping labels, not OpenAI sampling seeds.

| Node | Meaning |
| --- | --- |
| `action` | One semantic control operation through one physical click. |
| `sequence` | Execute children in order. |
| `if` | Evaluate current state once and execute the selected branch. |
| `repeat` | Execute the body exactly N times; zero executes nothing. |
| `while` | Before each iteration, execute the body only if the predicate is true. |
| `until` | Before each iteration, execute the body only if the predicate is false. |
| `foreach` | Execute the body for each remaining queue record, then click Next. |

While/until are pre-test loops and can execute zero times. Foreach includes a
final Next after the last record; its body may not advance the same queue or
iterate over it recursively. Done is required after the program finishes.

Predicates support typed comparisons, and/or/not; generated families use a
restricted subset. Queue predicates inspect the current record. Current-record
comparisons on an exhausted queue return false, including inequality; use the
remaining count to test exhaustion. Counters stay within their bounds. Generation
checks widget/operation types, executable actions, and reference termination.

The first release is click-only: no typing, dragging, scrolling, navigation,
clipboard, timers, stochastic transitions, or external sites. The viewport is
1000×760 with deterministic card permutations and a small label vocabulary.
This is controlled, finite diversity; more seeds do not create unlimited semantics
or realistic websites.

## Observation and correctness

The policy gets instruction, screenshot, and literal past clicks. Coordinates
range from [0,1], top-left to bottom-right pixel, rounded from
`x*(width-1), y*(height-1)`. Training targets reflect executed pixel coordinates.
There is no target snapping, automatic correction, semantic action label, DOM,
program counter, reward, or correctness annotation in model input.

Button counts, counter values, checkboxes, queue position/remaining count, current
record details, visited-record markers, a neutral action count, and recent physical events are
visible. Future record details are hidden until Next. The instruction is text
context outside the screenshot. Literal history describes what happened, not the
next correct step. Explicit recall tasks are excluded, though history may still
be needed for progress through nested/repeated instructions.

Every click must match the next required semantic operation. Wrong branches,
out-of-order controls, extra iterations, background/disabled clicks, premature
Done, and budget exhaustion fail. Valid controls still show their physical effect
on a semantically wrong click. A correct eventual final state cannot erase an
order violation; no partial-credit score is awarded.

Only Python holds the program and interpreter. Browser payloads contain public
state. Built-in model policies receive no browser objects; the privileged oracle
deliberately uses them. Plugins run in-process and are trusted integrations, not
security-isolated agents.

## Complexity and splits

Levels classify structure, not guaranteed difficulty. Empty loops and short
reference traces can occur at high levels. Each complexity vector records static
node count, control/loop depth, predicate sites and expression size, operators,
widget/item counts, reference clicks **including Done**, executed branch evaluations,
and loop iterations. Evaluate by level and realized action length. To isolate
control flow, match cohorts on length and visual complexity and report empty loops
separately. Starter datasets are pilots, not evidence of training-data sufficiency.

Seed splitting groups exact task instances by a fingerprint of the instruction,
effective card order, and full displayed world semantics (including future queue
contents), ignoring internal IDs and seeds. Stable 60/20/20 hash buckets assign a
task to one split, even across differently sized manifests. Within-split repeats
remain possible because simple task families have finite diversity. Inspect reports
their unique counts. Equivalent procedures or different future queue contents can
still share an initial screenshot; this is not semantic-equivalence deduplication.

Structure splitting assigns normalized AST hashes to fixed train/dev/test buckets
(60/20/20), then samples requested counts. Shapes retain operations, control flow,
sequence arity, and predicate structure/types; they omit targets, labels, values,
and repeat counts. This is a conservative syntactic holdout, not semantic
equivalence detection. Entire families or operators can land outside training,
so this is not a controlled compositional-only generalization experiment.
Families need not be equally represented in every split.
Mixing data from another split strategy can invalidate the holdout.

## Collection and reporting

Training collection permits retries and exports at most one success per episode.
Evaluation permits exactly one attempt on dev/test. Reports distinguish first
attempts from any success after retries, show coverage, and identify infrastructure
and budget interruptions. Rates over attempted tasks are incomplete if selected
tasks remain unattempted; publish coverage and failure counts alongside scores.
Collection yield is not student evaluation accuracy.

Token totals aggregate available provider metadata. Interrupted/failed requests
can have unknown usage, so totals may not reflect all billed usage. Request limits
do not enforce a dollar budget. Refusals and invalid model outputs are retained.

Attempts record before/after PNGs, requested/executed clicks, hashes, public context,
private verdicts, response IDs, usage, and error classes. Provider exception messages
are not saved because they may contain credentials. Atomic directory commits and
file locks prevent partial successes; interrupted attempts are retained as failed.

Fingerprints bind manifest, implementation, teacher/prompt/settings, action contract,
and browser mode. Larger episode/retry limits preserve existing data; changed
behavior requires a new run directory. Chrome versions are recorded per attempt.
The Python lockfile does not pin the system browser or promise identical pixels
across platforms.

Exports verify task identity and trajectory/image continuity, then copy only
successful training images and labels. Evaluator artifacts stay out of messages.
Model reasoning is not supervised. Portable JSONL needs a model-specific collator;
SeeClick conversations follow its loader with adapted context and JSON targets.
Check processed image legibility, coordinate conventions, and token lengths in
the actual model runtime. Export compatibility does not imply a trained/evaluated
SeeClick checkpoint is included.
