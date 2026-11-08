"""Export successful and failed example replays from saved SeeClick evaluations.

Run from the repository root with .venv/bin/python. Requires ffmpeg on PATH.
The videos use illustrative timing; they are not original screen recordings.
Use --only-failures to retain existing successful artifacts and render failures.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scriptwob.seeclick_prompt import DEFAULT_SYSTEM_MESSAGE, build_user_prompt

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
EXAMPLES = (
    (0, "Atomic action", "dev", "bfd7d11b3ee25bb6c1c0d25c-001"),
    (1, "Sequence", "dev", "a476695855e220293225fd7e-001"),
    (2, "Conditional", "dev", "b485bf2b953d34d272b1fb42-001"),
    (3, "Repetition and stopping", "test", "1ac4147e56df2f66bf348ac8-001"),
    (4, "Branches with loops", "dev", "4d5ceeaba964568042ca1e1b-001"),
    (5, "Nested control flow", "dev", "868cbe3f2477ffd66d6ad8da-001"),
)
FAILURE_EXAMPLES = (
    (1, "Sequence", "test", "996c65718871fae4320622fa-001"),
    (2, "Conditional", "dev", "68fbf117ff0d830ddd901963-001"),
    (3, "Repetition and stopping", "dev", "eab4ba76228e5f3bbd12e072-001"),
    (4, "Branches with loops", "dev", "749e77259133c013e671e5e1-001"),
    (5, "Nested control flow", "dev", "a96ddfd0ea58b5e56f826d9b-001"),
)
FAILURE_NOTES = {
    1: "Clicked Done at step 5 after only three of the four required Violet increments.",
    2: "Pressed Violet at step 1 although Maple was 3, so the condition required Coral.",
    3: "Clicked Done at step 4 with Maple at 6, before reaching the required value of 8.",
    4: "Approved blue Record 2 at step 3; the script required rejecting blue items.",
    5: ("Pressed Amber at step 2 after one Coral toggle, "
        "before completing three inner-loop toggles."),
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
STEP_SECONDS = 1.6
FINAL_SECONDS = 2.4


def build_example(example: tuple, *, success: bool = True) -> dict:
    level, title, split, attempt = example
    source = ROOT / "runs" / f"seeclick-800-devicefix-{split}" / "attempts" / attempt
    episode = json.loads((source / "episode.json").read_text())
    verdict = json.loads((source / "private.json").read_text())
    transitions = [
        json.loads(line) for line in (source / "transitions.jsonl").read_text().splitlines()
    ]
    if verdict["success"] is not success or episode["level"] != level:
        raise ValueError("Example outcome or level differs from the selected example")
    if not transitions or not transitions[-1]["terminal"]:
        raise ValueError("Example has no terminal action")
    for index, row in enumerate(transitions):
        if row["step"] != index + 1:
            raise ValueError("Nonsequential actions")
        for key in ("image", "next_image"):
            if hashlib.sha256((source / row[key]).read_bytes()).hexdigest() != row[key + "_sha256"]:
                raise ValueError("Screenshot integrity mismatch")
        if index and transitions[index - 1]["next_image_sha256"] != row["image_sha256"]:
            raise ValueError("Discontinuous screenshots")

    output = OUTPUT / f"L{level}"
    if not success:
        output = output / "failure"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / transitions[0]["image"], output / "initial.png")
    (output / "instruction.txt").write_text(episode["instruction"] + "\n")
    prompt = (
        "SYSTEM\n" + DEFAULT_SYSTEM_MESSAGE + "\n\nUSER\n"
        + build_user_prompt(episode["instruction"], (), output / "initial.png") + "\n"
    )
    (output / "prompt.txt").write_text(prompt)
    shutil.copyfile(source / "transitions.jsonl", output / "actions.jsonl")
    metadata = {
        "level": level, "title": title, "episode_id": episode["episode_id"],
        "family": episode["family"], "split": split, "success": success,
        "clicks": len(transitions), "instruction": episode["instruction"],
        "source_attempt": str(source.relative_to(ROOT)),
        "video_kind": "screenshot replay with fixed timing and recorded click markers",
        "seconds_per_click": STEP_SECONDS, "final_frame_seconds": FINAL_SECONDS,
        "prompt_note": "First-step model-format prompt reconstructed with the exported image path",
        "selection_note": "Selected illustrative rollout, not a performance sample",
        "output_subdirectory": str(output.relative_to(OUTPUT)),
        "failure_reason": None if success else verdict["environment"].get("error"),
        "failure_explanation": None if success else FAILURE_NOTES[level],
    }
    (output / "example.json").write_text(json.dumps(metadata, indent=2) + "\n")
    width, height = episode["image_size"]
    with tempfile.TemporaryDirectory(prefix="scriptwob-replay-") as work:
        work = Path(work)
        entries = []
        filters = [
            "fps=20", f"pad={width}:{height + 70}:0:0:color=0x172538",
            f"drawtext=fontfile={FONT}:text='L{level} | {title} | Screenshot replay':"
            f"x=16:y={height + 10}:fontsize=20:fontcolor=white",
        ]
        # Copying into temporary numbered files avoids concat/filter path escaping
        # and leaves the original dataset untouched.
        for index, row in enumerate(transitions):
            image = work / f"{index:06d}.png"
            shutil.copyfile(source / row["image"], image)
            entries.extend((f"file '{image}'", f"duration {STEP_SECONDS}"))
            start, end = index * STEP_SECONDS, (index + 1) * STEP_SECONDS
            enable = f"gte(t,{start:.3f})*lt(t,{end:.3f})"
            x = int(row["action"]["x"] * (width - 1) + 0.5)
            y = int(row["action"]["y"] * (height - 1) + 0.5)
            filters.append(
                f"drawbox=x={x-10}:y={y-10}:w=20:h=20:color=red:t=3:enable='{enable}'"
            )
            filters.append(
                f"drawtext=fontfile={FONT}:text='Click {index+1}/{len(transitions)} - "
                f"red box marks the recorded target':x=16:y={height+40}:"
                f"fontsize=18:fontcolor=white:enable='{enable}'"
            )
        final = work / f"{len(transitions):06d}.png"
        shutil.copyfile(source / transitions[-1]["next_image"], final)
        entries.extend((f"file '{final}'", f"duration {FINAL_SECONDS}", f"file '{final}'"))
        end_label = "Task completed successfully" if success else "Unsuccessful attempt"
        filters.append(
            f"drawtext=fontfile={FONT}:text='{end_label}':x=16:y={height+40}:"
            f"fontsize=18:fontcolor=white:enable='gte(t,{len(transitions)*STEP_SECONDS:.3f})'"
        )
        concat = work / "frames.txt"
        concat.write_text("\n".join(entries) + "\n")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat), "-vf", ",".join(filters),
            "-t", str(len(transitions) * STEP_SECONDS + FINAL_SECONDS),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "replay.mp4"),
        ], check=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "render_rollout_gif.py"),
        str(source), str(output / "replay.gif"), "--overwrite",
    ], check=True, capture_output=True, text=True)
    return metadata


def render_card(example: dict) -> str:
    level = example["level"]
    success = example["success"]
    path = example.get("output_subdirectory", f"L{level}")
    outcome = "Successful" if success else "Unsuccessful"
    note = (
        "This example uses nested counts on the same button: twelve Coral presses, then Done."
        if level == 5 and success else example.get("failure_explanation") or ""
    )
    return f"""<article>
<h3>{outcome} example</h3>
<p>{html.escape(example['family'])} · {example['split']} · {example['clicks']} clicks</p>
<p><strong>{html.escape(note)}</strong></p>
<blockquote>{html.escape(example['instruction'])}</blockquote>
<video controls preload="metadata" poster="{path}/initial.png" src="{path}/replay.mp4"></video>
<details><summary>GIF with task instruction</summary>
<img loading="lazy" src="{path}/replay.gif" style="max-width:100%" alt="L{level} {outcome} replay">
</details>
<p><a href="{path}/replay.gif">GIF with instruction</a> ·
<a href="{path}/replay.mp4">MP4 replay</a> ·
<a href="{path}/instruction.txt">Exact task instruction</a> ·
<a href="{path}/prompt.txt">First-step model-format prompt</a> ·
<a href="{path}/example.json">Source details</a></p>
</article>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-failures", action="store_true")
    args = parser.parse_args()
    jobs = [(example, False) for example in FAILURE_EXAMPLES]
    if not args.only_failures:
        jobs = [(example, True) for example in EXAMPLES] + jobs
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(build_example, example, success=success)
                   for example, success in jobs]
        for future in futures:
            future.result()
    examples, cards = [], []
    for level, title, _, _ in EXAMPLES:
        cards.append(f'<section id="L{level}"><h2>L{level}: {html.escape(title)}</h2>')
        for success in (True, False):
            path = OUTPUT / f"L{level}"
            if not success:
                path = path / "failure"
            metadata_path = path / "example.json"
            if metadata_path.exists():
                example = json.loads(metadata_path.read_text())
                examples.append(example)
                cards.append(render_card(example))
        if level == 0:
            cards.append(
                "<p>No unsuccessful L0 trial is available: all 34 dev/test tasks passed.</p>"
            )
        cards.append("</section>")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scriptWoB example prompts and replays</title>
<style>body{font:17px/1.55 system-ui,sans-serif;max-width:1040px;margin:40px auto;
padding:0 24px;color:#172538}h1,h2{line-height:1.2}
section{margin:44px 0;border-top:1px solid #ccd4df}
video{width:100%;max-width:1000px;background:#172538}blockquote{margin:20px 0;padding:16px;
background:#f1f4f8;border-left:4px solid #2868a5}a{color:#205c9a}</style>
<h1>scriptWoB: prompts and example replays</h1>
<p>A selected successful SeeClick rollout for each script level, plus an unsuccessful
rollout wherever one exists. These illustrate tasks and errors; they are not a
representative sample of model performance. Paired examples are different task instances.</p>
<p>Videos reconstruct saved screenshots with the recorded click locations marked in red.
Each click is shown for 1.6 seconds; this is illustrative playback, not original timing.
Successful tasks end with Done; failed trials can stop earlier. Instructions are copied
verbatim from saved episodes. Failure notes describe recorded behavior, not model reasoning.</p>
<p>The model also receives the current screenshot and executed-click history. Prompt files
reconstruct the first step using the exported image path; later steps include click history.</p>
""" + "\n".join(cards) + "</html>"
    (OUTPUT / "index.html").write_text(page)
    (OUTPUT / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    with zipfile.ZipFile(OUTPUT / "examples.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file() and path.suffix in {
                ".html", ".json", ".jsonl", ".txt", ".png", ".mp4", ".gif",
            }:
                archive.write(path, path.relative_to(OUTPUT))
    print(json.dumps({"output": str(OUTPUT), "examples": len(examples), "videos": len(examples)}))


if __name__ == "__main__":
    main()
