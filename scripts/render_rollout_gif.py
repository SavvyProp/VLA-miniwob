#!/usr/bin/env python3
"""Compile a saved scriptWoB rollout into a GIF with its task instruction.

Requires Python 3.11+ and ffmpeg. No model, browser, GPU, or API calls are needed.
Input: an attempt directory containing episode.json, transitions.jsonl and PNGs.
Playback uses fixed timing; it is a reconstruction, not an original recording.
Uses the logged presented instruction when available, including code-style prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import tempfile
import textwrap
from pathlib import Path

LEVEL_NAMES = (
    "Atomic action", "Sequence", "Conditional", "Repetition and stopping",
    "Branches with loops", "Nested control flow",
)
DEFAULT_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG screenshot: {path}")
    return struct.unpack(">II", header[16:24])


def read_rollout(source: Path) -> tuple[dict, list[dict], tuple[int, int]]:
    episode = json.loads((source / "episode.json").read_text())
    rows = [json.loads(line) for line in (source / "transitions.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError("No recorded clicks to render")
    size = tuple(episode["image_size"])
    for index, row in enumerate(rows):
        if row["step"] != index + 1:
            raise ValueError("Recorded click sequence is not contiguous")
        if index and rows[index - 1]["next_image_sha256"] != row["image_sha256"]:
            raise ValueError("Screenshot sequence is not contiguous")
        for key in ("image", "next_image"):
            image = (source / row[key]).resolve()
            if not image.is_relative_to(source):
                raise ValueError("Screenshot path escapes the attempt directory")
            if hashlib.sha256(image.read_bytes()).hexdigest() != row[key + "_sha256"]:
                raise ValueError(f"Screenshot integrity check failed: {image.name}")
            if png_size(image) != size:
                raise ValueError("Screenshot size differs from episode metadata")
        for coordinate in ("x", "y"):
            value = row["action"][coordinate]
            if (not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not 0 <= value <= 1):
                raise ValueError("Click coordinate is not normalized to [0, 1]")
    return episode, rows, size


def render(args: argparse.Namespace) -> None:
    source, output = args.attempt.resolve(), args.output.resolve()
    if output.suffix.lower() != ".gif":
        raise ValueError("Output filename must end in .gif")
    if output.is_relative_to(source):
        raise ValueError("Choose an output outside the saved attempt directory")
    if output.exists() and not args.overwrite:
        raise ValueError("Output exists; choose a new path or pass --overwrite")
    if shutil.which("ffmpeg") is None:
        raise ValueError("ffmpeg is required; install it before running this script")
    if not args.font.is_file():
        raise ValueError("Font not found; supply a monospace TrueType font using --font")
    if (not math.isfinite(args.seconds_per_step) or args.seconds_per_step <= 0
            or not math.isfinite(args.final_seconds) or args.final_seconds <= 0
            or args.width < 320 or not 1 <= args.fps <= 30):
        raise ValueError("Durations must be positive, width >= 320, and fps between 1 and 30")

    episode, rows, (width, height) = read_rollout(source)
    instruction = episode["instruction"]
    instruction_source = "episode.json"
    calls_path = source / "provider_calls.json"
    if calls_path.is_file():
        calls = json.loads(calls_path.read_text())
        if not isinstance(calls, list):
            raise ValueError("Provider calls must be a list")
        presented = [call["presented_instruction"] for call in calls
                     if "presented_instruction" in call]
        if presented:
            if any(not isinstance(text, str) or not text.strip() for text in presented):
                raise ValueError("Invalid logged instruction")
            if len(set(presented)) != 1:
                raise ValueError("Presented instruction changed during the rollout")
            instruction = presented[0]
            instruction_source = "provider_calls.json:presented_instruction"
    level = episode["level"]
    title = f"scriptWoB L{level} - {LEVEL_NAMES[level]}"
    # Monospace glyph width is approximately 0.60 * font size. Leave extra room
    # so the longest instructions remain inside the screenshot width.
    # drawtext's glyph metrics plus line_spacing can exceed font size + spacing,
    # especially with mixed ASCII and Unicode. Reserve enough for multiline code.
    font_size, line_height = 20, 34
    wrap_width = max(20, int((width - 40) / (font_size * 0.62)))
    prompt_lines = [
        "Task instruction (as presented):"
        if instruction_source != "episode.json" else "Task instruction:"
    ]
    for paragraph in instruction.splitlines():
        prompt_lines.extend(textwrap.wrap(paragraph, width=wrap_width) or [""])
    header_height = 64 + len(prompt_lines) * line_height + 16
    footer_height = 64
    verdict_path = source / "private.json"
    verdict = json.loads(verdict_path.read_text()) if verdict_path.exists() else {}
    outcome = (
        "Success" if verdict.get("success") is True
        else "Unsuccessful attempt" if verdict.get("success") is False
        else "End of saved trajectory"
    )
    if verdict.get("success") is False:
        explanations = {
            "wrong_action": "wrong action",
            "premature_done": "Done clicked too early",
            "extra_action": "extra action",
        }
        reason = (verdict.get("environment") or {}).get("error")
        if reason in explanations:
            outcome += " - " + explanations[reason]
    duration = len(rows) * args.seconds_per_step + args.final_seconds

    with tempfile.TemporaryDirectory(prefix="scriptwob-gif-") as directory:
        work = Path(directory)
        # Controlled temporary paths keep drawtext/concat quoting independent of
        # user-selected filenames. expansion=none treats prompt text literally.
        font = work / "font.ttf"
        shutil.copyfile(args.font, font)

        def text_filter(name: str, content: str, y: int, size: int, enable: str = "") -> str:
            path = work / f"{name}.txt"
            path.write_text(content)
            result = (
                f"drawtext=fontfile={font}:textfile={path}:expansion=none:x=20:y={y}:"
                f"fontsize={size}:line_spacing=6:fontcolor=0x172538"
            )
            return result + (f":enable='{enable}'" if enable else "")

        filters = [
            f"fps={args.fps}",
            f"pad={width}:{header_height+height+footer_height}:0:{header_height}:color=white",
            text_filter("title", title, 16, 25),
            text_filter("instruction", "\n".join(prompt_lines), 60, font_size),
            text_filter("timing", "Screenshot replay | fixed timing | red box marks recorded click",
                        header_height + height + 37, 16),
        ]
        entries = []
        for index, row in enumerate(rows):
            image = work / f"{index:06d}.png"
            shutil.copyfile(source / row["image"], image)
            entries.extend((f"file '{image}'", f"duration {args.seconds_per_step}"))
            start = index * args.seconds_per_step
            end = (index + 1) * args.seconds_per_step
            enable = f"gte(t,{start:.6f})*lt(t,{end:.6f})"
            x = int(row["action"]["x"] * (width - 1) + 0.5)
            y = int(row["action"]["y"] * (height - 1) + 0.5) + header_height
            filters.append(
                f"drawbox=x={x-10}:y={y-10}:w=20:h=20:color=red:t=3:enable='{enable}'"
            )
            filters.append(text_filter(
                f"step-{index}", f"Click {index+1} / {len(rows)}",
                header_height + height + 10, 18, enable,
            ))
        final = work / "final.png"
        shutil.copyfile(source / rows[-1]["next_image"], final)
        entries.extend((f"file '{final}'", f"duration {args.final_seconds}", f"file '{final}'"))
        filters.append(text_filter(
            "end", f"{outcome} | {len(rows)} recorded clicks",
            header_height + height + 10, 18,
            f"gte(t,{len(rows)*args.seconds_per_step:.6f})",
        ))
        filters.append(f"scale={args.width}:-1:flags=lanczos")
        graph = ",".join(filters) + (
            ",split[frames][colors];[colors]palettegen[palette];"
            "[frames][palette]paletteuse=dither=none[out]"
        )
        concat = work / "frames.txt"
        concat.write_text("\n".join(entries) + "\n")
        temporary_gif = work / "replay.gif"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-filter_complex_threads", "1", "-f", "concat", "-safe", "0", "-i", str(concat),
            "-filter_complex", graph, "-map", "[out]", "-t", str(duration),
            "-loop", "0", "-threads", "2", str(temporary_gif),
        ], check=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temporary_gif, output)
    print(json.dumps({"output": str(output), "level": level, "clicks": len(rows),
                      "seconds": duration, "source": str(source),
                      "instruction_source": instruction_source}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path, help="Saved rollout attempt directory")
    parser.add_argument("output", type=Path, help="Destination .gif file")
    parser.add_argument("--seconds-per-step", type=float, default=1.6)
    parser.add_argument("--final-seconds", type=float, default=2.4)
    parser.add_argument("--width", type=int, default=900, help="Output width in pixels")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT, help="Monospace TrueType font")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        render(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
