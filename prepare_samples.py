"""
Utility script to normalise the sample pack for Strudel:
- delete Ableton .asd sidecar files
- convert AIFF/AIFFC to WAV (if a WAV with the same basename does not already exist)
- trim leading/trailing silence on all WAV files

Run from the repo root:
    python prepare_samples.py
Requires ffmpeg in PATH.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
FFMPEG = "ffmpeg"

# How aggressively to trim silence. Adjust thresholds/durations if needed.
SILENCE_FILTER = (
    "silenceremove="
    "start_periods=1:start_duration=0.005:start_threshold=-40dB:"
    "stop_periods=1:stop_duration=0.01:stop_threshold=-40dB"
)


def list_files(patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(ROOT.rglob(pattern))
    return files


def delete_asd() -> None:
    asd_files = list_files(["*.asd"])
    for f in asd_files:
        try:
            f.unlink()
        except OSError as e:
            print(f"Failed to delete {f}: {e}", file=sys.stderr)
    print(f"Removed {len(asd_files)} .asd files")


def convert_aiff_to_wav() -> None:
    aiffs = list_files(["*.aif", "*.aiff"])
    converted = 0
    skipped = 0
    for src in aiffs:
        dest = src.with_suffix(".wav")
        if dest.exists():
            skipped += 1
            continue
        cmd = [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(dest),
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Conversion failed for {src}", file=sys.stderr)
        else:
            converted += 1
    print(f"AIFF -> WAV: converted {converted}, skipped (wav exists) {skipped}")


def trim_wavs() -> None:
    wavs = list_files(["*.wav"])
    trimmed = 0
    for src in wavs:
        tmp = src.with_suffix(".tmp.wav")
        cmd = [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-af",
            SILENCE_FILTER,
            "-acodec",
            "pcm_s16le",
            str(tmp),
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Trim failed for {src}", file=sys.stderr)
            if tmp.exists():
                tmp.unlink()
            continue
        tmp.replace(src)
        trimmed += 1
    print(f"Trimmed silence on {trimmed} WAV files")


def main() -> None:
    delete_asd()
    convert_aiff_to_wav()
    trim_wavs()
    # Remove original AIFFs after successful conversions
    aiffs = list_files(["*.aif", "*.aiff"])
    for f in aiffs:
        try:
            f.unlink()
        except OSError as e:
            print(f"Failed to delete {f}: {e}", file=sys.stderr)
    if aiffs:
        print(f"Deleted {len(aiffs)} AIFF originals")


if __name__ == "__main__":
    main()
