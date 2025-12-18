from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import soundfile as sf

from .config import repo_root
from .level_features import compute_peak_rms_dbfs

SUPPORTED_EXT = {".wav", ".aif", ".aiff", ".flac"}
CSV_COLUMNS = [
    "rel_path",
    "folder",
    "file_name",
    "duration_sec",
    "sr",
    "channels",
    "peak_dbfs",
    "rms_dbfs",
    "crest_db",
    "spec_centroid_mean",
    "spec_spread_mean",
    "spec_flatness_mean",
    "mfcc01_mean",
    "mfcc02_mean",
    "mfcc03_mean",
    "mfcc04_mean",
    "mfcc05_mean",
]


def find_audio_files(root_dir: Path, filter_prefix: Optional[str], limit: Optional[int]) -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("."):
                continue
            path = Path(dirpath) / fname
            if path.suffix.lower() not in SUPPORTED_EXT:
                continue
            rel = path.relative_to(root_dir).as_posix()
            if filter_prefix and not rel.startswith(filter_prefix):
                continue
            files.append(path)
    files.sort(key=lambda p: p.relative_to(root_dir).as_posix())
    if limit:
        files = files[:limit]
    return files


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, timeout: Optional[float] = None) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nstdout: {result.stdout}\nstderr: {result.stderr}")


def read_audio_info(path: Path) -> Dict[str, Optional[float]]:
    try:
        info = sf.info(path)
        dur = float(info.frames / info.samplerate) if info.samplerate else None
        return {
            "duration_sec": dur,
            "sr": float(info.samplerate) if info.samplerate else None,
            "channels": float(info.channels) if info.channels else None,
        }
    except Exception as exc:
        raise RuntimeError(f"Could not read headers: {path}") from exc


def parse_stats_wav(stats_wav: Path) -> Dict[str, List[float]]:
    data, _ = sf.read(stats_wav, always_2d=True)
    if data.shape[0] < 7:
        raise RuntimeError(f"Stats output too short: {stats_wav}")
    names = ["mean", "std", "skewness", "kurtosis", "low", "middle", "high"]
    out: Dict[str, List[float]] = {}
    for idx, name in enumerate(names):
        out[name] = data[idx, :].tolist()
    return out


def stats_value(stats: Dict[str, List[float]], name: str, idx: int) -> float:
    return float(stats.get(name, [])[idx])


def extract_file(
    audio_path: Path,
    root_dir: Path,
    flucoma_bin: Path,
    tmp_dir: Path,
    timeout: float,
) -> Dict[str, float]:
    rel_path = audio_path.relative_to(root_dir).as_posix()
    folder = rel_path.split("/")[0] if "/" in rel_path else ""
    file_name = audio_path.name

    info = read_audio_info(audio_path)
    level = compute_peak_rms_dbfs(audio_path)

    # spectralshape
    spec_feat = tmp_dir / "spec.wav"
    spec_stats = tmp_dir / "spec_stats.wav"
    run_cmd([
        str(flucoma_bin / "fluid-spectralshape"),
        "-source",
        str(audio_path),
        "-features",
        str(spec_feat),
        "-startchan",
        "0",
        "-numchans",
        "1",
    ], timeout=timeout)
    run_cmd([
        str(flucoma_bin / "fluid-stats"),
        "-source",
        str(spec_feat),
        "-stats",
        str(spec_stats),
    ], timeout=timeout)
    spec = parse_stats_wav(spec_stats)

    # mfcc
    mfcc_feat = tmp_dir / "mfcc.wav"
    mfcc_stats = tmp_dir / "mfcc_stats.wav"
    run_cmd([
        str(flucoma_bin / "fluid-mfcc"),
        "-source",
        str(audio_path),
        "-features",
        str(mfcc_feat),
        "-startchan",
        "0",
        "-numchans",
        "1",
    ], timeout=timeout)
    run_cmd([
        str(flucoma_bin / "fluid-stats"),
        "-source",
        str(mfcc_feat),
        "-stats",
        str(mfcc_stats),
    ], timeout=timeout)
    mfcc = parse_stats_wav(mfcc_stats)
    mfcc_means = mfcc.get("mean", [])
    if len(mfcc_means) < 5:
        raise RuntimeError(f"MFCC output has fewer than 5 coefficients: {audio_path}")

    row = {
        "rel_path": rel_path,
        "folder": folder,
        "file_name": file_name,
        **info,
        **level,
        "spec_centroid_mean": stats_value(spec, "mean", 0),
        "spec_spread_mean": stats_value(spec, "mean", 1),
        "spec_flatness_mean": stats_value(spec, "mean", 5),
        "mfcc01_mean": float(mfcc_means[0]),
        "mfcc02_mean": float(mfcc_means[1]),
        "mfcc03_mean": float(mfcc_means[2]),
        "mfcc04_mean": float(mfcc_means[3]),
        "mfcc05_mean": float(mfcc_means[4]),
    }
    return row


def extract_samples_csv(
    root_dir: Path,
    flucoma_bin_dir: Path,
    out_csv: Path,
    jobs: int = 4,
    overwrite: bool = True,
    limit: Optional[int] = None,
    filter_prefix: Optional[str] = None,
) -> None:
    if not root_dir.exists():
        raise RuntimeError(f"samples root does not exist: {root_dir}")
    if not flucoma_bin_dir.exists():
        raise RuntimeError(f"flucoma_bin_dir does not exist: {flucoma_bin_dir}")
    if out_csv.exists() and not overwrite:
        raise RuntimeError(f"Output already exists (use --overwrite to replace): {out_csv}")

    files = find_audio_files(root_dir, filter_prefix, limit)
    if not files:
        raise RuntimeError("No audio files found for extraction")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = repo_root() / "data" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "extract.log"
    fail_file = logs_dir / "extract_failures.jsonl"
    fail_file.write_text("")

    timeout = 120.0
    rows: List[Dict[str, float]] = []

    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        futures = {}
        for audio_path in files:
            tmp_dir = Path(tempfile.mkdtemp(prefix="sct_tmp_"))
            future = pool.submit(extract_file, audio_path, root_dir, flucoma_bin_dir, tmp_dir, timeout)
            futures[future] = (tmp_dir, audio_path)
        for fut in as_completed(futures):
            tmp_dir, audio_path = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
            except Exception as exc:
                with fail_file.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "file": str(audio_path),
                                "error": str(exc),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        + "\n"
                    )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    df = pd.DataFrame(rows)
    df = df[CSV_COLUMNS].sort_values("rel_path").reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    parquet_path = out_csv.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)

    with log_file.open("w", encoding="utf-8") as fh:
        fh.write(f"Processed {len(rows)} files. Output: {out_csv}\n")
