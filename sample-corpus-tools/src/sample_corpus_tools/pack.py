from __future__ import annotations

import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class PackConfig:
    dedupe_enabled: bool
    max_per_folder: int
    min_folder_size: int
    random_seed: int
    normalize_enabled: bool
    peak_ceiling_dbfs: float
    folder_targets: Dict[str, float]
    fallback_rms_dbfs: float
    use_active_rms: bool
    active_rms_floor_db: float
    similarity_columns: List[str]


DEFAULT_PACK_CONFIG = PackConfig(
    dedupe_enabled=True,
    max_per_folder=20,
    min_folder_size=21,
    random_seed=0,
    normalize_enabled=True,
    peak_ceiling_dbfs=-1.0,
    folder_targets={},
    fallback_rms_dbfs=-16.0,
    use_active_rms=True,
    active_rms_floor_db=-60.0,
    similarity_columns=[
        "spec_centroid_mean",
        "spec_spread_mean",
        "spec_flatness_mean",
        "mfcc01_mean",
        "mfcc02_mean",
        "mfcc03_mean",
        "mfcc04_mean",
        "mfcc05_mean",
    ],
)


def load_pack_config(path: Path) -> PackConfig:
    if not path.exists():
        raise FileNotFoundError(f"Pack config not found: {path}")
    data = json.loads(path.read_text()) if path.suffix.lower() == ".json" else None
    if data is None:
        data = yaml.safe_load(path.read_text()) or {}

    dedupe = data.get("dedupe", {})
    normalize = data.get("normalize", {})
    features = data.get("features", {})

    return PackConfig(
        dedupe_enabled=bool(dedupe.get("enabled", DEFAULT_PACK_CONFIG.dedupe_enabled)),
        max_per_folder=int(dedupe.get("max_per_folder", DEFAULT_PACK_CONFIG.max_per_folder)),
        min_folder_size=int(dedupe.get("min_folder_size", DEFAULT_PACK_CONFIG.min_folder_size)),
        random_seed=int(dedupe.get("random_seed", DEFAULT_PACK_CONFIG.random_seed)),
        normalize_enabled=bool(normalize.get("enabled", DEFAULT_PACK_CONFIG.normalize_enabled)),
        peak_ceiling_dbfs=float(normalize.get("peak_ceiling_dbfs", DEFAULT_PACK_CONFIG.peak_ceiling_dbfs)),
        folder_targets=dict(normalize.get("folder_targets_rms_dbfs", DEFAULT_PACK_CONFIG.folder_targets)),
        fallback_rms_dbfs=float(normalize.get("fallback_rms_dbfs", DEFAULT_PACK_CONFIG.fallback_rms_dbfs)),
        use_active_rms=bool(normalize.get("use_active_rms", DEFAULT_PACK_CONFIG.use_active_rms)),
        active_rms_floor_db=float(normalize.get("active_rms_floor_db", DEFAULT_PACK_CONFIG.active_rms_floor_db)),
        similarity_columns=list(features.get("similarity_columns", DEFAULT_PACK_CONFIG.similarity_columns)),
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}.{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def compute_levels(audio_path: Path, use_active: bool, active_floor_db: float) -> Tuple[Dict[str, float], np.ndarray, int, Optional[str]]:
    try:
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        info = sf.info(audio_path)
        subtype = getattr(info, "subtype", None)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio: {audio_path}") from exc

    mono = np.mean(data, axis=1)
    abs_mono = np.abs(mono)
    eps = 1e-12
    peak = float(abs_mono.max())

    if use_active:
        thresh = 10 ** (active_floor_db / 20.0)
        mask = abs_mono > thresh
        if np.any(mask):
            rms = float(np.sqrt(np.mean(np.square(mono[mask]))))
        else:
            rms = float(np.sqrt(np.mean(np.square(mono))))
    else:
        rms = float(np.sqrt(np.mean(np.square(mono))))

    peak_dbfs = 20.0 * math.log10(peak + eps)
    rms_dbfs = 20.0 * math.log10(rms + eps)
    crest_db = peak_dbfs - rms_dbfs

    return {"peak_dbfs": peak_dbfs, "rms_dbfs": rms_dbfs, "crest_db": crest_db}, data, int(sr), subtype


def normalize_file(
    row: Dict[str, any],
    root_dir: Path,
    cfg: PackConfig,
    target_rms_dbfs: float,
    apply: bool,
    backup_enabled: bool,
    backup_dir: Path,
    journal: Optional[List[Dict[str, str]]],
) -> Tuple[Dict[str, any], float, float]:
    rel_path = row["rel_path"]
    folder = row.get("folder", "")
    audio_path = root_dir / rel_path
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")

    levels, data, sr, subtype = compute_levels(audio_path, cfg.use_active_rms, cfg.active_rms_floor_db)
    peak_dbfs = levels["peak_dbfs"]
    rms_dbfs = levels["rms_dbfs"]

    g_rms = target_rms_dbfs - rms_dbfs
    g_peak = cfg.peak_ceiling_dbfs - peak_dbfs
    g = min(g_rms, g_peak)
    limited_by_peak = g_peak < g_rms

    multiplier = 10 ** (g / 20.0)
    data_out = data * multiplier

    max_abs = float(np.max(np.abs(data_out))) if data_out.size else 0.0
    post_clip_guard = False
    if max_abs > 0.9999:
        guard = 0.9999 / max_abs
        data_out *= guard
        post_clip_guard = True
        g += 20.0 * math.log10(guard)

    new_peak = peak_dbfs + g
    new_rms = rms_dbfs + g

    if apply:
        if backup_enabled:
            backup_path = ensure_unique_path(backup_dir / rel_path)
            ensure_dir(backup_path.parent)
            if not backup_path.exists():
                shutil.copy2(audio_path, backup_path)
                if journal is not None:
                    journal.append(
                        {"type": "backup", "rel_path": rel_path, "backup_path": str(backup_path)}
                    )
        ensure_dir(audio_path.parent)
        sf.write(audio_path, data_out, sr, subtype=subtype or None)

    plan_row = {
        "folder": folder,
        "rel_path": rel_path,
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "target_rms_dbfs": target_rms_dbfs,
        "applied_gain_db": g,
        "limited_by_peak": limited_by_peak,
        "new_peak_est_dbfs": new_peak,
        "new_rms_est_dbfs": new_rms,
        "post_clip_guard": post_clip_guard,
    }
    return plan_row, rms_dbfs, new_rms


def dedupe_folder(
    folder_df: pd.DataFrame,
    cfg: PackConfig,
    root_dir: Path,
    pruned_dir: Path,
    apply: bool,
    journal: Optional[List[Dict[str, str]]],
) -> Tuple[List[Dict[str, any]], int]:
    n_files = len(folder_df)
    plan_rows: List[Dict[str, any]] = []
    if not cfg.dedupe_enabled or n_files <= cfg.max_per_folder or n_files <= cfg.min_folder_size:
        for _, row in folder_df.iterrows():
            plan_rows.append(
                {
                    "folder": row["folder"],
                    "rel_path": row["rel_path"],
                    "action": "keep",
                    "cluster_id": None,
                    "distance_to_centroid": None,
                }
            )
        return plan_rows, n_files

    valid_df = folder_df.dropna(subset=cfg.similarity_columns)
    valid_count = len(valid_df)

    if valid_count == 0:
        for _, row in folder_df.iterrows():
            plan_rows.append(
                {
                    "folder": row["folder"],
                    "rel_path": row["rel_path"],
                    "action": "keep",
                    "cluster_id": None,
                    "distance_to_centroid": None,
                }
            )
        return plan_rows, n_files

    max_keep = cfg.max_per_folder
    n_clusters = min(max_keep, valid_count)
    features = valid_df[cfg.similarity_columns].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    rel_list = valid_df["rel_path"].tolist()
    cluster_map: Dict[str, Tuple[int, float]] = {}
    keep_set: set[str] = set()
    if n_clusters < 2:
        for idx, rel in enumerate(rel_list):
            cluster_map[rel] = (0, 0.0)
            if idx < max_keep:
                keep_set.add(rel)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=cfg.random_seed, n_init="auto")
        labels = kmeans.fit_predict(X_scaled)
        centers = kmeans.cluster_centers_
        for idx, rel in enumerate(rel_list):
            cid = int(labels[idx])
            dist = float(np.linalg.norm(X_scaled[idx] - centers[cid]))
            cluster_map[rel] = (cid, dist)
        for cid in range(n_clusters):
            cluster_idx = np.where(labels == cid)[0]
            if len(cluster_idx) == 0:
                continue
            cluster_points = X_scaled[cluster_idx]
            dists = np.linalg.norm(cluster_points - centers[cid], axis=1)
            best_local = cluster_idx[int(np.argmin(dists))]
            keep_set.add(rel_list[best_local])

    keep_list = list(keep_set)
    if len(keep_list) > max_keep:
        keep_list = keep_list[:max_keep]
    if len(keep_list) < max_keep:
        remaining = [rel for rel in rel_list if rel not in keep_set]
        for rel in remaining:
            if len(keep_list) >= max_keep:
                break
            keep_list.append(rel)
    keep_set = set(keep_list)

    moved_count = 0
    for _, row in folder_df.iterrows():
        rel = row["rel_path"]
        cid, dist = cluster_map.get(rel, (None, None))
        action = "keep" if rel in keep_set else "move"
        plan_rows.append(
            {
                "folder": row["folder"],
                "rel_path": rel,
                "action": action,
                "cluster_id": cid,
                "distance_to_centroid": dist,
            }
        )
        if action == "move":
            moved_count += 1
            if apply:
                src = root_dir / rel
                dest = ensure_unique_path(pruned_dir / rel)
                ensure_dir(dest.parent)
                shutil.move(src, dest)
                if journal is not None:
                    journal.append({"type": "prune_move", "rel_path": rel, "temp_path": str(dest)})

    count_after = n_files - moved_count
    return plan_rows, count_after


def run_pack(
    csv_path: Path,
    root_dir: Path,
    cfg_path: Path,
    apply: bool,
    backup_enabled: bool,
    backup_dir_name: str,
    pruned_dir_name: str,
    jobs: int,
    limit_folders: Optional[int],
    limit_files: Optional[int],
    filter_folder: Optional[str],
    plan_dir: Path,
) -> Dict[str, any]:
    cfg = load_pack_config(cfg_path)
    df = pd.read_csv(csv_path)
    required_cols = {"rel_path", "folder", "file_name", "peak_dbfs", "rms_dbfs"}
    missing_cols = required_cols - set(df.columns)
    sim_missing = set(cfg.similarity_columns) - set(df.columns)
    if missing_cols or sim_missing:
        raise RuntimeError(f"Missing columns: {sorted(missing_cols | sim_missing)}")

    root_dir = root_dir.resolve()
    backup_dir = root_dir / backup_dir_name
    pruned_dir = root_dir / pruned_dir_name

    plan_norm: List[Dict[str, any]] = []
    plan_dedupe: List[Dict[str, any]] = []
    summary_lines: List[str] = []
    failures: List[Dict[str, str]] = []
    journal: List[Dict[str, str]] = []

    folders = sorted(df["folder"].dropna().unique().tolist())
    if filter_folder:
        folders = [f for f in folders if f == filter_folder]
    if limit_folders:
        folders = folders[:limit_folders]

    for folder in folders:
        folder_df = df[df["folder"] == folder].reset_index(drop=True)
        if limit_files:
            folder_df = folder_df.iloc[:limit_files].copy()
        target_rms = cfg.folder_targets.get(folder, cfg.fallback_rms_dbfs)

        rms_before: List[float] = []
        rms_after: List[float] = []
        rewrites = 0

        with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
            future_map = {
                pool.submit(
                    normalize_file,
                    row.to_dict(),
                    root_dir,
                    cfg,
                    target_rms,
                    apply and cfg.normalize_enabled,
                    backup_enabled,
                    backup_dir,
                    journal,
                ): row.get("rel_path", "")
                for _, row in folder_df.iterrows()
            }
            for fut in as_completed(future_map):
                rel = future_map[fut]
                try:
                    plan_row, rms_b, rms_a = fut.result()
                    plan_norm.append(plan_row)
                    rms_before.append(rms_b)
                    rms_after.append(rms_a)
                    if apply and cfg.normalize_enabled:
                        rewrites += 1
                except Exception as exc:  # pragma: no cover
                    failures.append(
                        {
                            "rel_path": rel,
                            "stage": "normalize",
                            "error": str(exc),
                        }
                    )

        count_before = len(folder_df)
        count_after = count_before
        if cfg.dedupe_enabled:
            try:
                dedupe_rows, count_after = dedupe_folder(
                    folder_df,
                    cfg=cfg,
                    root_dir=root_dir,
                    pruned_dir=pruned_dir,
                    apply=apply,
                    journal=journal,
                )
                plan_dedupe.extend(dedupe_rows)
            except Exception as exc:  # pragma: no cover
                failures.append(
                    {
                        "rel_path": f"{folder}/*",
                        "stage": "dedupe",
                        "error": str(exc),
                    }
                )

        med_before = float(np.median(rms_before)) if rms_before else float("nan")
        med_after = float(np.median(rms_after)) if rms_after else float("nan")
        summary_lines.append(
            f"{folder}: count_before={count_before}, count_after={count_after}, "
            f"target_rms={target_rms} dBFS, median_rms_before={med_before:.2f}, "
            f"median_rms_after={med_after:.2f}, rewrites={'applied' if apply else 'planned'}={rewrites}"
        )

    ensure_dir(plan_dir)
    norm_path = plan_dir / "normalize_plan.csv"
    dedupe_path = plan_dir / "dedupe_plan.csv"
    summary_path = plan_dir / "summary.txt"
    failures_path = plan_dir / "failures.jsonl"

    pd.DataFrame(plan_norm).to_csv(norm_path, index=False)
    pd.DataFrame(plan_dedupe).to_csv(dedupe_path, index=False)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    if failures:
        with failures_path.open("w", encoding="utf-8") as fh:
            for item in failures:
                fh.write(json.dumps(item) + "\n")

    action = "APPLIED" if apply else "DRY-RUN"
    print(f"pack {action}: normalized {len(plan_norm)} files across {len(folders)} folders. Plans in {plan_dir}")

    stats = {
        "normalized_files": len(plan_norm),
        "folders": len(folders),
        "deduped_folders": len({r["folder"] for r in plan_dedupe if r.get("action") == "move"}),
        "pruned_files": len([r for r in plan_dedupe if r.get("action") == "move"]),
        "plan_dir": str(plan_dir),
    }
    return {"stats": stats, "journal": journal}


__all__ = ["run_pack", "load_pack_config", "PackConfig"]
