from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SIMILARITY_COLUMNS = [
    "spec_centroid_mean",
    "spec_spread_mean",
    "spec_flatness_mean",
    "mfcc01_mean",
    "mfcc02_mean",
    "mfcc03_mean",
    "mfcc04_mean",
    "mfcc05_mean",
]


class KNNArtifactsMissing(Exception):
    pass


class SampleNotFound(Exception):
    pass


class MissingColumns(Exception):
    def __init__(self, missing: List[str]):
        super().__init__("Missing required columns: " + ", ".join(missing))
        self.missing = missing


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise MissingColumns(missing)


def build_knn(csv_path: Path, outdir: Path) -> Tuple[int, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    _ensure_columns(df, ["rel_path", *SIMILARITY_COLUMNS])

    features = df[SIMILARITY_COLUMNS].astype(float).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    knn = NearestNeighbors(metric="euclidean")
    knn.fit(X_scaled)

    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, outdir / "scaler.joblib")
    joblib.dump(knn, outdir / "knn.joblib")

    rows_df = df[["rel_path", *SIMILARITY_COLUMNS]].copy()
    rows_df.to_parquet(outdir / "rows.parquet", index=False)
    rows_df.to_csv(outdir / "rows.csv", index=False)
    df.to_csv(outdir / "samples.csv", index=False)

    return len(df), len(SIMILARITY_COLUMNS)


def _load_artifacts(outdir: Path):
    scaler_path = outdir / "scaler.joblib"
    knn_path = outdir / "knn.joblib"
    rows_path = outdir / "rows.parquet"
    for p in [scaler_path, knn_path, rows_path]:
        if not p.exists():
            raise KNNArtifactsMissing(f"Missing artifact: {p}")
    scaler = joblib.load(scaler_path)
    knn = joblib.load(knn_path)
    rows_df = pd.read_parquet(rows_path)
    _ensure_columns(rows_df, ["rel_path", *SIMILARITY_COLUMNS])
    return scaler, knn, rows_df


def query_knn(outdir: Path, sample_rel_path: str, k: int = 10) -> List[Tuple[float, str]]:
    scaler, knn, rows_df = _load_artifacts(outdir)

    if sample_rel_path not in set(rows_df["rel_path"]):
        raise FileNotFoundError(f"Sample not found in rows: {sample_rel_path}")

    row = rows_df.loc[rows_df["rel_path"] == sample_rel_path].iloc[0]
    vec = pd.DataFrame([row[SIMILARITY_COLUMNS].astype(float).fillna(0.0)])
    vec_scaled = scaler.transform(vec)

    distances, indices = knn.kneighbors(vec_scaled, n_neighbors=min(k + 1, len(rows_df)))
    distances = distances[0]
    indices = indices[0]

    results: List[Tuple[float, str]] = []
    for dist, idx in zip(distances, indices):
        rel = str(rows_df.iloc[idx]["rel_path"])
        if rel == sample_rel_path:
            continue
        results.append((float(dist), rel))
        if len(results) >= k:
            break
    return results
