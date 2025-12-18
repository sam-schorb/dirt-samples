# Run everything (extract → pack → extract → KNN)

```
uv run -- sct runall --root "/path/to/Dirt-Samples" --config config/pack.yaml --apply
```

What it does
- Extracts features to `data/features/samples.csv`.
- Normalises per-folder to target RMS with a peak ceiling, dedupes folders >20 samples down to 20 (k-means), moving pruned files aside.
- Re-extracts the CSV after changes.
- Builds KNN artifacts in `data/knn/`.

Outputs
- Summary + plans: `plans/summary.txt`, `plans/normalize_plan.csv`, `plans/dedupe_plan.csv`, `plans/failures.jsonl`, `plans/run_journal.jsonl`.
- Features: `data/features/samples.csv` (and .parquet copy).
- KNN: `data/knn/{scaler.joblib,knn.joblib,rows.parquet,rows.csv,samples.csv}`.

Reruns
- If you add or change samples, rerun the same command.

Backup behavior
- Backups are temporary inside `.pack_tmp_backup` and `.pack_tmp_pruned`.
- On success they are deleted after commit.
- On failure the tool rolls the library back to the pre-run state, then deletes the temp folders.

Convenience script
```
./scripts/runall.sh --root "/path/to/Dirt-Samples" --config config/pack.yaml --apply
```
