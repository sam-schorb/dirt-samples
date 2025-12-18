# Sample Corpus Tools

## Quickstart (one command)

See `RUNBOOK.md` for the single-command workflow:
```
uv run -- sct runall --root "/path/to/Dirt-Samples" --config config/pack.yaml --apply
```
This extracts features, normalises + dedupes in place (transactional backups), re-extracts, and builds KNN artifacts. Dry-run by default if `--apply` is omitted. Convenience script: `./scripts/runall.sh ...`.

### Install
1) Install uv (https://astral.sh/uv) and verify `uv --version`.
2) Install FluCoMa CLI and set `config/config.yaml`:
```
samples_root: "/path/to/samples"
flucoma_bin_dir: "/Users/clam/tools/flucoma-cli/1.0.9/bin"
default_jobs: 4
```
3) Sync deps: `uv sync`

### Key outputs
- Features CSV: `data/features/samples.csv` (columns: rel_path, folder, file_name, duration_sec, sr, channels, peak_dbfs, rms_dbfs, crest_db, spec_centroid_mean, spec_spread_mean, spec_flatness_mean, mfcc01_mean..05_mean)
- KNN artifacts: `data/knn/{scaler.joblib,knn.joblib,rows.parquet,rows.csv,samples.csv}`
- Plans/summary: `plans/summary.txt` (+ normalize/dedupe plans and run journal)

### Other commands
- `uv run -- sct extract ...` — just feature extraction to CSV.
- `uv run -- sct pack ...` — run pack/normalize/dedupe only (with plans).
- `uv run -- sct knn query --outdir data/knn --sample "<rel_path>" --k 10` — query neighbors.

## Troubleshooting

See `docs/troubleshooting.md` for quick fixes (FluCoMa path, unreadable audio, slow runs, pack plans).
