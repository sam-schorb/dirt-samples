# Troubleshooting

- FluCoMa binaries not found: set `flucoma_bin_dir` in `config/config.yaml` to the directory that contains `fluid-*` executables, or run `uv run -- sct extract --doctor` to verify paths.
- Audio read errors: some files may be unsupported; errors are logged to `data/logs/extract_failures.jsonl`.
- Slow extraction: reduce `--jobs` or limit with `--limit`.
- KNN query says artifacts missing: re-run `uv run -- sct knn build --csv data/features/samples.csv --outdir data/knn`.
- Pack dry-run/apply: plans are written to `plans/`; temporary backups live in `.pack_tmp_backup/`; pruned files land in `_pruned/` after commit.
- Runall: if something fails, see `plans/summary.txt` (status will say ROLLED BACK) and `plans/run_journal.jsonl` for what was touched.
