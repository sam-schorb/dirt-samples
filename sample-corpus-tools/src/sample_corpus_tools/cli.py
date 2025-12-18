from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional, List

import typer

from .config import cfg_value, load_config, repo_root
from .extract import extract_samples_csv
from .knn import SIMILARITY_COLUMNS, build_knn, query_knn
from .pack import run_pack

app = typer.Typer(help="Sample Corpus Tools")
knn_app = typer.Typer(help="KNN build/query")
app.add_typer(knn_app, name="knn")


def _resolve_root(user_value: Optional[Path]) -> Path:
    cfg = load_config()
    root = user_value or cfg_value(cfg, "samples_root")
    if not root:
        raise typer.BadParameter("Set --root or samples_root in config/config.yaml")
    return Path(root).expanduser()


def _resolve_flucoma_bin() -> Path:
    cfg = load_config()
    path = cfg_value(cfg, "flucoma_bin_dir")
    if not path:
        raise typer.BadParameter("Set flucoma_bin_dir in config/config.yaml")
    p = Path(path).expanduser()
    if not p.exists():
        raise typer.BadParameter(f"FluCoMa bin dir not found: {p}")
    return p


def _default_jobs(user_jobs: Optional[int]) -> int:
    if user_jobs and user_jobs > 0:
        return user_jobs
    cfg = load_config()
    return int(cfg_value(cfg, "default_jobs", 4))


def _commit_pruned(temp_pruned: Path, final_pruned: Path) -> int:
    if not temp_pruned.exists():
        return 0
    moved = 0
    for src in sorted(temp_pruned.rglob("*")):
        if src.is_file():
            rel = src.relative_to(temp_pruned)
            dest = final_pruned / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            counter = 1
            while dest.exists():
                dest = dest.with_name(dest.stem + f".{counter}" + dest.suffix)
                counter += 1
            src.rename(dest)
            moved += 1
    return moved


def _rollback(root: Path, temp_backup: Path, temp_pruned: Path, journal: list) -> list:
    errors: list = []
    # Restore pruned moves first
    for entry in journal:
        if entry.get("type") != "prune_move":
            continue
        temp_path = Path(entry.get("temp_path", ""))
        dest = root / entry.get("rel_path", "")
        try:
            if temp_path.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp_path.rename(dest)
        except Exception as exc:  # pragma: no cover
            errors.append(f"restore prune {dest}: {exc}")
    # Restore backups
    for entry in journal:
        if entry.get("type") != "backup":
            continue
        backup_path = Path(entry.get("backup_path", ""))
        dest = root / entry.get("rel_path", "")
        try:
            if backup_path.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_path, dest)
        except Exception as exc:  # pragma: no cover
            errors.append(f"restore backup {dest}: {exc}")
    shutil.rmtree(temp_backup, ignore_errors=True)
    shutil.rmtree(temp_pruned, ignore_errors=True)
    return errors


def _doctor(bin_dir: Path) -> None:
    required = ["fluid-spectralshape", "fluid-mfcc", "fluid-stats"]
    missing = [exe for exe in required if not (bin_dir / exe).exists()]
    if missing:
        typer.echo(f"Missing FluCoMa executables: {', '.join(missing)}")
        raise typer.Exit(code=1)
    typer.echo("FluCoMa executables found:")
    for exe in required:
        typer.echo(f"- {bin_dir / exe}")
    raise typer.Exit(code=0)


@app.command()
def extract(
    root: Optional[Path] = typer.Option(None, help="Samples root directory"),
    out: Path = typer.Option(Path("data/features/samples.csv"), help="Output CSV path"),
    jobs: Optional[int] = typer.Option(None, help="Number of worker threads"),
    limit: Optional[int] = typer.Option(None, help="Process only the first N files"),
    filter_prefix: Optional[str] = typer.Option(None, help="Only process paths starting with this prefix"),
    overwrite: bool = typer.Option(True, help="Overwrite existing outputs"),
    doctor: bool = typer.Option(False, help="Check FluCoMa binaries and exit"),
) -> None:
    """Extract features and write samples.csv (primary artifact)."""

    root_dir = _resolve_root(root)
    flucoma_bin = _resolve_flucoma_bin()
    jobs_val = _default_jobs(jobs)

    if doctor:
        _doctor(flucoma_bin)

    extract_samples_csv(
        root_dir=root_dir,
        flucoma_bin_dir=flucoma_bin,
        out_csv=out,
        jobs=jobs_val,
        overwrite=overwrite,
        limit=limit,
        filter_prefix=filter_prefix,
    )
    typer.echo(f"Wrote {out}")


@knn_app.command("build")
def knn_build(
    csv: Path = typer.Option(Path("data/features/samples.csv"), help="Input samples CSV"),
    outdir: Path = typer.Option(Path("data/knn"), help="Output directory for KNN artifacts"),
) -> None:
    rows, cols = build_knn(csv, outdir)
    typer.echo(f"Built KNN on {rows} rows using {cols} similarity columns -> {outdir}")


@knn_app.command("query")
def knn_query(
    outdir: Path = typer.Option(Path("data/knn"), help="Directory containing KNN artifacts"),
    sample: str = typer.Option(..., help="rel_path of the sample to query"),
    k: int = typer.Option(10, help="Number of neighbors to return"),
    write_csv: Optional[Path] = typer.Option(None, help="Optional path to write query results CSV"),
) -> None:
    results = query_knn(outdir, sample_rel_path=sample, k=k)
    if write_csv:
        import pandas as pd

        df = pd.DataFrame(
            [
                {"rank": idx + 1, "distance": dist, "rel_path": rel}
                for idx, (dist, rel) in enumerate(results)
            ]
        )
        write_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(write_csv, index=False)
    typer.echo("Neighbors:")
    for idx, (dist, rel) in enumerate(results, start=1):
        typer.echo(f"{idx:02d}. {rel} (dist={dist:.4f})")


@app.command("pack")
def pack(
    csv: Path = typer.Option(Path("data/features/samples.csv"), help="Input features CSV"),
    root: Path = typer.Option(..., help="Samples root directory"),
    config: Path = typer.Option(Path("config/pack.yaml"), "--config", "-c", help="Pack config file"),
    apply: bool = typer.Option(False, "--apply", help="Apply changes (default is dry-run)"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Dry-run (default)"),
    backup_dir: str = typer.Option(".pack_backup", help="Backup directory relative to root"),
    no_backup: bool = typer.Option(False, "--no-backup", help="Disable backups when applying"),
    pruned_dir: str = typer.Option("_pruned", help="Pruned files destination relative to root"),
    jobs: int = typer.Option(4, help="Worker threads for normalization"),
    limit_folders: Optional[int] = typer.Option(None, help="Limit number of folders processed"),
    limit_files: Optional[int] = typer.Option(None, help="Limit number of files per folder"),
    filter_folder: Optional[str] = typer.Option(None, help="Process only this folder name"),
) -> None:
    """Normalize (RMS/peak) then dedupe per folder with backups and plans."""

    do_apply = apply or not dry_run
    run_pack(
        csv_path=csv,
        root_dir=root,
        cfg_path=config,
        apply=do_apply,
        backup_enabled=not no_backup,
        backup_dir_name=backup_dir,
        pruned_dir_name=pruned_dir,
        jobs=jobs,
        limit_folders=limit_folders,
        limit_files=limit_files,
        filter_folder=filter_folder,
        plan_dir=Path("plans"),
    )
    typer.echo(f"pack {'APPLIED' if do_apply else 'DRY-RUN'} complete")


@app.command("runall")
def runall(
    root: Optional[Path] = typer.Option(None, help="Samples root directory"),
    config: Path = typer.Option(Path("config/pack.yaml"), help="Pack config file"),
    jobs: Optional[int] = typer.Option(None, help="Worker threads"),
    apply: bool = typer.Option(False, "--apply", help="Apply changes (otherwise dry-run)"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Dry-run by default"),
    skip_knn: bool = typer.Option(False, help="Skip KNN build"),
    skip_pack: bool = typer.Option(False, help="Skip pack stage"),
    fail_after_pack: bool = typer.Option(False, help="Debug: fail after pack to test rollback"),
) -> None:
    """Extract -> pack (transactional) -> re-extract -> KNN build."""

    do_apply = apply
    if do_apply:
        dry_run = False

    root_dir = _resolve_root(root)
    flucoma_bin = _resolve_flucoma_bin()
    jobs_val = _default_jobs(jobs)

    plan_dir = Path("plans")
    plan_dir.mkdir(parents=True, exist_ok=True)

    tmp_backup = root_dir / ".pack_tmp_backup"
    tmp_pruned = root_dir / ".pack_tmp_pruned"
    final_pruned = root_dir / "_pruned"
    shutil.rmtree(tmp_backup, ignore_errors=True)
    shutil.rmtree(tmp_pruned, ignore_errors=True)

    summary_lines: List[str] = []
    journal: List[dict] = []
    pack_stats = {"normalized_files": 0, "deduped_folders": 0, "pruned_files": 0, "folders": 0}
    status = "UNKNOWN"
    try:
        extract_samples_csv(
            root_dir=root_dir,
            flucoma_bin_dir=flucoma_bin,
            out_csv=Path("data/features/samples.csv"),
            jobs=jobs_val,
            overwrite=True,
        )

        if not skip_pack:
            result = run_pack(
                csv_path=Path("data/features/samples.csv"),
                root_dir=root_dir,
                cfg_path=config,
                apply=do_apply,
                backup_enabled=do_apply,
                backup_dir_name=tmp_backup.name,
                pruned_dir_name=tmp_pruned.name,
                jobs=jobs_val,
                limit_folders=None,
                limit_files=None,
                filter_folder=None,
                plan_dir=plan_dir,
            )
            pack_stats = result.get("stats", pack_stats)
            journal = result.get("journal", [])
            jrnl_path = plan_dir / "run_journal.jsonl"
            with jrnl_path.open("w", encoding="utf-8") as fh:
                for entry in journal:
                    fh.write(json.dumps(entry) + "\n")
            if fail_after_pack:
                raise RuntimeError("fail_after_pack triggered")
            if do_apply:
                _commit_pruned(tmp_pruned, final_pruned)
            shutil.rmtree(tmp_backup, ignore_errors=True)
            shutil.rmtree(tmp_pruned, ignore_errors=True)

        extract_samples_csv(
            root_dir=root_dir,
            flucoma_bin_dir=flucoma_bin,
            out_csv=Path("data/features/samples.csv"),
            jobs=jobs_val,
            overwrite=True,
        )

        knn_msg = "skipped"
        if not skip_knn:
            build_knn(Path("data/features/samples.csv"), Path("data/knn"))
            knn_msg = "data/knn"
            try:
                import pandas as pd

                sample_example = pd.read_csv("data/features/samples.csv").iloc[0]["rel_path"]
            except Exception:
                sample_example = "<rel_path>"
            typer.echo(
                f"Example query: uv run -- sct knn query --outdir data/knn --sample \"{sample_example}\" --k 10"
            )

        status = "COMMITTED" if do_apply else "DRY-RUN"
        summary_lines.append(f"STATUS: {status}")
        summary_lines.append(f"normalized_files: {pack_stats.get('normalized_files', 0)}")
        summary_lines.append(f"deduped_folders: {pack_stats.get('deduped_folders', 0)}")
        summary_lines.append(f"pruned_files: {pack_stats.get('pruned_files', 0)}")
        summary_lines.append(f"features_csv: data/features/samples.csv")
        summary_lines.append(f"knn_artifacts: {knn_msg}")
        summary_lines.append(f"plans_dir: {plan_dir}")
        summary_lines.append("Backups: temporary; removed on success.")
    except Exception as exc:
        status = f"ROLLED BACK: {exc}"
        rollback_errors: List[str] = []
        if do_apply:
            rollback_errors = _rollback(root_dir, tmp_backup, tmp_pruned, journal)
        else:
            shutil.rmtree(tmp_backup, ignore_errors=True)
            shutil.rmtree(tmp_pruned, ignore_errors=True)
        summary_lines.append(f"STATUS: {status}")
        if rollback_errors:
            summary_lines.append("rollback_errors:")
            summary_lines.extend([f"- {e}" for e in rollback_errors])
        (plan_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        typer.echo(status)
        raise typer.Exit(code=1)

    (plan_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    typer.echo(status)


if __name__ == "__main__":
    app()
