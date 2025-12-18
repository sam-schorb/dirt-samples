from __future__ import annotations

from pathlib import Path
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config() -> dict:
    root = repo_root()
    for path in [root / "config" / "config.yaml", root / "config" / "config.example.yaml"]:
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
    return {}


def cfg_value(cfg: dict, key: str, default=None):
    return cfg.get(key, default) if isinstance(cfg, dict) else default
