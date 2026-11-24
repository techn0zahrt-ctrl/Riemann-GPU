"""
NAS filesystem wrapper — direct pathlib I/O on the CIFS mount at /mnt/math.
The share is mounted on all 4 nodes via WSL2 (sudo mount -t cifs ...).
"""

import json
import os
import time
from pathlib import Path

MOUNT    = Path("/mnt/math")
RH_ROOT  = MOUNT / "rh"

DEFAULT_CONFIG = {
    "T_start":  3_000_000_000_000.0,
    "T_end":    3_001_072_620_000.0,
    "dt":       0.01,
    "chunk_dt": 5_000.0,
    "note": "Extend RH verification beyond ~10^13 known zeros. Target: ~4.6B new zeros.",
}


def rh_path(*parts) -> Path:
    return RH_ROOT.joinpath(*parts)


def run_root(cfg: dict | None = None) -> Path:
    """Directory holding done/, active/ and summary.json for the current run.

    Legacy configs (no "run" key) use /mnt/math/rh directly, so the original
    10M-chunk verification run keeps working.  A config with "run": "<name>"
    puts everything under /mnt/math/rh/runs/<name>/ so runs never collide.
    """
    if cfg is None:
        try:
            cfg = read_config()
        except (FileNotFoundError, ValueError):
            cfg = {}
    run = cfg.get("run")
    return rh_path("runs", run) if run else RH_ROOT


def done_dir(cfg: dict | None = None) -> Path:
    return run_root(cfg) / "done"


def active_dir(cfg: dict | None = None) -> Path:
    return run_root(cfg) / "active"


def ensure_rh_dirs(cfg: dict | None = None):
    for d in ("queue", "logs"):
        rh_path(d).mkdir(parents=True, exist_ok=True)
    done_dir(cfg).mkdir(parents=True, exist_ok=True)
    active_dir(cfg).mkdir(parents=True, exist_ok=True)


def read_config() -> dict:
    return json.loads(rh_path("config.json").read_text())


def write_config(cfg: dict):
    RH_ROOT.mkdir(parents=True, exist_ok=True)
    rh_path("config.json").write_text(json.dumps(cfg, indent=2))


def ensure_default_config():
    if not rh_path("config.json").exists():
        write_config(DEFAULT_CONFIG)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path):
    # utf-8-sig strips BOM if present (written by older PowerShell Set-Content)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def nas_status() -> bool:
    return MOUNT.exists() and MOUNT.is_mount()
