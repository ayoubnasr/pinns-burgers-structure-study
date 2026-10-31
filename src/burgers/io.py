"""Run directories and the summary file.

Each run gets its own folder `results/<YYYYmmdd-HHMMSS>-<git short hash>/` holding a copy of the
config used, the data, the figures and `summary.json` (every number the README may quote).
Old runs are never overwritten: if the folder name already exists, a numeric suffix is added.
Gotcha: later steps (spectrum, pod, report) work on the most recent run folder unless told otherwise.
"""

import json
import subprocess
import time
from pathlib import Path


def git_hash():
    """Short git commit hash of the working tree, with '-dirty' if there are uncommitted changes."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "nogit"
    return h.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def new_run_dir(cfg):
    """Create a fresh run folder under cfg.output_root and copy the config text into it."""
    root = Path(cfg.output_root)
    base = f"{time.strftime('%Y%m%d-%H%M%S')}-{git_hash()}"
    run_dir, n = root / base, 0
    while run_dir.exists():
        n += 1
        run_dir = root / f"{base}-{n}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(cfg.raw_text, encoding="utf-8")
    update_summary(run_dir, {"run_id": run_dir.name, "git_hash": git_hash(), "config": cfg.raw_text})
    return run_dir


def latest_run_dir(root):
    """Most recent run folder (names start with a sortable timestamp)."""
    runs = sorted(p for p in Path(root).iterdir() if (p / "summary.json").is_file()) if Path(root).is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no run found under {root}; run the 'data' step first")
    return runs[-1]


def read_summary(run_dir):
    path = Path(run_dir) / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def update_summary(run_dir, entries):
    """Merge `entries` into the run's summary.json."""
    summary = read_summary(run_dir)
    summary.update(entries)
    (Path(run_dir) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
