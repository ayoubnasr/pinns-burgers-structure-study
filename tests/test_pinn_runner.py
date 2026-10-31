import copy
import json
import math
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("torch")

from burgers.pinn import runner  # noqa: E402
from burgers.pinn.config2 import load_phase2_config, validate  # noqa: E402

CFG = load_phase2_config(Path(__file__).resolve().parents[1] / "configs" / "phase2.yaml")
NU = 0.1 / math.pi


def tiny(root, **train):
    c = copy.deepcopy(CFG)
    c.update(nu_list=[NU], seeds=[0, 1])
    c["net"].update(width=8, depth=2)
    c["train"].update(adam_steps=5, n_f=64, n_ic=16, n_bc=16, n_val=32, log_every=5, **train)
    c["eval"].update(nx=257, coarse={"nx": 17, "nt": 3})
    c["output"]["root"] = str(root)
    return validate(c)


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=Ayoub Nasr", "-c", "user.email=ayoubnasrsmain@gmail.com", *args],
                   cwd=repo, check=True, capture_output=True)


def test_guard_needs_tag_and_unchanged_doc(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "docs").mkdir()
    doc = tmp_path / "docs" / "PHASE2.md"
    doc.write_text("predictions\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "doc")
    with pytest.raises(runner.GuardError, match="missing"):
        runner.check_guard(tmp_path)
    _git(tmp_path, "tag", "-a", runner.TAG, "-m", "t")
    runner.check_guard(tmp_path)
    doc.write_text("predictions, edited later\n", encoding="utf-8")
    with pytest.raises(runner.GuardError, match="differs"):
        runner.check_guard(tmp_path)


def test_lr_search_layout_and_resume(tmp_path):
    cfg = tiny(tmp_path)
    lines = []
    runner.run_variants(cfg, ["V0"], log=lines.append)
    assert len(lines) == len(cfg["train"]["lr_grid"]) + 1  # lr search for seed 0, then seed 1
    tag = runner.nu_tag(NU)
    assert sorted(p.name for p in (tmp_path / "runs" / "V0" / tag).glob("*.json")) == ["seed0.json", "seed1.json"]
    assert len(list((tmp_path / "lr_search" / "V0" / tag).glob("*.json"))) == len(cfg["train"]["lr_grid"])
    seed0 = json.loads((tmp_path / "runs" / "V0" / tag / "seed0.json").read_text(encoding="utf-8"))
    assert seed0["selected_from_lr_search"] and seed0["lr"] in seed0["lr_candidates"] or str(seed0["lr"]) in seed0["lr_candidates"]
    with pytest.raises(FileExistsError):
        runner.run_variants(cfg, ["V0"], log=lines.append)
    again = []
    runner.run_variants(cfg, ["V0"], resume=True, log=again.append)
    assert again == []


def test_failed_runs_are_recorded(tmp_path):
    cfg = tiny(tmp_path, divergence_bound=1e-12)
    runner.run_variants(cfg, ["V0"], log=lambda _: None)
    for seed in (0, 1):
        meta = json.loads((tmp_path / "runs" / "V0" / runner.nu_tag(NU) / f"seed{seed}.json").read_text(encoding="utf-8"))
        assert meta["failed"] and meta["reason"]
