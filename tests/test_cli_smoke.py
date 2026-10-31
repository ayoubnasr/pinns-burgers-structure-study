import json
from pathlib import Path

import pytest
import yaml

from burgers.cli import main
from burgers.config import load_config

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
OUTPUTS = ["config.yaml", "summary.json", "table_modes_needed.md", "report.md"] + [
    f"{name}.{ext}" for name in ("fig_spectrum", "fig_singular_values", "fig_modes_vs_error") for ext in ("png", "svg")]


def _tiny_config(tmp_path, **overrides):
    d = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    d["problem"].update(nx=129, nt=21)
    d["spectrum"]["fit_k_range"] = [2, 20]
    d["pod"]["energy_ranks"] = [2, 4]
    d["output"]["root"] = str(tmp_path / "results")
    for key, value in overrides.items():
        d[key].update(value)
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(d), encoding="utf-8")
    return path


def test_default_config_loads():
    cfg = load_config(DEFAULT)
    assert cfg.nx >= 8 and len(cfg.nu_list) == 3


def test_invalid_config_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_config(_tiny_config(tmp_path, pod={"tolerances": [1.5]}))


def test_all_produces_every_output(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- SETUP:START -->\n<!-- SETUP:END -->\n<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n",
                      encoding="utf-8")
    run_dir = main(["all", "--config", str(_tiny_config(tmp_path)), "--readme", str(readme)])
    for name in OUTPUTS:
        assert (run_dir / name).is_file(), name
    assert len(list(run_dir.glob("data_nu*.npz"))) == 3
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for key in ("resolution", "spectrum", "pod", "git_hash", "config", "total_seconds"):
        assert key in summary
    text = readme.read_text(encoding="utf-8")
    assert "Observed vs predicted" in text and "fig_spectrum.png" in text
    # a second run never overwrites the first
    assert main(["data", "--config", str(_tiny_config(tmp_path))]) != run_dir
