from pathlib import Path

import pytest

pytest.importorskip("torch")

from burgers.pinn.config2 import load_phase2_config  # noqa: E402
from burgers.pinn.report import NOT_RUN, write_report  # noqa: E402

CFG = load_phase2_config(Path(__file__).resolve().parents[1] / "configs" / "phase2.yaml")
TEXT = "# Title\n\nbefore\n<!-- PHASE2:START -->\nold block\n<!-- PHASE2:END -->\nafter $x$\n"


def test_readme_block_is_idempotent_and_touches_nothing_else(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(TEXT, encoding="utf-8")
    write_report(CFG, tmp_path / "phase2", readme)
    once = readme.read_text(encoding="utf-8")
    write_report(CFG, tmp_path / "phase2", readme)
    assert readme.read_text(encoding="utf-8") == once
    assert NOT_RUN in once and "old block" not in once
    head, tail = TEXT.split("<!-- PHASE2:START -->")[0], TEXT.split("<!-- PHASE2:END -->")[1]
    assert once.startswith(head) and once.endswith(tail)
