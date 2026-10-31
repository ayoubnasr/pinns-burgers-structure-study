import math
from pathlib import Path

import pytest

from burgers.pinn.config2 import check_against_doc, doc_predictions, load_phase2_config, parse_nu

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "PHASE2.md").read_text(encoding="utf-8")


def test_nu_values_match_phase1():
    assert math.isclose(parse_nu("0.01/pi"), 0.01 / math.pi)
    cfg = load_phase2_config(ROOT / "configs" / "phase2.yaml")
    assert all(math.isclose(a, b) for a, b in zip(cfg["nu_list"], [0.1 / math.pi, 0.01 / math.pi, 0.001 / math.pi]))


def test_every_prediction_in_doc_has_a_config_block():
    cfg = load_phase2_config(ROOT / "configs" / "phase2.yaml")
    ids = set(doc_predictions(DOC))
    assert ids == {"G0"} | {f"H{i}" for i in range(1, 9)}
    assert ids == set(cfg["predictions"])
    check_against_doc(cfg, DOC)  # thresholds agree


def test_threshold_mismatch_aborts():
    cfg = load_phase2_config(ROOT / "configs" / "phase2.yaml")
    cfg["predictions"]["H5"][0]["threshold"] = 0.6
    with pytest.raises(ValueError, match="H5"):
        check_against_doc(cfg, DOC)
