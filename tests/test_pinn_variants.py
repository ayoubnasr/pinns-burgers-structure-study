from pathlib import Path

import pytest

pytest.importorskip("torch")

from burgers.pinn.config2 import load_phase2_config  # noqa: E402
from burgers.pinn.models import param_count  # noqa: E402
from burgers.pinn.variants import base_spec, build_spec, declared, expand, resolve  # noqa: E402

CFG = load_phase2_config(Path(__file__).resolve().parents[1] / "configs" / "phase2.yaml")
MATCHED = ["V0", "V1a", "V1b", "V2", "V3", "V4", "V5", "V6", "relu"]
TARGET = param_count(base_spec(CFG))


@pytest.mark.parametrize("name", MATCHED)
def test_parameter_matching_within_five_percent(name):
    spec, info = resolve(CFG, name)
    assert info["matched"] and abs(param_count(spec) / TARGET - 1) <= 0.05


def test_combined_is_matched_and_v7_is_exempt():
    _, info = resolve(CFG, "combined", combine=["V1a", "V2", "V3", "V4"])
    assert info["matched"]
    names = expand(CFG, ["V7"])
    assert len(names) == 9
    for name in names:
        spec, info = resolve(CFG, name)
        assert info["exempt"] and spec["width"] == int(name.split("_")[1][1:])


@pytest.mark.parametrize("name", MATCHED[1:] + expand(CFG, ["V7"]))
def test_each_variant_changes_only_its_declared_component(name):
    base, spec = base_spec(CFG), build_spec(CFG, name)
    changed = {k for k in base if k != "name" and base[k] != spec[k]}
    assert changed and changed <= declared(CFG, name)
    matched, _ = resolve(CFG, name)
    assert {k for k in base if k not in ("name", "width") and base[k] != matched[k]} <= declared(CFG, name)
