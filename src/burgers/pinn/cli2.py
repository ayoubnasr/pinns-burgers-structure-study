"""Phase 2 command line: python -m burgers.pinn.cli2 time|gate|run|aggregate|report|smoke.

`gate` and `run` train networks and therefore first pass the pre-registration guard (runner.check_guard);
`run` also needs a finished gate unless --skip-gate is given, which is recorded in every run's metadata.
`smoke` runs the whole chain on a tiny config in a temporary folder and never touches results/phase2/.
Gotcha: tier C's 'combined' variant is resolved from summary.json, so `aggregate` must run after tiers A and B.
"""

import argparse
import copy
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

from burgers.pinn import runner
from burgers.pinn.config2 import load_phase2_config, validate


def _doc_text():
    return Path(runner.DOC).read_text(encoding="utf-8")


def cmd_gate(cfg, args):
    from burgers.pinn.aggregate import gate_check
    runner.check_guard()
    root = Path(cfg["output"]["root"])
    runner.run_variants(cfg, ["V0"], args.jobs, args.resume, log=_log)
    gate = gate_check(cfg, root)
    (root / "gate.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")
    for c in gate["clauses"]:
        print(f"  {c['text']}: observed {c['observed']:.4g} -> {'ok' if c['ok'] else 'FAILED'}")
    print(f"G0 gate: {'PASS' if gate['pass'] else 'FAIL'}")
    if not gate["pass"]:
        print("Read every tier as 'no spectral failure at this budget': the baseline does not fail where expected.")
        return 1
    return 0


def resolve_combined(cfg, root):
    """Variants merged into 'combined': explicit list in the config, or validated ones from summary.json."""
    combine = cfg["variants"]["combined"]["combine"]
    if combine != "auto":
        return list(combine)
    summary = root / "summary.json"
    if not summary.is_file():
        raise SystemExit("tier C needs results/phase2/summary.json: run tiers A and B, then 'aggregate'")
    return json.loads(summary.read_text(encoding="utf-8"))["combined_resolution"]


def cmd_run(cfg, args):
    runner.check_guard()
    root = Path(cfg["output"]["root"])
    if not (root / "gate.json").is_file() and not args.skip_gate:
        raise SystemExit("run the gate first (make phase2-gate), or pass --skip-gate (recorded in every run)")
    names = list(cfg["tiers"][args.tier])
    combine = ()
    if "pod_latent" in names:
        names.remove("pod_latent")
        print("pod_latent: skipped (NotImplementedError: underspecified in docs/PHASE2.md)")
    if "combined" in names:
        combine = resolve_combined(cfg, root)
        if not combine:
            names.remove("combined")
            print("combined: skipped (no validated variant to combine)")
        else:
            print(f"combined = {' + '.join(combine)}")
    runner.run_variants(cfg, names, args.jobs, args.resume, args.skip_gate, combine, log=_log)
    return 0


def cmd_aggregate(cfg, args):
    from burgers.pinn.aggregate import build_summary
    summary = build_summary(cfg, Path(cfg["output"]["root"]), _doc_text())
    print(f"summary: {summary['n_runs']} runs ({summary['n_failed']} failed) -> {cfg['output']['root']}/summary.json")
    return 0


def cmd_report(cfg, args):
    from burgers.pinn.report import write_report
    write_report(cfg, Path(cfg["output"]["root"]), args.readme)
    return 0


def smoke_config(cfg, root):
    """Tiny version of the real config: one nu, two seeds, small nets, few steps, coarse evaluation."""
    c = copy.deepcopy(cfg)
    c["nu_list"] = [0.1 / math.pi]
    c["seeds"] = [0, 1]
    c["net"].update(width=16, depth=2)
    c["train"].update(adam_steps=50, n_f=256, n_ic=64, n_bc=64, n_val=128, log_every=10)
    c["eval"].update(nx=513, coarse={"nx": 33, "nt": 6})
    c["bootstrap"]["n"] = 200
    c["output"]["root"] = str(root)
    return validate(c)


def cmd_smoke(cfg, args):
    from burgers.pinn.aggregate import build_summary
    from burgers.pinn.report import write_report
    root = Path(tempfile.mkdtemp(prefix="phase2_smoke_"))
    c = smoke_config(cfg, root)
    runner.run_variants(c, ["V0", "V3"], jobs=1, log=_log)
    summary = build_summary(c, root, _doc_text())
    readme = root / "README.md"
    shutil.copyfile("README.md", readme)
    write_report(c, root, readme)
    print(f"smoke: {summary['n_runs']} runs, summary, figures and README block written under {root}")
    return 0


def _log(line):
    print(line, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phase2", description="Burgers structure study, Phase 2 (PINNs).")
    parser.add_argument("command", choices=["time", "gate", "run", "aggregate", "report", "smoke"])
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--tier", choices=["A", "B", "C"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--readme", default=None, help="README whose PHASE2 block is regenerated (report)")
    args = parser.parse_args(argv)
    cfg = load_phase2_config(args.config)
    if args.command == "run" and not args.tier:
        parser.error("run needs --tier A|B|C")
    if args.command == "time":
        runner.estimate(cfg, args.jobs)
        return 0
    try:
        return {"gate": cmd_gate, "run": cmd_run, "aggregate": cmd_aggregate, "report": cmd_report,
                "smoke": cmd_smoke}[args.command](cfg, args)
    except runner.GuardError as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
