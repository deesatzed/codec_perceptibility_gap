#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .proof_ladder import load_config, run_ladder
from .codec_contest import run_codec_contest
from .magnetic_pendulum import run_magnetic_slope
from .cross_family import run_cross_family
from .report import environment_summary, write_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CodecGuard / CodecBench / proof-ladder simulation bundle.")
    p.add_argument("--config", default="configs/demo.json", help="Path to JSON config.")
    p.add_argument("--out", default="results/demo", help="Output directory.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Running proof ladder with config={args.config}")
    ladder = run_ladder(cfg)
    print("Running CodecGuard/CodecBench smoke tests")
    contest = run_codec_contest(cfg)
    print("Running magnetic-pendulum nonlinear stress test")
    magnetic = run_magnetic_slope(cfg)

    results: Dict[str, Any] = {
        "config_path": args.config,
        "config": cfg,
        "environment": environment_summary(),
        "proof_ladder": ladder,
        "codec_contest": contest,
        "magnetic_pendulum": magnetic,
    }

    # Cross-family generalization (F5): run the ladder + codec contest across
    # every configured continuous source family and build a gate matrix. Only
    # runs when more than the default single family is configured.
    sources = cfg.get("sources")
    if sources and len(sources) > 1:
        print(f"Running cross-family generalization over {sources}")
        results["cross_family"] = run_cross_family(cfg)

    results_path = outdir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    report_path = write_report(results, outdir, commands=["pytest -q", f"python -m src.run_all --config {args.config} --out {args.out}"])
    print(f"Wrote {results_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
