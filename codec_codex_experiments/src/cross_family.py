#!/usr/bin/env python3
"""Cross-family orchestration (F5 mitigation, Phase 5).

Runs the proof ladder and codec contest across every configured continuous
source family, then builds a gate matrix marking which gates are family-robust
(PASS on >=2 families) versus family-specific.

Categorical families (e.g. magnetic) keep their own dedicated runner
(`run_magnetic_slope`) because their targets are labels, not continuous theta;
they are surfaced in the same results bundle but not forced through the
continuous nRMSE ladder. No gate threshold is tuned per family: a family FAIL
is reported as an honest result.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .proof_ladder import run_ladder, normalize_cfg
from .codec_contest import run_codec_contest
from .sources import resolve_sources


LADDER_GATES = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]


def run_cross_family(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = normalize_cfg(raw_cfg)
    families = resolve_sources(cfg)

    by_source: Dict[str, Dict[str, Any]] = {}
    for fam in families:
        if getattr(fam, "target_kind", "continuous") != "continuous":
            # Categorical families are reported via their own runner elsewhere.
            continue
        ladder = run_ladder(cfg, source=fam)
        contest = run_codec_contest(cfg, source=fam)
        by_source[fam.key] = {"proof_ladder": ladder, "codec_contest": contest}

    matrix = _gate_matrix(by_source)
    return {
        "families_run": list(by_source.keys()),
        "by_source": by_source,
        "gate_matrix": matrix,
        "interpretation": (
            "family_robust gates PASS on >=2 families; family_specific gates "
            "PASS on only one. A FAIL everywhere is an honest negative, not a "
            "threshold to tune."
        ),
    }


def _gate_matrix(by_source: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = list(by_source.keys())
    per_gate: Dict[str, Dict[str, bool]] = {}
    for gate in LADDER_GATES:
        per_gate[gate] = {
            k: bool(by_source[k]["proof_ladder"].get(gate, {}).get("passed", False)) for k in keys
        }
    robust: List[str] = []
    specific: List[str] = []
    failing: List[str] = []
    for gate, results in per_gate.items():
        n_pass = sum(1 for v in results.values() if v)
        if n_pass >= 2:
            robust.append(gate)
        elif n_pass == 1:
            specific.append(gate)
        else:
            failing.append(gate)
    return {
        "per_gate_pass_by_family": per_gate,
        "family_robust_gates": robust,
        "family_specific_gates": specific,
        "failing_on_all_families": failing,
    }
